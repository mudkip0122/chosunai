from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.audio_dataset import AudioDataset
from src.model import build_model
from train import joint_macro_f1


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}


def standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def evaluate(checkpoint_path: Path, seed: int, data_dir: Path, features: np.ndarray) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = checkpoint["config"].get("arch", "baseline")
    frame = pd.read_csv(data_dir / "train.csv")
    key = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
    train_frame, valid_frame = train_test_split(frame, test_size=0.2, random_state=seed, stratify=key)
    class_to_idx, azimuth_to_idx = checkpoint["class_to_idx"], checkpoint["azimuth_to_idx"]
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in checkpoint["idx_to_azimuth"].items()}
    dataset = AudioDataset(
        valid_frame, data_dir / "train_audio", 16000, 2.0, 64,
        class_to_idx, azimuth_to_idx, train=False, include_gcc=arch in GCC_ARCHES,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    class_parts, angle_parts = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch[0].to(device))
            class_parts.append(output[0].cpu())
            angle_parts.append(output[1].cpu())
    class_logits = torch.cat(class_parts).float()
    neural = standardize(torch.cat(angle_parts).float())

    x = np.concatenate((features[:, :64], features[:, 65:66]), axis=1)
    train_idx, valid_idx = train_frame.index.to_numpy(), valid_frame.index.to_numpy()
    angle_degrees = frame["azimuth"].astype(float).to_numpy()
    radians = np.deg2rad(angle_degrees)
    circular_targets = np.stack((np.cos(radians), np.sin(radians)), axis=1)
    angle_targets = frame["azimuth"].map(azimuth_to_idx).to_numpy()
    class_targets = frame["sound_class"].map(class_to_idx).to_numpy()

    classifier = ExtraTreesClassifier(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        class_weight="balanced", random_state=5, n_jobs=-1,
    ).fit(x[train_idx], angle_targets[train_idx])
    class_probability = classifier.predict_proba(x[valid_idx])
    tree_logits = standardize(torch.from_numpy(np.log(np.clip(class_probability, 1e-6, 1.0))).float())

    common = dict(n_estimators=500, random_state=5, n_jobs=-1)
    regressors = {
        "et1": ExtraTreesRegressor(max_features=1.0, min_samples_leaf=1, **common),
        "et2": ExtraTreesRegressor(max_features=1.0, min_samples_leaf=2, **common),
        "et4": ExtraTreesRegressor(max_features=1.0, min_samples_leaf=4, **common),
        "rf1": RandomForestRegressor(max_features=0.7, min_samples_leaf=1, **common),
        "rf2": RandomForestRegressor(max_features=0.7, min_samples_leaf=2, **common),
    }
    candidate_angles = torch.tensor(
        [np.deg2rad(idx_to_azimuth[i]) for i in range(len(idx_to_azimuth))], dtype=torch.float32
    )
    regression_scores = {}
    for name, regressor in regressors.items():
        regressor.fit(x[train_idx], circular_targets[train_idx])
        vector = regressor.predict(x[valid_idx])
        predicted_angle = torch.from_numpy(np.arctan2(vector[:, 1], vector[:, 0])).float()
        score_matrix = torch.cos(predicted_angle[:, None] - candidate_angles[None, :])
        regression_scores[name] = standardize(score_matrix)

    true_class = class_targets[valid_idx].tolist()
    true_angle = angle_targets[valid_idx].tolist()
    class_pred = class_logits.argmax(1).tolist()

    def score(pred: torch.Tensor) -> float:
        return joint_macro_f1(true_class, true_angle, class_pred, pred.tolist(), idx_to_class, idx_to_azimuth)

    low = (neural + 0.15 * tree_logits).argmax(1)
    high = (neural + 0.30 * tree_logits).argmax(1)
    changed = low != high
    sorted_probability = np.sort(class_probability, axis=1)
    margin = torch.from_numpy(sorted_probability[:, -1] - sorted_probability[:, -2])
    threshold = torch.quantile(margin[changed], 0.5)
    baseline = low.clone()
    baseline[changed & (margin >= threshold)] = high[changed & (margin >= threshold)]
    rows = [{"seed": seed, "rule": "global_conditional", "score": score(baseline)}]

    for name, circular_score in regression_scores.items():
        for global_weight in (0.10, 0.15, 0.20):
            for circular_weight in (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30):
                prediction = (
                    neural + global_weight * tree_logits + circular_weight * circular_score
                ).argmax(1)
                rows.append({
                    "seed": seed,
                    "rule": f"global{global_weight:.3f}_{name}{circular_weight:.3f}",
                    "score": score(prediction),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("cv_runs/gcc_features.npy"))
    parser.add_argument("--out", type=Path, default=Path("cv_runs/gcc_circular_regression.csv"))
    args = parser.parse_args()
    features = np.load(args.features)
    result = pd.concat([
        evaluate(Path("checkpoints/gcc_seed5.pt"), 5, args.data_dir, features),
        evaluate(Path("checkpoints/seed17.pt"), 17, args.data_dir, features),
    ], ignore_index=True)
    pivot = result.pivot(index="rule", columns="seed", values="score")
    baseline = pivot.loc["global_conditional"]
    pivot["min_gain"] = (pivot - baseline).min(axis=1)
    pivot["mean_gain"] = (pivot - baseline).mean(axis=1)
    print(pivot.sort_values(["min_gain", "mean_gain"], ascending=False).head(25))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

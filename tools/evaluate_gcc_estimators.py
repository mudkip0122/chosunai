from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
    class_to_idx = checkpoint["class_to_idx"]
    azimuth_to_idx = checkpoint["azimuth_to_idx"]
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
    class_parts, azimuth_parts = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch[0].to(device))
            class_parts.append(outputs[0].cpu())
            azimuth_parts.append(outputs[1].cpu())
    class_logits = torch.cat(class_parts).float()
    neural = standardize(torch.cat(azimuth_parts).float())

    x = np.concatenate((features[:, :64], features[:, 65:66]), axis=1)
    train_idx, valid_idx = train_frame.index.to_numpy(), valid_frame.index.to_numpy()
    targets = frame["azimuth"].map(azimuth_to_idx).to_numpy()
    common = dict(n_estimators=500, class_weight="balanced", random_state=5, n_jobs=-1)
    estimators = {
        "et1": ExtraTreesClassifier(max_features=1.0, min_samples_leaf=1, **common),
        "et2": ExtraTreesClassifier(max_features=1.0, min_samples_leaf=2, **common),
        "et4": ExtraTreesClassifier(max_features=1.0, min_samples_leaf=4, **common),
        "rf1": RandomForestClassifier(max_features=0.7, min_samples_leaf=1, **common),
        "rf2": RandomForestClassifier(max_features=0.7, min_samples_leaf=2, **common),
        "rfsqrt": RandomForestClassifier(max_features="sqrt", min_samples_leaf=1, **common),
    }
    logs: dict[str, torch.Tensor] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, estimator in estimators.items():
        estimator.fit(x[train_idx], targets[train_idx])
        probability = estimator.predict_proba(x[valid_idx])
        probabilities[name] = probability
        logs[name] = standardize(torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0))).float())

    true_class = valid_frame["sound_class"].map(class_to_idx).tolist()
    true_azimuth = valid_frame["azimuth"].map(azimuth_to_idx).tolist()
    class_pred = class_logits.argmax(1).tolist()

    def score(pred: torch.Tensor) -> float:
        return joint_macro_f1(true_class, true_azimuth, class_pred, pred.tolist(), idx_to_class, idx_to_azimuth)

    et = logs["et1"]
    low = (neural + 0.15 * et).argmax(1)
    high = (neural + 0.30 * et).argmax(1)
    changed = low != high
    sorted_p = np.sort(probabilities["et1"], axis=1)
    margin = torch.from_numpy(sorted_p[:, -1] - sorted_p[:, -2])
    threshold = torch.quantile(margin[changed], 0.5)
    baseline = low.clone()
    baseline[changed & (margin >= threshold)] = high[changed & (margin >= threshold)]
    rows = [{"seed": seed, "rule": "et_conditional", "score": score(baseline)}]

    for name, candidate in logs.items():
        if name == "et1":
            continue
        for weight in (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
            # Add a small independent correction to the proven global ET signal.
            prediction = (neural + 0.15 * et + weight * candidate).argmax(1)
            rows.append({"seed": seed, "rule": f"et015_{name}{weight:.3f}", "score": score(prediction)})
        for mix in (0.25, 0.50, 0.75):
            blended = (1.0 - mix) * et + mix * candidate
            for weight in (0.15, 0.20, 0.25, 0.30):
                prediction = (neural + weight * blended).argmax(1)
                rows.append({"seed": seed, "rule": f"{name}_mix{mix:.2f}_w{weight:.2f}", "score": score(prediction)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("cv_runs/gcc_features.npy"))
    parser.add_argument("--out", type=Path, default=Path("cv_runs/gcc_estimators.csv"))
    args = parser.parse_args()
    features = np.load(args.features)
    result = pd.concat([
        evaluate(Path("checkpoints/gcc_seed5.pt"), 5, args.data_dir, features),
        evaluate(Path("checkpoints/seed17.pt"), 17, args.data_dir, features),
    ], ignore_index=True)
    pivot = result.pivot(index="rule", columns="seed", values="score")
    baseline = pivot.loc["et_conditional"]
    pivot["min_gain"] = (pivot - baseline).min(axis=1)
    pivot["mean_gain"] = (pivot - baseline).mean(axis=1)
    print(pivot.sort_values(["min_gain", "mean_gain"], ascending=False).head(25))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

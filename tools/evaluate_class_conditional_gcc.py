from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.audio_dataset import AudioDataset
from src.model import build_model
from train import joint_macro_f1


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}


def standardized(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std().clamp_min(1e-6)


def tree_log_proba(model: ExtraTreesClassifier, features: np.ndarray, n_angles: int) -> np.ndarray:
    probability = np.full((len(features), n_angles), 1e-6, dtype=np.float64)
    probability[:, model.classes_.astype(int)] = model.predict_proba(features)
    return np.log(np.clip(probability, 1e-6, 1.0))


def evaluate(checkpoint_path: Path, seed: int, data_dir: Path, features: np.ndarray) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    arch = config.get("arch", "baseline")
    frame = pd.read_csv(data_dir / "train.csv")
    key = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
    train_frame, valid_frame = train_test_split(
        frame, test_size=0.2, random_state=seed, stratify=key
    )

    class_to_idx = checkpoint["class_to_idx"]
    azimuth_to_idx = checkpoint["azimuth_to_idx"]
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in checkpoint["idx_to_azimuth"].items()}
    dataset = AudioDataset(
        valid_frame,
        data_dir / "train_audio",
        sample_rate=16000,
        duration=2.0,
        n_mels=64,
        class_to_idx=class_to_idx,
        azimuth_to_idx=azimuth_to_idx,
        train=False,
        include_gcc=arch in GCC_ARCHES,
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
    neural_logits = standardized(torch.cat(azimuth_parts).float())

    x_tree = np.concatenate((features[:, :64], features[:, 65:66]), axis=1)
    train_idx = train_frame.index.to_numpy()
    valid_idx = valid_frame.index.to_numpy()
    y_angle = frame["azimuth"].map(azimuth_to_idx).to_numpy()
    y_class = frame["sound_class"].map(class_to_idx).to_numpy()
    n_angles = len(azimuth_to_idx)

    common = dict(
        n_estimators=600,
        max_features=1.0,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=5,
        n_jobs=-1,
    )
    global_tree = ExtraTreesClassifier(**common).fit(x_tree[train_idx], y_angle[train_idx])
    global_log = standardized(
        torch.from_numpy(tree_log_proba(global_tree, x_tree[valid_idx], n_angles)).float()
    )

    # Each sound class gets its own spatial mapping. At validation/test time the
    # neural class prediction chooses the applicable mapping, avoiding target leakage.
    conditional_all = []
    for class_index in range(len(class_to_idx)):
        subset = train_idx[y_class[train_idx] == class_index]
        tree = ExtraTreesClassifier(**common).fit(x_tree[subset], y_angle[subset])
        conditional_all.append(tree_log_proba(tree, x_tree[valid_idx], n_angles))
    conditional_all = np.stack(conditional_all, axis=1)
    predicted_class = class_logits.argmax(1).numpy()
    conditional_log = standardized(
        torch.from_numpy(conditional_all[np.arange(len(valid_idx)), predicted_class]).float()
    )

    true_class = y_class[valid_idx].tolist()
    true_azimuth = y_angle[valid_idx].tolist()
    class_pred = predicted_class.tolist()

    def score(prediction: torch.Tensor) -> float:
        return joint_macro_f1(
            true_class,
            true_azimuth,
            class_pred,
            prediction.tolist(),
            idx_to_class,
            idx_to_azimuth,
        )

    low = (neural_logits + 0.15 * global_log).argmax(1)
    high = (neural_logits + 0.30 * global_log).argmax(1)
    changed = low != high
    probabilities = global_tree.predict_proba(x_tree[valid_idx])
    sorted_probability = np.sort(probabilities, axis=1)
    margin = torch.from_numpy(sorted_probability[:, -1] - sorted_probability[:, -2])
    threshold = torch.quantile(margin[changed], 0.50)
    baseline = low.clone()
    baseline[changed & (margin >= threshold)] = high[changed & (margin >= threshold)]

    rows = [{"seed": seed, "rule": "global_conditional", "score": score(baseline)}]
    for conditional_weight in (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30):
        for global_weight in (0.10, 0.15, 0.20):
            prediction = (
                neural_logits + global_weight * global_log + conditional_weight * conditional_log
            ).argmax(1)
            rows.append(
                {
                    "seed": seed,
                    "rule": f"global{global_weight:.3f}_class{conditional_weight:.3f}",
                    "score": score(prediction),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("cv_runs/gcc_features.npy"))
    parser.add_argument("--out", type=Path, default=Path("cv_runs/class_conditional_gcc.csv"))
    args = parser.parse_args()
    features = np.load(args.features)
    results = pd.concat(
        [
            evaluate(Path("checkpoints/gcc_seed5.pt"), 5, args.data_dir, features),
            evaluate(Path("checkpoints/seed17.pt"), 17, args.data_dir, features),
        ],
        ignore_index=True,
    )
    pivot = results.pivot(index="rule", columns="seed", values="score")
    baseline = pivot.loc["global_conditional"]
    pivot["min_gain"] = (pivot - baseline).min(axis=1)
    pivot["mean_gain"] = (pivot - baseline).mean(axis=1)
    print(pivot.sort_values(["min_gain", "mean_gain"], ascending=False).head(20))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

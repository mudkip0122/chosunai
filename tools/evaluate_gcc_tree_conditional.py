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


def infer_validation(checkpoint_path: Path, seed: int, data_dir: Path, features: np.ndarray):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    arch = config.get("arch", "baseline")
    frame = pd.read_csv(data_dir / "train.csv")
    key = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
    stratify = key if key.value_counts().min() >= 2 else frame["sound_class"]
    train_frame, valid_frame = train_test_split(
        frame, test_size=0.2, random_state=seed, stratify=stratify
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
        include_gcc=arch in {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"},
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    class_parts, azimuth_parts = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            outputs = model(x)
            class_parts.append(outputs[0].cpu())
            azimuth_parts.append(outputs[1].cpu())
    class_logits = torch.cat(class_parts).float()
    neural_logits = torch.cat(azimuth_parts).float()

    tree = ExtraTreesClassifier(
        n_estimators=600,
        max_features=1.0,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=5,
        n_jobs=-1,
    )
    x_tree = np.concatenate((features[:, :64], features[:, 65:66]), axis=1)
    tree.fit(x_tree[train_frame.index], frame.loc[train_frame.index, "azimuth"].map(azimuth_to_idx))
    probability = tree.predict_proba(x_tree[valid_frame.index])
    tree_logits = torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0))).float()

    neural_logits = (neural_logits - neural_logits.mean()) / neural_logits.std().clamp_min(1e-6)
    tree_logits = (tree_logits - tree_logits.mean()) / tree_logits.std().clamp_min(1e-6)
    true_class = valid_frame["sound_class"].map(class_to_idx).tolist()
    true_azimuth = valid_frame["azimuth"].map(azimuth_to_idx).tolist()
    class_pred = class_logits.argmax(1).tolist()

    def score(pred: np.ndarray) -> float:
        return joint_macro_f1(
            true_class,
            true_azimuth,
            class_pred,
            pred.tolist(),
            idx_to_class,
            idx_to_azimuth,
        )

    pred15 = (neural_logits + 0.15 * tree_logits).argmax(1).numpy()
    pred30 = (neural_logits + 0.30 * tree_logits).argmax(1).numpy()
    changed = pred15 != pred30
    sorted_probability = np.sort(probability, axis=1)
    diagnostics = {
        "tree_conf": sorted_probability[:, -1],
        "tree_margin": sorted_probability[:, -1] - sorted_probability[:, -2],
        "neural_margin": np.sort(torch.softmax(neural_logits, 1).numpy(), axis=1)[:, -1]
        - np.sort(torch.softmax(neural_logits, 1).numpy(), axis=1)[:, -2],
    }
    rows = [
        {"seed": seed, "rule": "w000", "score": score(neural_logits.argmax(1).numpy()), "selected": 0},
        {"seed": seed, "rule": "w015", "score": score(pred15), "selected": 0},
        {"seed": seed, "rule": "w030", "score": score(pred30), "selected": int(changed.sum())},
    ]
    for name, values in diagnostics.items():
        for quantile in (0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
            # Stronger tree correction where the tree is confident; stronger
            # correction where the neural model is uncertain.
            threshold = np.quantile(values[changed], quantile) if changed.any() else 0.0
            select = changed & ((values >= threshold) if name != "neural_margin" else (values <= threshold))
            pred = pred15.copy()
            pred[select] = pred30[select]
            rows.append(
                {
                    "seed": seed,
                    "rule": f"{name}_q{quantile:.2f}",
                    "score": score(pred),
                    "selected": int(select.sum()),
                }
            )
    quantiles = (0.25, 0.40, 0.50, 0.60, 0.70)
    for margin_q in quantiles:
        margin_threshold = np.quantile(diagnostics["tree_margin"][changed], margin_q)
        margin_select = diagnostics["tree_margin"] >= margin_threshold
        for confidence_q in quantiles:
            confidence_threshold = np.quantile(
                diagnostics["tree_conf"][changed], confidence_q
            )
            confidence_select = diagnostics["tree_conf"] >= confidence_threshold
            for operator, select_extra in (
                ("and", margin_select & confidence_select),
                ("or", margin_select | confidence_select),
            ):
                select = changed & select_extra
                pred = pred15.copy()
                pred[select] = pred30[select]
                rows.append(
                    {
                        "seed": seed,
                        "rule": f"margin_q{margin_q:.2f}_{operator}_conf_q{confidence_q:.2f}",
                        "score": score(pred),
                        "selected": int(select.sum()),
                    }
                )

        for neural_q in quantiles:
            neural_threshold = np.quantile(
                diagnostics["neural_margin"][changed], neural_q
            )
            select = changed & margin_select & (
                diagnostics["neural_margin"] <= neural_threshold
            )
            pred = pred15.copy()
            pred[select] = pred30[select]
            rows.append(
                {
                    "seed": seed,
                    "rule": f"margin_q{margin_q:.2f}_and_neural_q{neural_q:.2f}",
                    "score": score(pred),
                    "selected": int(select.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("cv_runs/gcc_features.npy"))
    parser.add_argument("--out", type=Path, default=Path("cv_runs/gcc_tree_conditional.csv"))
    args = parser.parse_args()
    features = np.load(args.features)
    results = pd.concat(
        [
            infer_validation(Path("checkpoints/gcc_seed5.pt"), 5, args.data_dir, features),
            infer_validation(Path("checkpoints/seed17.pt"), 17, args.data_dir, features),
        ],
        ignore_index=True,
    )
    pivot = results.pivot(index="rule", columns="seed", values="score")
    base = pivot.loc["w015"]
    pivot["min_gain_vs_w015"] = (pivot - base).min(axis=1)
    pivot["mean_gain_vs_w015"] = (pivot - base).mean(axis=1)
    print(pivot.sort_values(["min_gain_vs_w015", "mean_gain_vs_w015"], ascending=False).head(20))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

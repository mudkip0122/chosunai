from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from torch.utils.data import DataLoader

from src.audio_dataset import AudioDataset, invert_map, make_label_maps
from src.model import build_model
from train import joint_macro_f1


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}


def main() -> None:
    frame = pd.read_csv("data/train.csv")
    valid_indices = np.load("cv_runs/domain_valid_indices.npy").astype(int)
    valid_mask = np.zeros(len(frame), dtype=bool)
    valid_mask[valid_indices] = True
    valid = frame.loc[valid_mask]
    class_to_idx, azimuth_to_idx = make_label_maps(frame)
    idx_to_class, idx_to_azimuth = invert_map(class_to_idx), invert_map(azimuth_to_idx)
    dataset = AudioDataset(
        valid, Path("data/train_audio"), 16000, 2.0, 64,
        class_to_idx, azimuth_to_idx, train=False, include_gcc=True,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    paths = {
        "base_gcc": Path("checkpoints/gcc_domain_base_seed5.pt"),
        "robust_gcc": Path("checkpoints/gcc_domain_robust_seed5.pt"),
        "circ29": Path("checkpoints/circ29_domain.pt"),
        "angle37": Path("checkpoints/angle37_domain.pt"),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = {}
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        arch = checkpoint["config"].get("arch", "baseline")
        model = build_model(arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        class_parts, azimuth_parts = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                c, a = model(x if arch in GCC_ARCHES else x[:, :5])[:2]
                class_parts.append(c.cpu())
                azimuth_parts.append(a.cpu())
        c, a = torch.cat(class_parts), torch.cat(azimuth_parts)
        logits[name] = (c / c.std().clamp_min(1e-6), a / a.std().clamp_min(1e-6))

    cached = np.load("cv_runs/gcc_features.npy")
    tree_x = np.concatenate((cached[:, :64], cached[:, 65:66]), axis=1)
    targets = frame.azimuth.map(azimuth_to_idx).to_numpy()
    tree = ExtraTreesClassifier(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        class_weight="balanced", random_state=5, n_jobs=-1,
    ).fit(tree_x[~valid_mask], targets[~valid_mask])
    probability = tree.predict_proba(tree_x[valid_mask])
    tree_logits = torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0))).float()
    tree_logits = (tree_logits - tree_logits.mean()) / tree_logits.std().clamp_min(1e-6)
    ordered = np.sort(probability, axis=1)
    tree_margin = torch.from_numpy(ordered[:, -1] - ordered[:, -2])

    true_class = valid.sound_class.map(class_to_idx).tolist()
    true_azimuth = valid.azimuth.map(azimuth_to_idx).tolist()
    combinations = {
        "public_like": [("base_gcc", 3.0), ("circ29", 2.0), ("angle37", 2.0)],
        "replace_gcc": [("robust_gcc", 3.0), ("circ29", 2.0), ("angle37", 2.0)],
        "two_gcc_circ": [("base_gcc", 3.0), ("robust_gcc", 2.0), ("circ29", 2.0)],
        "two_gcc_angle": [("base_gcc", 3.0), ("robust_gcc", 2.0), ("angle37", 2.0)],
    }
    for robust_weight in (0.5, 1.0, 1.5, 2.0, 2.5):
        combinations[f"replace_gcc_rw{robust_weight:.1f}"] = [
            ("robust_gcc", robust_weight), ("circ29", 2.0), ("angle37", 2.0)
        ]
    rows = []
    for name, members in combinations.items():
        class_sum = sum(weight * logits[member][0] for member, weight in members)
        neural = sum(weight * logits[member][1] for member, weight in members)
        neural = (neural - neural.mean()) / neural.std().clamp_min(1e-6)
        class_pred = class_sum.argmax(1)
        predictions = {weight: (neural + weight * tree_logits).argmax(1) for weight in (0.0, 0.15, 0.30)}
        low, high = predictions[0.15], predictions[0.30]
        changed = low != high
        threshold = torch.quantile(tree_margin[changed], 0.50)
        conditional = low.clone()
        conditional[changed & (tree_margin >= threshold)] = high[changed & (tree_margin >= threshold)]
        predictions["conditional"] = conditional
        for rule, azimuth_pred in predictions.items():
            rows.append({
                "combination": name,
                "rule": str(rule),
                "class_acc": float((class_pred == torch.tensor(true_class)).float().mean()),
                "azimuth_acc": float((azimuth_pred == torch.tensor(true_azimuth)).float().mean()),
                "joint_macro_f1": joint_macro_f1(
                    true_class, true_azimuth, class_pred.tolist(), azimuth_pred.tolist(),
                    idx_to_class, idx_to_azimuth,
                ),
            })
    result = pd.DataFrame(rows).sort_values("joint_macro_f1", ascending=False)
    result.to_csv("cv_runs/domain_ensemble_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

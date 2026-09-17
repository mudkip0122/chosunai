from __future__ import annotations

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


def evaluate(old_path: Path, new_path: Path, seed: int, cached: np.ndarray) -> pd.DataFrame:
    old_ckpt = torch.load(old_path, map_location="cpu", weights_only=False)
    new_ckpt = torch.load(new_path, map_location="cpu", weights_only=False)
    frame = pd.read_csv("data/train.csv")
    key = frame.sound_class.astype(str) + "_" + frame.azimuth.astype(str)
    train_frame, valid_frame = train_test_split(
        frame, test_size=0.2, random_state=seed, stratify=key
    )
    class_to_idx = old_ckpt["class_to_idx"]
    azimuth_to_idx = old_ckpt["azimuth_to_idx"]
    idx_to_class = {int(k): v for k, v in old_ckpt["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in old_ckpt["idx_to_azimuth"].items()}
    dataset = AudioDataset(
        valid_frame, Path("data/train_audio"), 16000, 2.0, 64,
        class_to_idx, azimuth_to_idx, train=False, include_gcc=True,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(ckpt: dict):
        arch = ckpt["config"].get("arch", "baseline")
        model = build_model(arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
        model.load_state_dict(ckpt["model"])
        return model.eval(), arch

    old_model, old_arch = load(old_ckpt)
    new_model, new_arch = load(new_ckpt)
    old_c, old_a, new_a = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            oc, oa = old_model(x if old_arch in GCC_ARCHES else x[:, :5])[:2]
            _, na = new_model(x if new_arch in GCC_ARCHES else x[:, :5])[:2]
            old_c.append(oc.cpu())
            old_a.append(oa.cpu())
            new_a.append(na.cpu())
    old_c, old_a, new_a = torch.cat(old_c), torch.cat(old_a), torch.cat(new_a)
    old_a = old_a / old_a.std().clamp_min(1e-6)
    new_a = new_a / new_a.std().clamp_min(1e-6)

    tree_x = np.concatenate((cached[:, :64], cached[:, 65:66]), axis=1)
    targets = frame.azimuth.map(azimuth_to_idx).to_numpy()
    tree = ExtraTreesClassifier(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        class_weight="balanced", random_state=5, n_jobs=-1,
    ).fit(tree_x[train_frame.index], targets[train_frame.index])
    probability = tree.predict_proba(tree_x[valid_frame.index])
    tree_logits = torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0))).float()
    tree_logits = (tree_logits - tree_logits.mean()) / tree_logits.std().clamp_min(1e-6)
    sorted_probability = np.sort(probability, axis=1)
    tree_margin = torch.from_numpy(sorted_probability[:, -1] - sorted_probability[:, -2])
    tree_pred = tree_logits.argmax(1)

    old_neural = (old_a - old_a.mean()) / old_a.std().clamp_min(1e-6)
    low = (old_neural + 0.15 * tree_logits).argmax(1)
    high = (old_neural + 0.30 * tree_logits).argmax(1)
    strength_changed = low != high
    threshold = torch.quantile(tree_margin[strength_changed], 0.50)
    base = low.clone()
    base[strength_changed & (tree_margin >= threshold)] = high[
        strength_changed & (tree_margin >= threshold)
    ]

    robust_neural = old_a + new_a
    robust_neural = (robust_neural - robust_neural.mean()) / robust_neural.std().clamp_min(1e-6)
    robust_logits = robust_neural + 0.25 * tree_logits
    robust = robust_logits.argmax(1)
    robust_margin = torch.topk(robust_logits, 2, dim=1).values
    robust_margin = robust_margin[:, 0] - robust_margin[:, 1]
    old_logits = old_neural + 0.15 * tree_logits
    old_margin = torch.topk(old_logits, 2, dim=1).values
    old_margin = old_margin[:, 0] - old_margin[:, 1]
    new_pred = new_a.argmax(1)
    changed = robust != base

    true_class = valid_frame.sound_class.map(class_to_idx).tolist()
    true_angle = valid_frame.azimuth.map(azimuth_to_idx).tolist()
    class_pred = old_c.argmax(1).tolist()

    def score(pred: torch.Tensor) -> float:
        return joint_macro_f1(
            true_class, true_angle, class_pred, pred.tolist(), idx_to_class, idx_to_azimuth
        )

    masks: dict[str, torch.Tensor] = {
        "robust_all": changed,
        "tree_agrees": changed & (tree_pred == robust),
        "new_agrees": changed & (new_pred == robust),
        "tree_and_new_agree": changed & (tree_pred == robust) & (new_pred == robust),
    }
    tree_agreement = changed & (tree_pred == robust)
    for q in (0.25, 0.50, 0.75):
        masks[f"tree_agrees_margin_q{q:.2f}"] = tree_agreement & (
            tree_margin >= torch.quantile(tree_margin[tree_agreement], q)
        )
    for q in (0.50, 0.75, 0.90):
        masks[f"robust_margin_q{q:.2f}"] = changed & (
            robust_margin >= torch.quantile(robust_margin[changed], q)
        )
        advantage = robust_margin - old_margin
        masks[f"margin_advantage_q{q:.2f}"] = changed & (
            advantage >= torch.quantile(advantage[changed], q)
        )
    rows = [{"seed": seed, "rule": "base_conditional", "score": score(base), "selected": 0}]
    for name, mask in masks.items():
        pred = base.clone()
        pred[mask] = robust[mask]
        rows.append({"seed": seed, "rule": name, "score": score(pred), "selected": int(mask.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    cached = np.load("cv_runs/gcc_features.npy")
    result = pd.concat(
        [
            evaluate(Path("checkpoints/gcc_seed5.pt"), Path("checkpoints/gcc_circnoise_seed5.pt"), 5, cached),
            evaluate(Path("checkpoints/seed17.pt"), Path("checkpoints/gcc_circnoise_seed17.pt"), 17, cached),
        ],
        ignore_index=True,
    )
    scores = result.pivot(index="rule", columns="seed", values="score")
    selected = result.pivot(index="rule", columns="seed", values="selected")
    base = scores.loc["base_conditional"]
    scores["min_gain"] = (scores[[5, 17]] - base).min(axis=1)
    scores["mean_gain"] = (scores[[5, 17]] - base).mean(axis=1)
    scores["selected_s5"] = selected[5]
    scores["selected_s17"] = selected[17]
    print(scores.sort_values(["min_gain", "mean_gain"], ascending=False).to_string())
    result.to_csv("cv_runs/robust_azimuth_gate.csv", index=False)


if __name__ == "__main__":
    main()

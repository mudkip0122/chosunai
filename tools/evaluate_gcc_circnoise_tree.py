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


def infer_pair(old_path: Path, new_path: Path, seed: int, features: np.ndarray) -> pd.DataFrame:
    old_ckpt = torch.load(old_path, map_location="cpu", weights_only=False)
    new_ckpt = torch.load(new_path, map_location="cpu", weights_only=False)
    frame = pd.read_csv("data/train.csv")
    key = frame.sound_class.astype(str) + "_" + frame.azimuth.astype(str)
    train_frame, valid_frame = train_test_split(frame, test_size=0.2, random_state=seed, stratify=key)
    class_to_idx, azimuth_to_idx = old_ckpt["class_to_idx"], old_ckpt["azimuth_to_idx"]
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
    old_c, old_a, new_c, new_a = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            oc, oa = old_model(x if old_arch in GCC_ARCHES else x[:, :5])[:2]
            nc, na = new_model(x if new_arch in GCC_ARCHES else x[:, :5])[:2]
            old_c.append(oc.cpu()); old_a.append(oa.cpu())
            new_c.append(nc.cpu()); new_a.append(na.cpu())
    old_c, old_a = torch.cat(old_c), torch.cat(old_a)
    new_c, new_a = torch.cat(new_c), torch.cat(new_a)
    class_logits = old_c / old_c.std().clamp_min(1e-6) + 1.8 * new_c / new_c.std().clamp_min(1e-6)
    neural = old_a / old_a.std().clamp_min(1e-6) + new_a / new_a.std().clamp_min(1e-6)
    neural = (neural - neural.mean()) / neural.std().clamp_min(1e-6)

    x_tree = np.concatenate((features[:, :64], features[:, 65:66]), axis=1)
    targets = frame.azimuth.map(azimuth_to_idx).to_numpy()
    tree = ExtraTreesClassifier(
        n_estimators=600, max_features=1.0, min_samples_leaf=1,
        class_weight="balanced", random_state=5, n_jobs=-1,
    ).fit(x_tree[train_frame.index], targets[train_frame.index])
    probability = tree.predict_proba(x_tree[valid_frame.index])
    tree_logits = torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0))).float()
    tree_logits = (tree_logits - tree_logits.mean()) / tree_logits.std().clamp_min(1e-6)

    true_class = valid_frame.sound_class.map(class_to_idx).tolist()
    true_angle = valid_frame.azimuth.map(azimuth_to_idx).tolist()
    class_pred = class_logits.argmax(1).tolist()

    def score(pred: torch.Tensor) -> float:
        return joint_macro_f1(true_class, true_angle, class_pred, pred.tolist(), idx_to_class, idx_to_azimuth)

    rows = []
    predictions = {}
    for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        pred = (neural + weight * tree_logits).argmax(1)
        predictions[weight] = pred
        rows.append({"seed": seed, "rule": f"w{weight:.2f}", "score": score(pred)})
    margin_np = np.sort(probability, axis=1)
    margin = torch.from_numpy(margin_np[:, -1] - margin_np[:, -2])
    for low_weight, high_weight in ((0.10, 0.20), (0.10, 0.25), (0.15, 0.25), (0.15, 0.30)):
        low, high = predictions[low_weight], predictions[high_weight]
        changed = low != high
        for quantile in (0.25, 0.50, 0.75):
            threshold = torch.quantile(margin[changed], quantile)
            selected = changed & (margin >= threshold)
            pred = low.clone(); pred[selected] = high[selected]
            rows.append({
                "seed": seed,
                "rule": f"cond{low_weight:.2f}_{high_weight:.2f}_q{quantile:.2f}",
                "score": score(pred),
            })
    return pd.DataFrame(rows)


def main() -> None:
    features = np.load("cv_runs/gcc_features.npy")
    result = pd.concat([
        infer_pair(Path("checkpoints/gcc_seed5.pt"), Path("checkpoints/gcc_circnoise_seed5.pt"), 5, features),
        infer_pair(Path("checkpoints/seed17.pt"), Path("checkpoints/gcc_circnoise_seed17.pt"), 17, features),
    ], ignore_index=True)
    pivot = result.pivot(index="rule", columns="seed", values="score")
    base = pivot.loc["w0.00"]
    pivot["min_gain"] = (pivot - base).min(axis=1)
    pivot["mean_gain"] = (pivot - base).mean(axis=1)
    print(pivot.sort_values(["min_gain", "mean_gain"], ascending=False).to_string())
    result.to_csv("cv_runs/gcc_circnoise_tree.csv", index=False)


if __name__ == "__main__":
    main()

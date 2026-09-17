from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.audio_dataset import AudioDataset
from src.model import build_model
from train import joint_macro_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    base_ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    new_ckpt = torch.load(args.new, map_location="cpu", weights_only=False)
    idx_to_class = {int(k): v for k, v in base_ckpt["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in base_ckpt["idx_to_azimuth"].items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    azimuth_to_idx = {v: k for k, v in idx_to_azimuth.items()}

    frame = pd.read_csv("data/train.csv")
    stratify = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
    _, valid = train_test_split(frame, test_size=0.2, random_state=args.seed, stratify=stratify)
    dataset = AudioDataset(
        valid,
        Path("data/train_audio"),
        16000,
        2.0,
        64,
        class_to_idx,
        azimuth_to_idx,
        train=False,
        include_gcc=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(ckpt: dict) -> torch.nn.Module:
        arch = ckpt["config"].get("arch", "baseline")
        model = build_model(arch, len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(ckpt["model"])
        return model.eval()

    base_model, new_model = load(base_ckpt), load(new_ckpt)
    base_arch = base_ckpt["config"].get("arch", "baseline")
    new_arch = new_ckpt["config"].get("arch", "baseline")
    gcc_arches = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}
    base_class, new_class, base_azimuth, new_azimuth, true_class, true_azimuth = [], [], [], [], [], []
    with torch.no_grad():
        for features, y_class, y_azimuth in loader:
            features = features.to(device)
            base_outputs = base_model(features if base_arch in gcc_arches else features[:, :5])
            new_outputs = new_model(features if new_arch in gcc_arches else features[:, :5])
            base_class.append(base_outputs[0].cpu())
            new_class.append(new_outputs[0].cpu())
            base_azimuth.append(base_outputs[1].cpu())
            new_azimuth.append(new_outputs[1].cpu())
            true_class.append(y_class)
            true_azimuth.append(y_azimuth)

    bc, nc = torch.cat(base_class), torch.cat(new_class)
    ba, na = torch.cat(base_azimuth), torch.cat(new_azimuth)
    yc, ya = torch.cat(true_class), torch.cat(true_azimuth)
    base_class_scale = bc.std().clamp_min(1e-6)
    new_class_scale = nc.std().clamp_min(1e-6)
    base_azimuth_scale = ba.std().clamp_min(1e-6)
    new_azimuth_scale = na.std().clamp_min(1e-6)

    rows = []
    for class_weight in np.arange(0.0, 2.01, 0.1):
        class_logits = bc / base_class_scale + class_weight * nc / new_class_scale
        for azimuth_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            azimuth_logits = ba / base_azimuth_scale + azimuth_weight * na / new_azimuth_scale
            pc = class_logits.argmax(1)
            pa = azimuth_logits.argmax(1)
            score = joint_macro_f1(
                yc.tolist(), ya.tolist(), pc.tolist(), pa.tolist(), idx_to_class, idx_to_azimuth
            )
            rows.append(
                {
                    "class_weight": float(class_weight),
                    "azimuth_weight": float(azimuth_weight),
                    "class_acc": float((pc == yc).float().mean()),
                    "azimuth_acc": float((pa == ya).float().mean()),
                    "joint_macro_f1": score,
                }
            )
    result = pd.DataFrame(rows).sort_values("joint_macro_f1", ascending=False)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

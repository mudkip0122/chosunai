from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from infer import mirror_azimuth_logits, mirror_spatial_features
from src.audio_dataset import AudioDataset
from src.model import build_model
from train import joint_macro_f1
from make_decode_sweep import angle_scores


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in checkpoint["idx_to_azimuth"].items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    azimuth_to_idx = {v: k for k, v in idx_to_azimuth.items()}

    frame = pd.read_csv(Path(config["data_dir"]) / "train.csv")
    stratify = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
    _, valid = train_test_split(
        frame, test_size=0.2, random_state=int(config["seed"]), stratify=stratify
    )
    dataset = AudioDataset(
        valid,
        Path(config["audio_dir"]),
        int(config["sample_rate"]),
        float(config["duration"]),
        int(config["n_mels"]),
        class_to_idx,
        azimuth_to_idx,
        train=False,
        include_gcc=config["arch"] in GCC_ARCHES,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["arch"], len(idx_to_class), len(idx_to_azimuth)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    truth_c, truth_a = [], []
    plain_c, plain_a, tta_c, tta_a = [], [], [], []
    plain_azimuth_logits, plain_angle_vectors = [], []
    with torch.no_grad():
        for features, target_c, target_a in loader:
            features = features.to(device)
            outputs = model(features)
            logits_c, logits_a = outputs[:2]
            mirror_c, mirror_a = model(mirror_spatial_features(features))[:2]
            mirror_a = mirror_azimuth_logits(mirror_a, idx_to_azimuth)
            truth_c.extend(target_c.tolist())
            truth_a.extend(target_a.tolist())
            plain_c.extend(logits_c.argmax(1).cpu().tolist())
            plain_a.extend(logits_a.argmax(1).cpu().tolist())
            plain_azimuth_logits.append(logits_a.cpu())
            if len(outputs) >= 3:
                plain_angle_vectors.append(outputs[2].cpu())
            tta_c.extend((logits_c + mirror_c).argmax(1).cpu().tolist())
            tta_a.extend((logits_a + mirror_a).argmax(1).cpu().tolist())

    for name, pred_c, pred_a in (
        ("plain", plain_c, plain_a),
        ("mirror_tta", tta_c, tta_a),
    ):
        score = joint_macro_f1(
            truth_c, truth_a, pred_c, pred_a, idx_to_class, idx_to_azimuth
        )
        class_acc = sum(x == y for x, y in zip(truth_c, pred_c)) / len(truth_c)
        azimuth_acc = sum(x == y for x, y in zip(truth_a, pred_a)) / len(truth_a)
        print(
            f"{name}: joint_macro_f1={score:.8f} "
            f"class_acc={class_acc:.8f} azimuth_acc={azimuth_acc:.8f}"
        )

    if plain_angle_vectors:
        azimuth_logits = torch.cat(plain_azimuth_logits)
        scores = angle_scores(torch.cat(plain_angle_vectors), idx_to_azimuth)
        for weight in (0.25, 0.5, 1.0):
            pred_a = (azimuth_logits + weight * scores).argmax(1).tolist()
            score = joint_macro_f1(
                truth_c, truth_a, plain_c, pred_a, idx_to_class, idx_to_azimuth
            )
            print(f"angle_weight={weight:.2f}: joint_macro_f1={score:.8f}")


if __name__ == "__main__":
    main()

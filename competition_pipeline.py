"""End-to-end training and submission pipeline for the Chosun audio challenge.

The default ``disentangled`` network uses separate lightweight trunks for
sound identity and spatial localization.  Input features are stereo log-mel,
ILD, cosine/sine IPD, and a 64-bin GCC-PHAT vector.  Every inference command
profiles the requested model/TTA combination and refuses to exceed 100 MMAC.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from thop import profile
from thop.vision.basic_hooks import zero_ops
from torch import nn
from torch.utils.data import DataLoader

from infer import mirror_azimuth_logits, mirror_spatial_features
from make_decode_sweep import angle_scores
from src.audio_dataset import AudioDataset, invert_map, make_label_maps
from src.model import build_model
from train import run_epoch


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}
SUPPORTED_ARCHES = ("baseline", "gcc", "gcc_stats", "disentangled")
MMAC_LIMIT = 100.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_mmac(arch: str, classes: int = 8, azimuths: int = 25) -> float:
    model = build_model(arch, classes, azimuths).eval()
    channels = 6 if arch in GCC_ARCHES else 5
    sample = torch.zeros(1, channels, 64, 201)
    macs, _ = profile(
        model,
        inputs=(sample,),
        custom_ops={
            nn.BatchNorm2d: zero_ops,
            nn.SiLU: zero_ops,
            nn.AdaptiveAvgPool2d: zero_ops,
            nn.Dropout: zero_ops,
        },
        verbose=False,
    )
    return float(macs / 1_000_000)


def enforce_compute_limit(arches: list[str], tta: str) -> float:
    views = 2 if tta == "mirror" else 1
    total = sum(model_mmac(arch) for arch in arches) * views
    if total >= MMAC_LIMIT:
        raise ValueError(
            f"Requested inference costs {total:.6f} MMAC/audio, "
            f"which violates the < {MMAC_LIMIT:.0f} MMAC limit."
        )
    return total


def build_train_loaders(args: argparse.Namespace):
    frame = pd.read_csv(args.data_dir / "train.csv")
    class_to_idx, azimuth_to_idx = make_label_maps(frame)
    idx_to_class = invert_map(class_to_idx)
    idx_to_azimuth = invert_map(azimuth_to_idx)

    if args.validation_fraction > 0:
        strata = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
        train_frame, valid_frame = train_test_split(
            frame,
            test_size=args.validation_fraction,
            random_state=args.seed,
            stratify=strata,
        )
    else:
        train_frame, valid_frame = frame, None

    common = dict(
        audio_dir=args.data_dir / "train_audio",
        sample_rate=16000,
        duration=2.0,
        n_mels=64,
        class_to_idx=class_to_idx,
        azimuth_to_idx=azimuth_to_idx,
        include_gcc=args.arch in GCC_ARCHES,
    )
    train_set = AudioDataset(
        train_frame,
        train=True,
        background_noise_prob=args.background_noise_prob,
        **common,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = None
    if valid_frame is not None:
        valid_set = AudioDataset(valid_frame, train=False, **common)
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
    return train_loader, valid_loader, class_to_idx, azimuth_to_idx, idx_to_class, idx_to_azimuth


def train_checkpoint(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    (
        train_loader,
        valid_loader,
        class_to_idx,
        azimuth_to_idx,
        idx_to_class,
        idx_to_azimuth,
    ) = build_train_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            azimuth_loss_weight=args.azimuth_loss_weight,
            class_label_smoothing=args.class_label_smoothing,
            azimuth_soft_sigma=args.azimuth_soft_sigma,
            angle_loss_weight=args.angle_loss_weight,
            idx_to_class=idx_to_class,
            idx_to_azimuth=idx_to_azimuth,
        )
        valid_metrics = None
        if valid_loader is not None:
            valid_metrics = run_epoch(
                model,
                valid_loader,
                optimizer,
                device,
                train=False,
                azimuth_loss_weight=args.azimuth_loss_weight,
                class_label_smoothing=args.class_label_smoothing,
                azimuth_soft_sigma=args.azimuth_soft_sigma,
                angle_loss_weight=args.angle_loss_weight,
                idx_to_class=idx_to_class,
                idx_to_azimuth=idx_to_azimuth,
            )
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        selected = valid_metrics["joint_macro_f1"] if valid_metrics else train_metrics["joint_macro_f1"]
        print(
            f"epoch={epoch:03d} train_joint_f1={train_metrics['joint_macro_f1']:.6f} "
            + (f"valid_joint_f1={selected:.6f}" if valid_metrics else "")
        )
        if valid_loader is None or selected > best_score:
            best_score = selected
            config = vars(args).copy()
            config.update({"sample_rate": 16000, "duration": 2.0, "n_mels": 64})
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "azimuth_to_idx": azimuth_to_idx,
                    "idx_to_class": idx_to_class,
                    "idx_to_azimuth": idx_to_azimuth,
                    "config": config,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                },
                args.out,
            )

    args.out.with_suffix(".json").write_text(
        json.dumps({"best_score": best_score, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.out}")
    return args.out


def infer_submission(args: argparse.Namespace) -> Path:
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoint]
    arches = [item["config"].get("arch", "baseline") for item in loaded]
    total_mmac = enforce_compute_limit(arches, args.tta)
    first = loaded[0]
    idx_to_class = {int(k): v for k, v in first["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in first["idx_to_azimuth"].items()}
    if any(item["idx_to_class"] != first["idx_to_class"] for item in loaded[1:]):
        raise ValueError("Checkpoint class maps do not match.")

    test_frame = pd.read_csv(args.data_dir / "test.csv")
    config = first["config"]
    dataset = AudioDataset(
        test_frame,
        args.data_dir / "test_audio",
        int(config.get("sample_rate", 16000)),
        float(config.get("duration", 2.0)),
        int(config.get("n_mels", 64)),
        train=False,
        include_gcc=any(arch in GCC_ARCHES for arch in arches),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for item, arch in zip(loaded, arches):
        model = build_model(arch, len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(item["model"])
        models.append(model.eval())

    rows: list[dict] = []
    with torch.no_grad():
        for features, ids in loader:
            features = features.to(device)
            class_parts, azimuth_parts, angle_parts = [], [], []
            for model, arch in zip(models, arches):
                model_input = features if arch in GCC_ARCHES else features[:, :5]
                outputs = model(model_input)
                class_logits, azimuth_logits = outputs[:2]
                if args.tta == "mirror":
                    mirrored = mirror_spatial_features(features)
                    mirror_input = mirrored if arch in GCC_ARCHES else mirrored[:, :5]
                    mirror_outputs = model(mirror_input)
                    class_logits = (class_logits + mirror_outputs[0]) * 0.5
                    azimuth_logits = (
                        azimuth_logits
                        + mirror_azimuth_logits(mirror_outputs[1], idx_to_azimuth)
                    ) * 0.5
                class_parts.append(class_logits)
                azimuth_parts.append(azimuth_logits)
                if len(outputs) >= 3:
                    angle_parts.append(angle_scores(outputs[2], idx_to_azimuth))

            class_logits = torch.stack(class_parts).mean(0)
            azimuth_logits = torch.stack(azimuth_parts).mean(0)
            if angle_parts and args.angle_score_weight != 0:
                azimuth_logits += args.angle_score_weight * torch.stack(angle_parts).mean(0)
            class_pred = class_logits.argmax(1).cpu().tolist()
            azimuth_pred = azimuth_logits.argmax(1).cpu().tolist()
            rows.extend(
                {
                    "id": sample_id,
                    "sound_class": idx_to_class[class_idx],
                    "azimuth": idx_to_azimuth[azimuth_idx],
                }
                for sample_id, class_idx, azimuth_idx in zip(ids, class_pred, azimuth_pred)
            )

    submission = pd.DataFrame(rows)
    if len(submission) != 5000 or submission["id"].duplicated().any() or submission.isna().any().any():
        raise ValueError("Submission failed row-count, duplicate-ID, or null validation.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out}; total_mmac={total_mmac:.6f}")
    return args.out


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=SUPPORTED_ARCHES, default="disentangled")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--azimuth-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-label-smoothing", type=float, default=0.03)
    parser.add_argument("--azimuth-soft-sigma", type=float, default=8.0)
    parser.add_argument("--angle-loss-weight", type=float, default=0.25)
    parser.add_argument("--background-noise-prob", type=float, default=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train with optional validation split.")
    add_training_args(train_parser)

    infer_parser = subparsers.add_parser("infer", help="Generate and validate submission.csv.")
    infer_parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    infer_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    infer_parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    infer_parser.add_argument("--batch-size", type=int, default=128)
    infer_parser.add_argument("--num-workers", type=int, default=4)
    infer_parser.add_argument("--tta", choices=("none", "mirror"), default="none")
    infer_parser.add_argument("--angle-score-weight", type=float, default=0.0)

    mmac_parser = subparsers.add_parser("mmac", help="Profile an inference plan.")
    mmac_parser.add_argument("--arch", choices=SUPPORTED_ARCHES, default="disentangled")
    mmac_parser.add_argument("--models", type=int, default=1)
    mmac_parser.add_argument("--tta", choices=("none", "mirror"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_checkpoint(args)
    elif args.command == "infer":
        infer_submission(args)
    else:
        total = enforce_compute_limit([args.arch] * args.models, args.tta)
        print(f"total_mmac={total:.6f}; under_100_mmac=True")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.audio_dataset import AudioDataset, invert_map, make_label_maps
from src.model import build_model
from train import run_epoch, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/train_audio"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--azimuth-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-label-smoothing", type=float, default=0.0)
    parser.add_argument("--azimuth-soft-sigma", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=("baseline", "baseline_angle", "spatial", "dualpath", "gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled", "v2", "v3"), default="baseline")
    parser.add_argument("--background-noise-prob", type=float, default=0.0)
    parser.add_argument("--angle-loss-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--head-only", choices=("gcc", "temporal", "classbranch"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_csv = args.data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing {args.audio_dir}")

    train_df = pd.read_csv(train_csv)
    class_to_idx, azimuth_to_idx = make_label_maps(train_df)
    idx_to_class = invert_map(class_to_idx)
    idx_to_azimuth = invert_map(azimuth_to_idx)

    train_ds = AudioDataset(
        train_df,
        args.audio_dir,
        args.sample_rate,
        args.duration,
        args.n_mels,
        class_to_idx,
        azimuth_to_idx,
        train=not args.no_augment,
        background_noise_prob=args.background_noise_prob,
        include_gcc=args.arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(initial["model"], strict=False)
        print(
            f"initialized from {args.init_checkpoint}; "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    if args.freeze_base:
        model.freeze_base = True
        for name, parameter in model.named_parameters():
            if args.head_only == "temporal":
                parameter.requires_grad = name.startswith("temporal_")
            elif args.head_only == "classbranch":
                parameter.requires_grad = name.startswith("class_branch")
            elif args.head_only == "gcc":
                parameter.requires_grad = name.startswith("gcc_head") or name == "gcc_scale"
            else:
                parameter.requires_grad = (
                    name.startswith("gcc_head") or name == "gcc_scale" or name.startswith("temporal_")
                )
        print(f"trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
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
        scheduler.step()
        row = {"epoch": epoch, **train_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_class_acc={train_metrics['class_acc']:.4f} "
            f"train_azimuth_acc={train_metrics['azimuth_acc']:.4f} "
            f"train_joint_f1={train_metrics['joint_macro_f1']:.4f}"
        )

    checkpoint = {
        "model": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "azimuth_to_idx": azimuth_to_idx,
        "idx_to_azimuth": idx_to_azimuth,
        "config": vars(args),
        "train_metrics": history[-1],
    }
    torch.save(checkpoint, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps({"final_train_metrics": history[-1], "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

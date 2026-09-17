from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioDataset, invert_map, make_label_maps
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/train_audio"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--azimuth-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-label-smoothing", type=float, default=0.0)
    parser.add_argument("--azimuth-soft-sigma", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/baseline.pt"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument(
        "--valid-indices",
        type=Path,
        default=None,
        help="Optional .npy row indices for an exact validation split.",
    )
    parser.add_argument("--arch", choices=("baseline", "baseline_angle", "spatial", "dualpath", "gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled", "v2", "v3"), default="baseline")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--head-only", choices=("gcc", "temporal", "classbranch"), default=None)
    parser.add_argument("--selection-metric", choices=("joint_macro_f1", "class_acc"), default="joint_macro_f1")
    parser.add_argument("--background-noise-prob", type=float, default=0.0)
    parser.add_argument("--angle-loss-weight", type=float, default=0.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(1) == target).float().mean().item()


def joint_macro_f1(
    true_class: list[int],
    true_azimuth: list[int],
    pred_class: list[int],
    pred_azimuth: list[int],
    idx_to_class: dict[int, str],
    idx_to_azimuth: dict[int, int],
) -> float:
    true_class_names = np.asarray([idx_to_class[i] for i in true_class])
    pred_class_names = np.asarray([idx_to_class[i] for i in pred_class])
    true_angles = np.asarray([idx_to_azimuth[i] for i in true_azimuth], dtype=float)
    pred_angles = np.asarray([idx_to_azimuth[i] for i in pred_azimuth], dtype=float)
    labels = sorted(set(idx_to_class.values()))

    tolerance_scores = []
    for tolerance in (5.0, 10.0, 20.0):
        matched = (true_class_names == pred_class_names) & (np.abs(true_angles - pred_angles) <= tolerance)
        class_scores = []
        for label in labels:
            tp = int(np.sum(matched & (true_class_names == label)))
            fn = int(np.sum(true_class_names == label)) - tp
            fp = int(np.sum(pred_class_names == label)) - tp
            denominator = 2 * tp + fp + fn
            class_scores.append(2 * tp / denominator if denominator else 0.0)
        tolerance_scores.append(float(np.mean(class_scores)))
    return float(np.mean(tolerance_scores))


def make_azimuth_soft_targets(
    target: torch.Tensor,
    idx_to_azimuth: dict[int, int],
    sigma: float,
) -> torch.Tensor:
    angles = torch.tensor(
        [idx_to_azimuth[i] for i in range(len(idx_to_azimuth))],
        dtype=torch.float32,
        device=target.device,
    )
    target_angles = angles[target].unsqueeze(1)
    raw_distances = torch.abs(target_angles - angles.unsqueeze(0))
    distances = torch.minimum(raw_distances, 360.0 - raw_distances)
    weights = torch.exp(-0.5 * (distances / sigma).square())
    return weights / weights.sum(dim=1, keepdim=True)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def make_angle_targets(target: torch.Tensor, idx_to_azimuth: dict[int, int]) -> torch.Tensor:
    angles = torch.tensor(
        [idx_to_azimuth[i] for i in range(len(idx_to_azimuth))],
        dtype=torch.float32,
        device=target.device,
    )
    radians = torch.deg2rad(angles[target])
    return torch.stack((torch.cos(radians), torch.sin(radians)), dim=1)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    azimuth_loss_weight: float,
    class_label_smoothing: float,
    azimuth_soft_sigma: float,
    angle_loss_weight: float,
    idx_to_class: dict[int, str],
    idx_to_azimuth: dict[int, int],
) -> dict:
    model.train(train)
    class_ce = nn.CrossEntropyLoss(label_smoothing=class_label_smoothing)
    azimuth_ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    class_acc = 0.0
    azimuth_acc = 0.0
    count = 0
    true_class = []
    true_azimuth = []
    pred_class = []
    pred_azimuth = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y_class, y_azimuth in tqdm(loader, leave=False):
            x = x.to(device)
            y_class = y_class.to(device)
            y_azimuth = y_azimuth.to(device)

            outputs = model(x)
            class_logits, azimuth_logits = outputs[:2]
            class_loss = class_ce(class_logits, y_class)
            if azimuth_soft_sigma > 0.0:
                soft_targets = make_azimuth_soft_targets(y_azimuth, idx_to_azimuth, azimuth_soft_sigma)
                azimuth_loss = soft_cross_entropy(azimuth_logits, soft_targets)
            else:
                azimuth_loss = azimuth_ce(azimuth_logits, y_azimuth)
            loss = class_loss + azimuth_loss_weight * azimuth_loss
            if angle_loss_weight > 0.0 and len(outputs) >= 3:
                angle_loss = nn.functional.mse_loss(outputs[2], make_angle_targets(y_azimuth, idx_to_azimuth))
                loss = loss + angle_loss_weight * angle_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch = x.size(0)
            class_pred = class_logits.argmax(1)
            azimuth_pred = azimuth_logits.argmax(1)
            total_loss += loss.item() * batch
            class_acc += accuracy(class_logits, y_class) * batch
            azimuth_acc += accuracy(azimuth_logits, y_azimuth) * batch
            count += batch
            true_class.extend(y_class.detach().cpu().tolist())
            true_azimuth.extend(y_azimuth.detach().cpu().tolist())
            pred_class.extend(class_pred.detach().cpu().tolist())
            pred_azimuth.extend(azimuth_pred.detach().cpu().tolist())

    return {
        "loss": total_loss / count,
        "class_acc": class_acc / count,
        "azimuth_acc": azimuth_acc / count,
        "mean_acc": (class_acc + azimuth_acc) / (2 * count),
        "joint_macro_f1": joint_macro_f1(
            true_class,
            true_azimuth,
            pred_class,
            pred_azimuth,
            idx_to_class,
            idx_to_azimuth,
        ),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_csv = args.data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Download train.csv before training.")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing {args.audio_dir}. Extract train_audio.zip before training.")

    df = pd.read_csv(train_csv)
    class_to_idx, azimuth_to_idx = make_label_maps(df)
    idx_to_class = invert_map(class_to_idx)
    idx_to_azimuth = invert_map(azimuth_to_idx)
    if args.valid_indices is not None:
        valid_indices = np.load(args.valid_indices).astype(int)
        if valid_indices.ndim != 1 or len(np.unique(valid_indices)) != len(valid_indices):
            raise ValueError("valid-indices must contain unique one-dimensional row indices")
        if len(valid_indices) == 0 or valid_indices.min() < 0 or valid_indices.max() >= len(df):
            raise ValueError("valid-indices contains an out-of-range row index")
        valid_mask = np.zeros(len(df), dtype=bool)
        valid_mask[valid_indices] = True
        train_df = df.loc[~valid_mask]
        valid_df = df.loc[valid_mask]
        print(f"using fixed validation split {args.valid_indices}: train={len(train_df)} valid={len(valid_df)}")
    else:
        stratify_key = df["sound_class"].astype(str) + "_" + df["azimuth"].astype(str)
        stratify = stratify_key if stratify_key.value_counts().min() >= 2 else df["sound_class"]
        train_df, valid_df = train_test_split(
            df, test_size=0.2, random_state=args.seed, stratify=stratify
        )
    if args.train_limit is not None:
        train_df = train_df.head(args.train_limit)
    if args.valid_limit is not None:
        valid_df = valid_df.head(args.valid_limit)

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
    valid_ds = AudioDataset(
        valid_df,
        args.audio_dir,
        args.sample_rate,
        args.duration,
        args.n_mels,
        class_to_idx,
        azimuth_to_idx,
        train=False,
        include_gcc=args.arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

    best_score = -1.0
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

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_class_acc={valid_metrics['class_acc']:.4f} "
            f"valid_azimuth_acc={valid_metrics['azimuth_acc']:.4f} "
            f"valid_mean_acc={valid_metrics['mean_acc']:.4f} "
            f"valid_joint_f1={valid_metrics['joint_macro_f1']:.4f}"
        )

        selection_score = valid_metrics[args.selection_metric]
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "azimuth_to_idx": azimuth_to_idx,
                    "idx_to_azimuth": idx_to_azimuth,
                    "config": vars(args),
                    "valid_metrics": valid_metrics,
                },
                args.out,
            )
            print(f"saved {args.out} {args.selection_metric}={best_score:.4f}")

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps({"best_score": best_score}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

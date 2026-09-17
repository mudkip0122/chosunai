from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader

from src.audio_dataset import AudioDataset, invert_map, make_label_maps
from src.model import build_model
from train import joint_macro_f1, run_epoch, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/train_audio"))
    parser.add_argument("--out-dir", type=Path, default=Path("cv_runs/baseline"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arch", choices=("baseline", "baseline_angle", "spatial", "dualpath", "gcc", "gcc_temporal", "gcc_class", "gcc_stats", "v2", "v3"), default="baseline")
    parser.add_argument("--angle-loss-weight", type=float, default=0.0)
    parser.add_argument("--background-noise-prob", type=float, default=0.0)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--start-split", type=int, default=1)
    parser.add_argument("--max-splits", type=int, default=None)
    return parser.parse_args()


def make_stratify_key(df: pd.DataFrame, folds: int) -> pd.Series:
    class_azimuth = df["sound_class"].astype(str) + "_" + df["azimuth"].astype(str)
    if class_azimuth.value_counts().min() >= folds:
        return class_azimuth
    return df["sound_class"].astype(str)


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: dict[int, str],
    idx_to_azimuth: dict[int, int],
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for x, y_class, y_azimuth in loader:
            outputs = model(x.to(device))
            class_logits, azimuth_logits = outputs[:2]
            for true_class, true_azimuth, pred_class, pred_azimuth in zip(
                y_class.cpu().tolist(),
                y_azimuth.cpu().tolist(),
                class_logits.argmax(1).cpu().tolist(),
                azimuth_logits.argmax(1).cpu().tolist(),
            ):
                rows.append(
                    {
                        "true_sound_class": idx_to_class[true_class],
                        "true_azimuth": idx_to_azimuth[true_azimuth],
                        "pred_sound_class": idx_to_class[pred_class],
                        "pred_azimuth": idx_to_azimuth[pred_azimuth],
                    }
                )
    return pd.DataFrame(rows)


def add_score_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.copy()
    scored["class_correct"] = scored["true_sound_class"] == scored["pred_sound_class"]
    angle_error = (scored["true_azimuth"].astype(float) - scored["pred_azimuth"].astype(float)).abs()
    scored["azimuth_abs_error"] = angle_error
    for tolerance in (5, 10, 20):
        scored[f"joint_correct_{tolerance}"] = scored["class_correct"] & (angle_error <= tolerance)
    return scored


def make_error_reports(oof: pd.DataFrame, out_dir: Path) -> None:
    scored = add_score_columns(oof)
    aggregations = {
        "samples": ("id", "count"),
        "class_acc": ("class_correct", "mean"),
        "joint_at_5": ("joint_correct_5", "mean"),
        "joint_at_10": ("joint_correct_10", "mean"),
        "joint_at_20": ("joint_correct_20", "mean"),
        "mean_azimuth_abs_error": ("azimuth_abs_error", "mean"),
    }
    scored.groupby("true_sound_class").agg(**aggregations).sort_values("joint_at_20").to_csv(
        out_dir / "error_by_class.csv"
    )
    scored.groupby("true_azimuth").agg(**aggregations).sort_values("joint_at_20").to_csv(
        out_dir / "error_by_azimuth.csv"
    )
    scored.groupby(["true_sound_class", "true_azimuth"]).agg(**aggregations).sort_values(
        ["joint_at_20", "samples"]
    ).to_csv(out_dir / "error_by_class_azimuth.csv")


def score_predictions(
    predictions: pd.DataFrame,
    class_to_idx: dict[str, int],
    azimuth_to_idx: dict[int, int],
    idx_to_class: dict[int, str],
    idx_to_azimuth: dict[int, int],
) -> float:
    return joint_macro_f1(
        [class_to_idx[value] for value in predictions["true_sound_class"]],
        [azimuth_to_idx[int(value)] for value in predictions["true_azimuth"]],
        [class_to_idx[value] for value in predictions["pred_sound_class"]],
        [azimuth_to_idx[int(value)] for value in predictions["pred_azimuth"]],
        idx_to_class,
        idx_to_azimuth,
    )


def write_reports(
    out_dir: Path,
    fold_rows: list[dict],
    oof_parts: list[pd.DataFrame],
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    azimuth_to_idx: dict[int, int],
    idx_to_class: dict[int, str],
    idx_to_azimuth: dict[int, int],
) -> None:
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(out_dir / "fold_summary.csv", index=False)
    if not oof_parts:
        return

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    make_error_reports(oof, out_dir)
    summary = {
        "arch": args.arch,
        "folds": args.folds,
        "repeats": args.repeats,
        "epochs": args.epochs,
        "completed_splits": len(fold_rows),
        "mean_fold_joint_macro_f1": float(fold_summary["best_joint_macro_f1"].mean()),
        "std_fold_joint_macro_f1": float(fold_summary["best_joint_macro_f1"].std(ddof=0)),
        "oof_joint_macro_f1": float(
            score_predictions(oof, class_to_idx, azimuth_to_idx, idx_to_class, idx_to_azimuth)
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    train_csv = args.data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing {args.audio_dir}")

    df = pd.read_csv(train_csv)
    if args.row_limit is not None:
        df = df.sample(n=min(args.row_limit, len(df)), random_state=args.seed).reset_index(drop=True)

    class_to_idx, azimuth_to_idx = make_label_maps(df)
    idx_to_class = invert_map(class_to_idx)
    idx_to_azimuth = invert_map(azimuth_to_idx)
    stratify_key = make_stratify_key(df, args.folds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    oof_parts = []

    completed_this_run = 0
    for split_index, (train_index, valid_index) in enumerate(splitter.split(df, stratify_key), start=1):
        if split_index < args.start_split:
            continue
        if args.max_splits is not None and completed_this_run >= args.max_splits:
            break

        repeat = (split_index - 1) // args.folds + 1
        fold = (split_index - 1) % args.folds + 1
        fold_seed = args.seed + split_index
        seed_everything(fold_seed)

        train_df = df.iloc[train_index].reset_index(drop=True)
        valid_df = df.iloc[valid_index].reset_index(drop=True)
        train_ds = AudioDataset(
            train_df,
            args.audio_dir,
            args.sample_rate,
            args.duration,
            args.n_mels,
            class_to_idx,
            azimuth_to_idx,
            train=True,
            background_noise_prob=args.background_noise_prob,
            include_gcc=args.arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats"),
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
            include_gcc=args.arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats"),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = build_model(args.arch, len(class_to_idx), len(azimuth_to_idx)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_score = -1.0
        best_metrics = None
        checkpoint_path = checkpoints_dir / f"repeat{repeat}_fold{fold}.pt"

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
            score = valid_metrics["joint_macro_f1"]
            print(
                f"repeat={repeat} fold={fold} epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"valid_loss={valid_metrics['loss']:.4f} "
                f"valid_joint_f1={score:.4f}"
            )
            if score > best_score:
                best_score = score
                best_metrics = valid_metrics
                torch.save(
                    {
                        "model": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "idx_to_class": idx_to_class,
                        "azimuth_to_idx": azimuth_to_idx,
                        "idx_to_azimuth": idx_to_azimuth,
                        "config": {
                            **vars(args),
                            "repeat": repeat,
                            "fold": fold,
                            "seed": fold_seed,
                        },
                        "valid_metrics": valid_metrics,
                    },
                    checkpoint_path,
                )

        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(loaded["model"])
        predictions = collect_predictions(model, valid_loader, device, idx_to_class, idx_to_azimuth)
        predictions.insert(0, "id", valid_df["id"].astype(str).tolist())
        predictions.insert(1, "repeat", repeat)
        predictions.insert(2, "fold", fold)
        oof_parts.append(predictions)
        fold_rows.append(
            {
                "split_index": split_index,
                "repeat": repeat,
                "fold": fold,
                "checkpoint": str(checkpoint_path),
                "best_joint_macro_f1": best_score,
                **{f"best_{key}": value for key, value in (best_metrics or {}).items()},
            }
        )
        completed_this_run += 1
        write_reports(
            args.out_dir,
            fold_rows,
            oof_parts,
            args,
            class_to_idx,
            azimuth_to_idx,
            idx_to_class,
            idx_to_azimuth,
        )

    if not fold_rows:
        raise ValueError("No CV splits were run. Check --start-split and --max-splits.")


if __name__ == "__main__":
    main()

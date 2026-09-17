from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from quota_infer import joint_quota_assign, quota_assign, uniform_counts
from src.audio_dataset import AudioDataset, invert_map, make_label_maps
from src.model import build_model
from train import joint_macro_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/train_audio"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def score_predictions(
    y_class: list[int],
    y_azimuth: list[int],
    pred_class: np.ndarray,
    pred_azimuth: np.ndarray,
    idx_to_class: dict[int, str],
    idx_to_azimuth: dict[int, int],
) -> float:
    return joint_macro_f1(
        y_class,
        y_azimuth,
        pred_class.astype(int).tolist(),
        pred_azimuth.astype(int).tolist(),
        idx_to_class,
        idx_to_azimuth,
    )


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    df = pd.read_csv(args.data_dir / "train.csv")
    class_to_idx, azimuth_to_idx = make_label_maps(df)
    idx_to_class = invert_map(class_to_idx)
    idx_to_azimuth = invert_map(azimuth_to_idx)
    stratify_key = df["sound_class"].astype(str) + "_" + df["azimuth"].astype(str)
    train_test_split(
        df,
        test_size=0.2,
        random_state=config["seed"],
        stratify=stratify_key,
    )
    _, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config["seed"],
        stratify=stratify_key,
    )

    dataset = AudioDataset(
        valid_df,
        args.audio_dir,
        config["sample_rate"],
        config["duration"],
        config["n_mels"],
        class_to_idx,
        azimuth_to_idx,
        train=False,
        include_gcc=config.get("arch") in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.get("arch", "baseline"), len(idx_to_class), len(idx_to_azimuth)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_class: list[int] = []
    y_azimuth: list[int] = []
    class_batches = []
    azimuth_batches = []
    with torch.no_grad():
        for x, batch_class, batch_azimuth in tqdm(loader):
            x = x.to(device)
            outputs = model(x)
            class_logits, azimuth_logits = outputs[:2]
            y_class.extend(batch_class.tolist())
            y_azimuth.extend(batch_azimuth.tolist())
            class_batches.append(class_logits.cpu().numpy())
            azimuth_batches.append(azimuth_logits.cpu().numpy())

    class_logits_np = np.concatenate(class_batches)
    azimuth_logits_np = np.concatenate(azimuth_batches)
    raw_class = class_logits_np.argmax(axis=1)
    raw_azimuth = azimuth_logits_np.argmax(axis=1)
    ind_class = quota_assign(class_logits_np, uniform_counts(len(y_class), class_logits_np.shape[1]))
    ind_azimuth = quota_assign(azimuth_logits_np, uniform_counts(len(y_class), azimuth_logits_np.shape[1]))
    joint_class, joint_azimuth = joint_quota_assign(class_logits_np, azimuth_logits_np)

    print(f"checkpoint={args.checkpoint}")
    print(f"raw={score_predictions(y_class, y_azimuth, raw_class, raw_azimuth, idx_to_class, idx_to_azimuth):.6f}")
    print(f"class_only={score_predictions(y_class, y_azimuth, ind_class, raw_azimuth, idx_to_class, idx_to_azimuth):.6f}")
    print(f"azimuth_only={score_predictions(y_class, y_azimuth, raw_class, ind_azimuth, idx_to_class, idx_to_azimuth):.6f}")
    print(f"independent={score_predictions(y_class, y_azimuth, ind_class, ind_azimuth, idx_to_class, idx_to_azimuth):.6f}")
    print(f"joint={score_predictions(y_class, y_azimuth, joint_class, joint_azimuth, idx_to_class, idx_to_azimuth):.6f}")


if __name__ == "__main__":
    main()

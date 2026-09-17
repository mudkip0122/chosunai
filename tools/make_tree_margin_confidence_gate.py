from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.make_gcc_tree_fusion import TestWithTreeFeatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, default=Path("outputs/gcc_tree/submission_gcc_tree_w015.csv"))
    parser.add_argument("--high", type=Path, default=Path("outputs/gcc_tree/submission_gcc_tree_w030.csv"))
    parser.add_argument(
        "--tree", type=Path, default=Path("outputs/gcc_tree/gcc_azimuth_extratrees.joblib")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gcc_tree/submission_gcc_tree_marginq050_confq050_and.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    low = pd.read_csv(args.low)
    high = pd.read_csv(args.high)
    if low["id"].tolist() != high["id"].tolist():
        raise ValueError("low/high IDs do not match")

    test = pd.read_csv(args.data_dir / "test.csv")
    loader = DataLoader(
        TestWithTreeFeatures(test, args.data_dir / "test_audio"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    tree = joblib.load(args.tree)
    probabilities: list[np.ndarray] = []
    ids: list[str] = []
    for _, tree_features, batch_ids in tqdm(loader):
        probabilities.append(tree.predict_proba(tree_features.numpy()))
        ids.extend(batch_ids)
    if ids != low["id"].tolist():
        raise ValueError("audio loader IDs do not match submission IDs")

    probability = np.concatenate(probabilities)
    ordered = np.sort(probability, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    changed = low["azimuth"].to_numpy() != high["azimuth"].to_numpy()
    margin_threshold = np.quantile(margin[changed], 0.50)
    confidence_threshold = np.quantile(confidence[changed], 0.50)
    selected = changed & (margin >= margin_threshold) & (confidence >= confidence_threshold)

    output = low.copy()
    output.loc[selected, "azimuth"] = high.loc[selected, "azimuth"].to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    print(f"selected={int(selected.sum())}/{int(changed.sum())} w015-to-w030 changes")
    print(f"margin_threshold={margin_threshold:.8f}")
    print(f"confidence_threshold={confidence_threshold:.8f}")


if __name__ == "__main__":
    main()

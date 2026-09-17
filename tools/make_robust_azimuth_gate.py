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
    parser.add_argument(
        "--best",
        type=Path,
        default=Path("outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv"),
    )
    parser.add_argument(
        "--robust",
        type=Path,
        default=Path("outputs/gcc_circnoise/submission_gcc_tree_w025.csv"),
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path("outputs/gcc_circnoise/gcc_azimuth_extratrees.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robust_gate/submission_bestclass_robustaz_treeagree.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    best = pd.read_csv(args.best)
    robust = pd.read_csv(args.robust)
    if best["id"].tolist() != robust["id"].tolist():
        raise ValueError("best and robust submission IDs do not match")

    test = pd.read_csv(args.data_dir / "test.csv")
    loader = DataLoader(
        TestWithTreeFeatures(test, args.data_dir / "test_audio"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    tree = joblib.load(args.tree)
    tree_predictions: list[np.ndarray] = []
    ids: list[str] = []
    for _, tree_features, batch_ids in tqdm(loader):
        tree_predictions.append(tree.predict(tree_features.numpy()))
        ids.extend(batch_ids)
    if ids != best["id"].tolist():
        raise ValueError("audio loader IDs do not match submission IDs")

    angles = np.sort(pd.read_csv(args.data_dir / "train.csv")["azimuth"].astype(int).unique())
    tree_azimuth = angles[np.concatenate(tree_predictions)]
    robust_azimuth = robust["azimuth"].to_numpy()
    best_azimuth = best["azimuth"].to_numpy()
    selected = (robust_azimuth != best_azimuth) & (tree_azimuth == robust_azimuth)

    output = best.copy()
    output.loc[selected, "azimuth"] = robust_azimuth[selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    print(f"selected={int(selected.sum())}/{len(output)}")
    print(f"class_changes={int((output.sound_class != best.sound_class).sum())}")
    print(f"azimuth_changes={int((output.azimuth != best.azimuth).sum())}")


if __name__ == "__main__":
    main()

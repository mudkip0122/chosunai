from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedGroupKFold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--embedding", type=Path, default=Path("cv_runs/source_fingerprint_direct.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("cv_runs/acoustic_group_folds"))
    parser.add_argument("--clusters-per-class", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frame = pd.read_csv(args.train_csv)
    embeddings = np.load(args.embedding)
    if len(frame) != len(embeddings):
        raise ValueError("embedding row count does not match train.csv")

    groups = np.empty(len(frame), dtype=np.int64)
    offset = 0
    for class_name, indices in frame.groupby("sound_class", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        model = MiniBatchKMeans(
            n_clusters=args.clusters_per_class,
            batch_size=512,
            n_init=10,
            random_state=args.seed,
        )
        labels = model.fit_predict(embeddings[indices])
        groups[indices] = labels + offset
        offset += args.clusters_per_class

    target = frame.sound_class.astype(str) + "_" + frame.azimuth.astype(str)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, (_, valid_indices) in enumerate(splitter.split(frame, target, groups)):
        np.save(args.out_dir / f"fold{fold}_valid_indices.npy", valid_indices.astype(np.int64))
        valid = frame.iloc[valid_indices]
        rows.append({
            "fold": fold,
            "rows": len(valid_indices),
            "groups": int(pd.Series(groups[valid_indices]).nunique()),
            "min_joint_count": int(valid.groupby(["sound_class", "azimuth"]).size().min()),
            "max_joint_count": int(valid.groupby(["sound_class", "azimuth"]).size().max()),
        })
    group_sizes = pd.Series(groups).value_counts()
    report = {
        "clusters_per_class": args.clusters_per_class,
        "total_groups": int(group_sizes.size),
        "smallest_group": int(group_sizes.min()),
        "median_group": float(group_sizes.median()),
        "largest_group": int(group_sizes.max()),
        "folds": rows,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(args.out_dir / "groups.npy", groups)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

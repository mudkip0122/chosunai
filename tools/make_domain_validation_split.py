from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler


def transform_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    transformed = values.copy()
    for column in (0, 1, 3, 4):
        transformed[:, column] = np.sign(values[:, column]) * np.log1p(np.abs(values[:, column]))
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, default=Path("cv_runs/train_simple_audio.npy"))
    parser.add_argument("--test-features", type=Path, default=Path("cv_runs/test_simple_audio.npy"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("cv_runs/domain_valid_indices.npy"))
    args = parser.parse_args()

    if not 0.05 <= args.fraction <= 0.40:
        raise ValueError("fraction must be between 0.05 and 0.40")
    train_x = transform_features(np.load(args.train_features))
    test_x = transform_features(np.load(args.test_features))
    all_x = np.concatenate((train_x, test_x))
    domain = np.concatenate((np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)))
    scaler = RobustScaler().fit(all_x)
    all_x = scaler.transform(all_x)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(all_x), dtype=np.float64)
    for fold, (fit_index, valid_index) in enumerate(splitter.split(all_x, domain)):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=args.seed + fold,
        )
        model.fit(all_x[fit_index], domain[fit_index])
        oof[valid_index] = model.predict_proba(all_x[valid_index])[:, 1]
    auc = roc_auc_score(domain, oof)
    train_propensity = oof[: len(train_x)]

    frame = pd.read_csv(args.train_csv)
    if len(frame) != len(train_propensity):
        raise ValueError("train feature count does not match train.csv")
    key = frame["sound_class"].astype(str) + "_" + frame["azimuth"].astype(str)
    selected: list[int] = []
    for _, group_indices in frame.groupby(key, sort=True).groups.items():
        group_indices = np.asarray(list(group_indices), dtype=int)
        count = max(1, int(round(len(group_indices) * args.fraction)))
        ranked = group_indices[np.argsort(train_propensity[group_indices])[::-1]]
        selected.extend(ranked[:count].tolist())
    selected_array = np.asarray(sorted(selected), dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, selected_array)
    random_reference = np.random.default_rng(args.seed).choice(
        len(train_x), size=len(selected_array), replace=False
    )
    report = {
        "domain_oof_auc": float(auc),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "validation_rows": int(len(selected_array)),
        "validation_fraction": float(len(selected_array) / len(train_x)),
        "mean_propensity_all_train": float(train_propensity.mean()),
        "mean_propensity_domain_validation": float(train_propensity[selected_array].mean()),
        "mean_propensity_random_reference": float(train_propensity[random_reference].mean()),
        "min_joint_stratum_validation_count": int(frame.loc[selected_array].assign(key=key.loc[selected_array]).groupby("key").size().min()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

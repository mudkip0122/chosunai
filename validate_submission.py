from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--class-map", type=Path, default=Path("data/class_map.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sub = pd.read_csv(args.submission)
    sample = pd.read_csv(args.sample)
    class_map = pd.read_csv(args.class_map)

    expected_columns = ["id", "sound_class", "azimuth"]
    if sub.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {sub.columns.tolist()}")
    if len(sub) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, got {len(sub)}")
    if sub["id"].tolist() != sample["id"].tolist():
        raise ValueError("Submission ids or row order do not match sample_submission.csv")

    valid_classes = set(class_map["sound_class"])
    invalid_classes = sorted(set(sub["sound_class"]) - valid_classes)
    if invalid_classes:
        raise ValueError(f"Invalid sound_class values: {invalid_classes}")

    if sub["azimuth"].isna().any():
        raise ValueError("azimuth contains missing values")
    if not pd.api.types.is_integer_dtype(sub["azimuth"]):
        raise ValueError("azimuth must be integer dtype")

    print(f"OK: {args.submission} matches required format with {len(sub)} rows.")


if __name__ == "__main__":
    main()

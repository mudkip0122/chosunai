from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    primary = torch.load(args.primary, map_location="cpu", weights_only=False)
    secondary = torch.load(args.secondary, map_location="cpu", weights_only=False)
    if primary["idx_to_class"] != secondary["idx_to_class"]:
        raise ValueError("Class maps do not match")
    if primary["idx_to_azimuth"] != secondary["idx_to_azimuth"]:
        raise ValueError("Azimuth maps do not match")

    state = {}
    for key, value in primary["model"].items():
        other = secondary["model"][key]
        if torch.is_floating_point(value):
            state[key] = value * args.alpha + other * (1.0 - args.alpha)
        else:
            state[key] = value

    out = dict(primary)
    out["model"] = state
    out["config"] = dict(primary["config"])
    out["config"]["weight_soup"] = {
        "primary": str(args.primary),
        "secondary": str(args.secondary),
        "alpha": args.alpha,
    }
    out["train_metrics"] = {"note": "weight soup checkpoint; train metrics not directly evaluated"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from itertools import product

import torch
from torch import nn
from thop import profile
from thop.vision.basic_hooks import zero_ops

from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--azimuths", type=int, default=25)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--mels", type=int, default=64)
    parser.add_argument("--frames", type=int, default=201)
    parser.add_argument("--models", type=int, default=1)
    parser.add_argument("--tta-views", type=int, default=1)
    parser.add_argument("--arch", choices=("baseline", "baseline_angle", "spatial", "dualpath", "gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled", "v2", "v3"), default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.arch, args.classes, args.azimuths).eval()
    input_channels = 6 if args.arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled") else args.channels
    sample = torch.zeros(1, input_channels, args.mels, args.frames)
    single_macs, params = profile(
        model,
        inputs=(sample,),
        custom_ops={
            nn.BatchNorm2d: zero_ops,
            nn.SiLU: zero_ops,
            nn.AdaptiveAvgPool2d: zero_ops,
            nn.Dropout: zero_ops,
        },
        verbose=False,
    )
    total_macs = single_macs * args.models * args.tta_views
    print(f"input={tuple(sample.shape)}")
    print(f"params={int(params):,}")
    print(f"single_forward_mmac={single_macs / 1_000_000:.6f}")
    print(f"total_mmac={total_macs / 1_000_000:.6f}")
    print(f"under_100_mmac={total_macs / 1_000_000 < 100.0}")
    print()
    for models, tta_views in product((1, 2, 3), (1, 2)):
        current = single_macs * models * tta_views / 1_000_000
        print(
            f"models={models} tta_views={tta_views} "
            f"total_mmac={current:.6f} under_100_mmac={current < 100.0}"
        )


if __name__ == "__main__":
    main()

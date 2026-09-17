from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioDataset
from src.model import build_model


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats"}


def parse_variant(value: str) -> tuple[str, list[float], list[float]]:
    parts = value.split("=")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("variant must be NAME=CLASS_WEIGHTS=AZIMUTH_WEIGHTS")
    name, class_raw, azimuth_raw = parts
    return name, [float(x) for x in class_raw.split(",")], [float(x) for x in azimuth_raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoint]
    count = len(loaded)
    for name, class_weights, azimuth_weights in args.variant:
        if len(class_weights) != count or len(azimuth_weights) != count:
            raise ValueError(f"{name}: each weight list must have {count} entries")

    idx_to_class = {int(k): v for k, v in loaded[0]["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in loaded[0]["idx_to_azimuth"].items()}
    arches = [checkpoint["config"].get("arch", "baseline") for checkpoint in loaded]
    needs_gcc = any(arch in GCC_ARCHES for arch in arches)
    test = pd.read_csv("data/test.csv")
    config = loaded[0]["config"]
    dataset = AudioDataset(
        test,
        Path("data/test_audio"),
        config["sample_rate"],
        config["duration"],
        config["n_mels"],
        train=False,
        include_gcc=needs_gcc,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for checkpoint, arch in zip(loaded, arches):
        model = build_model(arch, len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(checkpoint["model"])
        models.append(model.eval())

    ids_all: list[str] = []
    class_parts = [[] for _ in models]
    azimuth_parts = [[] for _ in models]
    with torch.no_grad():
        for features, ids in tqdm(loader):
            features = features.to(device)
            ids_all.extend(ids)
            for index, (model, arch) in enumerate(zip(models, arches)):
                outputs = model(features if arch in GCC_ARCHES else features[:, :5])
                class_parts[index].append(outputs[0].cpu())
                azimuth_parts[index].append(outputs[1].cpu())

    class_logits = [torch.cat(parts) for parts in class_parts]
    azimuth_logits = [torch.cat(parts) for parts in azimuth_parts]
    # Architectures have different head scales. Global standardization makes
    # the task weights describe evidence rather than arbitrary logit magnitude.
    class_logits = [part / part.std().clamp_min(1e-6) for part in class_logits]
    azimuth_logits = [part / part.std().clamp_min(1e-6) for part in azimuth_logits]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, class_weights, azimuth_weights in args.variant:
        class_sum = sum(weight * logits for weight, logits in zip(class_weights, class_logits))
        azimuth_sum = sum(weight * logits for weight, logits in zip(azimuth_weights, azimuth_logits))
        class_pred = class_sum.argmax(1).tolist()
        azimuth_pred = azimuth_sum.argmax(1).tolist()
        frame = pd.DataFrame(
            {
                "id": ids_all,
                "sound_class": [idx_to_class[index] for index in class_pred],
                "azimuth": [idx_to_azimuth[index] for index in azimuth_pred],
            }
        )
        output = args.out_dir / f"submission_aggressive_{name}.csv"
        frame.to_csv(output, index=False)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()

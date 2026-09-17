from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioDataset
from src.model import build_model


def parse_weighted_checkpoint(value: str) -> tuple[Path, float]:
    if "=" not in value:
        return Path(value), 1.0
    path, weight = value.rsplit("=", 1)
    return Path(path), float(weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=parse_weighted_checkpoint, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/test_audio"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path for path, _ in args.checkpoint]
    weights = torch.tensor([weight for _, weight in args.checkpoint], dtype=torch.float32)
    if torch.any(weights <= 0):
        raise ValueError("All checkpoint weights must be positive.")
    weights = weights / weights.sum()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing audio dir: {args.audio_dir}")

    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    configs = [ckpt["config"] for ckpt in loaded]
    idx_to_class = {int(k): v for k, v in loaded[0]["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in loaded[0]["idx_to_azimuth"].items()}
    needs_gcc = any(config.get("arch") in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats") for config in configs)
    test_df = pd.read_csv(args.data_dir / "test.csv")
    dataset = AudioDataset(
        test_df,
        args.audio_dir,
        configs[0]["sample_rate"],
        configs[0]["duration"],
        configs[0]["n_mels"],
        train=False,
        include_gcc=needs_gcc,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    model_arches = []
    for ckpt in loaded:
        model_arches.append(ckpt["config"].get("arch", "baseline"))
        model = build_model(ckpt["config"].get("arch", "baseline"), len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
    weights = weights.to(device)

    rows = []
    with torch.no_grad():
        for x, ids in tqdm(loader):
            x = x.to(device)
            class_logits_sum = None
            azimuth_logits_sum = None
            for model, weight, arch in zip(models, weights, model_arches):
                outputs = model(x if arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats") else x[:, :5])
                class_logits, azimuth_logits = outputs[:2]
                class_logits = class_logits * weight
                azimuth_logits = azimuth_logits * weight
                class_logits_sum = class_logits if class_logits_sum is None else class_logits_sum + class_logits
                azimuth_logits_sum = azimuth_logits if azimuth_logits_sum is None else azimuth_logits_sum + azimuth_logits

            class_pred = class_logits_sum.argmax(1).cpu().tolist()
            azimuth_pred = azimuth_logits_sum.argmax(1).cpu().tolist()
            for sample_id, class_idx, azimuth_idx in zip(ids, class_pred, azimuth_pred):
                rows.append(
                    {
                        "id": sample_id,
                        "sound_class": idx_to_class[class_idx],
                        "azimuth": idx_to_azimuth[azimuth_idx],
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

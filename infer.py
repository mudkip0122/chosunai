from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioDataset
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/test_audio"))
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", choices=("none", "mirror"), default="none")
    return parser.parse_args()


def mirror_spatial_features(features: torch.Tensor) -> torch.Tensor:
    mirrored = features[:, (1, 0, 2, 3, 4)].clone()
    mirrored[:, 2].neg_()
    mirrored[:, 4].neg_()
    if features.shape[1] > 5:
        gcc = features[:, 5:].clone()
        gcc[:, 0, :, 0] = torch.flip(gcc[:, 0, :, 0], dims=(1,))
        mirrored = torch.cat((mirrored, gcc), dim=1)
    return mirrored


def mirror_azimuth_logits(logits: torch.Tensor, idx_to_azimuth: dict[int, int]) -> torch.Tensor:
    azimuth_to_idx = {angle: idx for idx, angle in idx_to_azimuth.items()}
    order = [azimuth_to_idx[-idx_to_azimuth[idx]] for idx in range(len(idx_to_azimuth))]
    return logits[:, order]


def main() -> None:
    args = parse_args()
    checkpoints = args.checkpoint or [Path("checkpoints/baseline.pt")]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing {args.audio_dir}. Extract test_audio.zip before inference.")

    loaded = [torch.load(checkpoint, map_location="cpu", weights_only=False) for checkpoint in checkpoints]
    config = loaded[0]["config"]
    idx_to_class = {int(k): v for k, v in loaded[0]["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in loaded[0]["idx_to_azimuth"].items()}

    test_df = pd.read_csv(args.data_dir / "test.csv")
    dataset = AudioDataset(
        test_df,
        args.audio_dir,
        config["sample_rate"],
        config["duration"],
        config["n_mels"],
        train=False,
        include_gcc=config.get("arch") in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt in loaded:
        model = build_model(ckpt["config"].get("arch", "baseline"), len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)

    rows = []
    with torch.no_grad():
        for x, ids in tqdm(loader):
            x = x.to(device)
            class_logits_sum = None
            azimuth_logits_sum = None
            for model in models:
                outputs = model(x)
                class_logits, azimuth_logits = outputs[:2]
                if args.tta == "mirror":
                    mirror_outputs = model(mirror_spatial_features(x))
                    mirror_class_logits, mirror_azimuth_logits_raw = mirror_outputs[:2]
                    class_logits = (class_logits + mirror_class_logits) * 0.5
                    azimuth_logits = (
                        azimuth_logits + mirror_azimuth_logits(mirror_azimuth_logits_raw, idx_to_azimuth)
                    ) * 0.5
                class_logits_sum = class_logits if class_logits_sum is None else class_logits_sum + class_logits
                azimuth_logits_sum = azimuth_logits if azimuth_logits_sum is None else azimuth_logits_sum + azimuth_logits
            class_logits = class_logits_sum / len(models)
            azimuth_logits = azimuth_logits_sum / len(models)
            class_pred = class_logits.argmax(1).cpu().tolist()
            azimuth_pred = azimuth_logits.argmax(1).cpu().tolist()
            for sample_id, c, a in zip(ids, class_pred, azimuth_pred):
                rows.append(
                    {
                        "id": sample_id,
                        "sound_class": idx_to_class[c],
                        "azimuth": idx_to_azimuth[a],
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioDataset
from src.model import build_model


def parse_named_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for each checkpoint.")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Checkpoint name cannot be empty.")
    return name, Path(path)


def parse_weight_spec(value: str) -> tuple[str, list[float], Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=w1:w2:w3=OUT.csv for each weight spec.")
    label, rest = value.split("=", 1)
    weights_text, out_text = rest.rsplit("=", 1)
    weights = [float(part) for part in weights_text.split(":")]
    if any(weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("Weights must be positive.")
    return label, weights, Path(out_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=parse_named_checkpoint, required=True)
    parser.add_argument("--weight", action="append", type=parse_weight_spec, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/test_audio"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [name for name, _ in args.checkpoint]
    paths = [path for _, path in args.checkpoint]

    for _, weights, _ in args.weight:
        if len(weights) != len(paths):
            raise ValueError(f"Expected {len(paths)} weights, got {len(weights)}.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Missing audio dir: {args.audio_dir}")

    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    configs = [ckpt["config"] for ckpt in loaded]
    idx_to_class = {int(k): v for k, v in loaded[0]["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in loaded[0]["idx_to_azimuth"].items()}
    test_df = pd.read_csv(args.data_dir / "test.csv")
    dataset = AudioDataset(
        test_df,
        args.audio_dir,
        configs[0]["sample_rate"],
        configs[0]["duration"],
        configs[0]["n_mels"],
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt in loaded:
        model = build_model(ckpt["config"].get("arch", "baseline"), len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)

    weight_sets = []
    for label, raw_weights, out_path in args.weight:
        tensor = torch.tensor(raw_weights, dtype=torch.float32, device=device)
        tensor = tensor / tensor.sum()
        weight_sets.append((label, tensor, out_path, []))

    with torch.no_grad():
        for x, ids in tqdm(loader):
            x = x.to(device)
            class_logits_by_model = []
            azimuth_logits_by_model = []
            for model in models:
                outputs = model(x)
                class_logits, azimuth_logits = outputs[:2]
                class_logits_by_model.append(class_logits)
                azimuth_logits_by_model.append(azimuth_logits)

            for _, weights, _, rows in weight_sets:
                class_logits_sum = None
                azimuth_logits_sum = None
                for model_index, weight in enumerate(weights):
                    class_part = class_logits_by_model[model_index] * weight
                    azimuth_part = azimuth_logits_by_model[model_index] * weight
                    class_logits_sum = class_part if class_logits_sum is None else class_logits_sum + class_part
                    azimuth_logits_sum = azimuth_part if azimuth_logits_sum is None else azimuth_logits_sum + azimuth_part

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

    for label, _, out_path, rows in weight_sets:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"wrote {out_path} ({label}: {', '.join(names)})")


if __name__ == "__main__":
    main()

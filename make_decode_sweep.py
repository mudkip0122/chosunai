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
    parser.add_argument("--class-temp", type=float, default=1.0)
    parser.add_argument("--azimuth-temp", type=float, default=1.0)
    parser.add_argument("--angle-score-weight", type=float, default=0.0)
    parser.add_argument(
        "--azimuth-decode",
        choices=("argmax", "smooth", "local_mean", "circular_mean"),
        default="argmax",
    )
    parser.add_argument("--smooth-sigma", type=float, default=8.0)
    parser.add_argument("--local-topk", type=int, default=3)
    return parser.parse_args()


def smooth_logits(logits: torch.Tensor, idx_to_azimuth: dict[int, int], sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return logits
    angles = torch.tensor(
        [idx_to_azimuth[i] for i in range(len(idx_to_azimuth))],
        dtype=torch.float32,
        device=logits.device,
    )
    distances = torch.abs(angles[:, None] - angles[None, :])
    kernel = torch.exp(-0.5 * (distances / sigma).square())
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    probs = torch.softmax(logits, dim=1)
    return torch.log((probs @ kernel.T).clamp_min(1e-12))


def nearest_azimuth_indices(values: torch.Tensor, idx_to_azimuth: dict[int, int]) -> torch.Tensor:
    angles = torch.tensor(
        [idx_to_azimuth[i] for i in range(len(idx_to_azimuth))],
        dtype=torch.float32,
        device=values.device,
    )
    distances = torch.abs(values[:, None] - angles[None, :])
    return distances.argmin(dim=1)


def decode_azimuth(
    logits: torch.Tensor,
    idx_to_azimuth: dict[int, int],
    mode: str,
    smooth_sigma: float,
    local_topk: int,
) -> torch.Tensor:
    if mode == "argmax":
        return logits.argmax(dim=1)
    if mode == "smooth":
        return smooth_logits(logits, idx_to_azimuth, smooth_sigma).argmax(dim=1)

    angles = torch.tensor(
        [idx_to_azimuth[i] for i in range(len(idx_to_azimuth))],
        dtype=torch.float32,
        device=logits.device,
    )
    probs = torch.softmax(logits, dim=1)

    if mode == "local_mean":
        k = max(1, min(local_topk, probs.size(1)))
        top_values, top_indices = probs.topk(k, dim=1)
        weighted_angles = (angles[top_indices] * top_values).sum(dim=1) / top_values.sum(dim=1).clamp_min(1e-12)
        return nearest_azimuth_indices(weighted_angles, idx_to_azimuth)

    if mode == "circular_mean":
        radians = torch.deg2rad(angles)
        x = (probs * torch.cos(radians)).sum(dim=1)
        y = (probs * torch.sin(radians)).sum(dim=1)
        mean_degrees = torch.rad2deg(torch.atan2(y, x))
        return nearest_azimuth_indices(mean_degrees, idx_to_azimuth)

    raise ValueError(f"Unknown azimuth decode mode: {mode}")


def angle_scores(angle_vectors: torch.Tensor, idx_to_azimuth: dict[int, int]) -> torch.Tensor:
    angles = torch.tensor(
        [idx_to_azimuth[i] for i in range(len(idx_to_azimuth))],
        dtype=torch.float32,
        device=angle_vectors.device,
    )
    radians = torch.deg2rad(angles)
    prototypes = torch.stack((torch.cos(radians), torch.sin(radians)), dim=1)
    return angle_vectors @ prototypes.T


def main() -> None:
    args = parse_args()
    paths = [path for _, path in args.checkpoint]
    names = [name for name, _ in args.checkpoint]

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
    needs_gcc = any(config.get("arch") in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled") for config in configs)
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
        include_gcc=needs_gcc,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    model_arches = []
    for ckpt in loaded:
        arch = ckpt["config"].get("arch", "baseline")
        model_arches.append(arch)
        model = build_model(arch, len(idx_to_class), len(idx_to_azimuth)).to(device)
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
            angle_scores_by_model = []
            for model, arch in zip(models, model_arches):
                outputs = model(x if arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled") else x[:, :5])
                class_logits, azimuth_logits = outputs[:2]
                class_logits_by_model.append(class_logits / args.class_temp)
                azimuth_logits_by_model.append(azimuth_logits / args.azimuth_temp)
                angle_scores_by_model.append(
                    angle_scores(outputs[2], idx_to_azimuth) if len(outputs) >= 3 else None
                )

            for _, weights, _, rows in weight_sets:
                class_logits_sum = None
                azimuth_logits_sum = None
                angle_scores_sum = None
                for model_index, weight in enumerate(weights):
                    class_part = class_logits_by_model[model_index] * weight
                    azimuth_part = azimuth_logits_by_model[model_index] * weight
                    class_logits_sum = class_part if class_logits_sum is None else class_logits_sum + class_part
                    azimuth_logits_sum = azimuth_part if azimuth_logits_sum is None else azimuth_logits_sum + azimuth_part
                    if args.angle_score_weight != 0.0 and angle_scores_by_model[model_index] is not None:
                        angle_part = angle_scores_by_model[model_index] * weight
                        angle_scores_sum = angle_part if angle_scores_sum is None else angle_scores_sum + angle_part

                class_pred = class_logits_sum.argmax(1).cpu().tolist()
                if angle_scores_sum is not None:
                    azimuth_logits_sum = azimuth_logits_sum + args.angle_score_weight * angle_scores_sum
                azimuth_pred = decode_azimuth(
                    azimuth_logits_sum,
                    idx_to_azimuth,
                    args.azimuth_decode,
                    args.smooth_sigma,
                    args.local_topk,
                ).cpu().tolist()
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
        print(
            f"wrote {out_path} "
            f"({label}: {', '.join(names)}; azimuth_decode={args.azimuth_decode}; "
            f"angle_score_weight={args.angle_score_weight})"
        )


if __name__ == "__main__":
    main()

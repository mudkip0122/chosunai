from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioDataset
from src.model import build_model
from infer import mirror_azimuth_logits, mirror_spatial_features


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
    parser.add_argument("--mode", choices=("class", "azimuth", "independent", "joint"), default="independent")
    parser.add_argument("--tta", choices=("none", "mirror"), default="none")
    return parser.parse_args()


def uniform_counts(total: int, labels: int) -> list[int]:
    base = total // labels
    counts = [base] * labels
    for index in range(total - base * labels):
        counts[index] += 1
    return counts


def quota_assign(scores: np.ndarray, counts: list[int]) -> np.ndarray:
    if scores.shape[0] != sum(counts):
        raise ValueError(f"Counts sum to {sum(counts)}, but there are {scores.shape[0]} samples.")

    expanded_labels = np.repeat(np.arange(scores.shape[1]), counts)
    expanded_scores = scores[:, expanded_labels]
    row_ind, col_ind = linear_sum_assignment(-expanded_scores)
    assigned = np.empty(scores.shape[0], dtype=np.int64)
    assigned[row_ind] = expanded_labels[col_ind]
    return assigned


def joint_quota_assign(class_logits: np.ndarray, azimuth_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_classes = class_logits.shape
    num_azimuths = azimuth_logits.shape[1]
    num_joint_labels = num_classes * num_azimuths
    counts = uniform_counts(num_samples, num_joint_labels)

    joint_scores = (
        class_logits[:, :, None] + azimuth_logits[:, None, :]
    ).reshape(num_samples, num_joint_labels)
    joint_pred = quota_assign(joint_scores, counts)
    return joint_pred // num_azimuths, joint_pred % num_azimuths


def collect_logits(args: argparse.Namespace) -> tuple[list[str], np.ndarray, np.ndarray, dict[int, str], dict[int, int]]:
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
    arch = configs[0].get("arch", "baseline")
    if any(config.get("arch", "baseline") != arch for config in configs):
        raise ValueError("All checkpoints must use the same architecture.")

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
        include_gcc=arch in ("gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt in loaded:
        model = build_model(arch, len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
    weights = weights.to(device)

    ids: list[str] = []
    class_batches = []
    azimuth_batches = []
    with torch.no_grad():
        for x, batch_ids in tqdm(loader):
            x = x.to(device)
            class_logits_sum = None
            azimuth_logits_sum = None
            for model, weight in zip(models, weights):
                outputs = model(x)
                class_logits, azimuth_logits = outputs[:2]
                if args.tta == "mirror":
                    mirror_class_logits, mirror_azimuth_logits_raw = model(mirror_spatial_features(x))
                    class_logits = (class_logits + mirror_class_logits) * 0.5
                    azimuth_logits = (
                        azimuth_logits + mirror_azimuth_logits(mirror_azimuth_logits_raw, idx_to_azimuth)
                    ) * 0.5
                class_logits = class_logits * weight
                azimuth_logits = azimuth_logits * weight
                class_logits_sum = class_logits if class_logits_sum is None else class_logits_sum + class_logits
                azimuth_logits_sum = azimuth_logits if azimuth_logits_sum is None else azimuth_logits_sum + azimuth_logits
            ids.extend(batch_ids)
            class_batches.append(class_logits_sum.cpu().numpy())
            azimuth_batches.append(azimuth_logits_sum.cpu().numpy())

    return ids, np.concatenate(class_batches), np.concatenate(azimuth_batches), idx_to_class, idx_to_azimuth


def main() -> None:
    args = parse_args()
    ids, class_logits, azimuth_logits, idx_to_class, idx_to_azimuth = collect_logits(args)

    if args.mode == "joint":
        class_pred, azimuth_pred = joint_quota_assign(class_logits, azimuth_logits)
    else:
        class_pred = class_logits.argmax(axis=1)
        azimuth_pred = azimuth_logits.argmax(axis=1)
        if args.mode in ("class", "independent"):
            class_pred = quota_assign(class_logits, uniform_counts(len(ids), class_logits.shape[1]))
        if args.mode in ("azimuth", "independent"):
            azimuth_pred = quota_assign(azimuth_logits, uniform_counts(len(ids), azimuth_logits.shape[1]))

    rows = [
        {
            "id": sample_id,
            "sound_class": idx_to_class[int(class_idx)],
            "azimuth": idx_to_azimuth[int(azimuth_idx)],
        }
        for sample_id, class_idx, azimuth_idx in zip(ids, class_pred, azimuth_pred)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

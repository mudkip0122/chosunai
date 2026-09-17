from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MonoWaveDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, audio_dir: Path) -> None:
        self.frame = frame.reset_index(drop=True)
        self.audio_dir = audio_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> torch.Tensor:
        wav, sample_rate = torchaudio.load(self.audio_dir / self.frame.iloc[index].filename)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        mono = wav.mean(0)
        if len(mono) < 32000:
            mono = torch.nn.functional.pad(mono, (0, 32000 - len(mono)))
        return mono[:32000]


def extract_embeddings(frame: pd.DataFrame, audio_dir: Path, batch_size: int, workers: int, mode: str) -> np.ndarray:
    loader = DataLoader(
        MonoWaveDataset(frame, audio_dir), batch_size=batch_size, shuffle=False,
        num_workers=workers, persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, hop_length=256, n_mels=64, power=2.0
    ).to(device)
    parts = []
    generator = torch.Generator(device=device).manual_seed(2026)
    projection = torch.randn(64 * 64, 512, generator=generator, device=device) / np.sqrt(512)
    with torch.no_grad():
        for wav in tqdm(loader):
            logmel = torch.log1p(mel(wav.to(device)) * 1000.0)
            # Remove gain and fixed spectral coloration (including most HRTF),
            # then use temporal-modulation magnitude for shift invariance.
            normalized = (logmel - logmel.mean(2, keepdim=True)) / logmel.std(2, keepdim=True).clamp_min(1e-4)
            if mode == "modulation":
                embedding = torch.fft.rfft(normalized, dim=2).abs()[:, :, 1:33]
                embedding = torch.log1p(embedding).flatten(1)
            else:
                # Retain fine event timing while compressing with a fixed random
                # projection. This separates true repeated sources from generic
                # stationary sounds that share modulation statistics.
                direct = torch.nn.functional.adaptive_avg_pool1d(normalized, 64).flatten(1)
                embedding = direct @ projection
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            parts.append(embedding.cpu())
    return torch.cat(parts).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding", type=Path, default=Path("cv_runs/source_fingerprint.npy"))
    parser.add_argument("--groups", type=Path, default=Path("cv_runs/source_groups.npy"))
    parser.add_argument("--threshold", type=float, default=0.965)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", choices=("modulation", "direct"), default="modulation")
    args = parser.parse_args()

    frame = pd.read_csv(args.data_dir / "train.csv")
    if args.embedding.exists():
        embeddings = np.load(args.embedding)
    else:
        embeddings = extract_embeddings(
            frame, args.data_dir / "train_audio", args.batch_size, args.num_workers, args.mode
        )
        args.embedding.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embedding, embeddings)

    parent = np.arange(len(frame))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nearest_similarities = []
    edge_count = 0
    for _, indices in frame.groupby("sound_class", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        count = min(args.neighbors + 1, len(indices))
        knn = NearestNeighbors(n_neighbors=count, metric="cosine", n_jobs=-1).fit(embeddings[indices])
        distances, neighbors = knn.kneighbors(embeddings[indices])
        nearest_similarities.extend((1.0 - distances[:, 1]).tolist())
        for row, source_index in enumerate(indices):
            for distance, neighbor_position in zip(distances[row, 1:], neighbors[row, 1:]):
                if 1.0 - distance >= args.threshold:
                    union(int(source_index), int(indices[neighbor_position]))
                    edge_count += 1

    roots = np.asarray([find(i) for i in range(len(frame))])
    _, groups = np.unique(roots, return_inverse=True)
    np.save(args.groups, groups.astype(np.int64))
    sizes = pd.Series(groups).value_counts()
    similarities = np.asarray(nearest_similarities)
    report = {
        "threshold": args.threshold,
        "mode": args.mode,
        "rows": len(frame),
        "groups": int(sizes.size),
        "multirow_groups": int((sizes > 1).sum()),
        "rows_in_multirow_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "edges": edge_count,
        "nearest_similarity_quantiles": {
            str(q): float(np.quantile(similarities, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
        },
    }
    args.groups.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

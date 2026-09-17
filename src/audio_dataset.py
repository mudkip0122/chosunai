from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio


def make_label_maps(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, int]]:
    classes = sorted(train_df["sound_class"].unique().tolist())
    azimuths = sorted(int(v) for v in train_df["azimuth"].unique().tolist())
    return {name: idx for idx, name in enumerate(classes)}, {value: idx for idx, value in enumerate(azimuths)}


def invert_map(mapping: Dict) -> Dict:
    return {value: key for key, value in mapping.items()}


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        sample_rate: int,
        duration: float,
        n_mels: int,
        class_to_idx: Optional[Dict[str, int]] = None,
        azimuth_to_idx: Optional[Dict[int, int]] = None,
        train: bool = False,
        background_noise_prob: float = 0.0,
        include_gcc: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.class_to_idx = class_to_idx
        self.azimuth_to_idx = azimuth_to_idx
        self.train = train
        self.background_noise_prob = background_noise_prob
        self.include_gcc = include_gcc

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=n_mels,
            f_min=50,
            f_max=min(7800, sample_rate // 2),
            power=2.0,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=50,
            f_max=min(7800, sample_rate // 2),
            n_stft=257,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.window = torch.hann_window(400)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        wav = self._load_audio(self.audio_dir / row["filename"])

        if self.train:
            wav = self._augment_waveform(wav)

        features = self._features(wav)

        if self.class_to_idx is None or self.azimuth_to_idx is None:
            return features, row["id"]

        class_target = self.class_to_idx[row["sound_class"]]
        azimuth_target = self.azimuth_to_idx[int(row["azimuth"])]
        return features, torch.tensor(class_target), torch.tensor(azimuth_target)

    def _load_audio(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if wav.shape[1] < self.num_samples:
            pad = self.num_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.shape[1] > self.num_samples:
            if self.train:
                start = random.randint(0, wav.shape[1] - self.num_samples)
            else:
                start = (wav.shape[1] - self.num_samples) // 2
            wav = wav[:, start : start + self.num_samples]

        return wav.clamp(-1.0, 1.0)

    def _augment_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(0.7, 1.3)
        wav = wav * gain

        if random.random() < 0.5:
            shift = random.randint(-self.sample_rate // 2, self.sample_rate // 2)
            wav = torch.roll(wav, shifts=shift, dims=1)

        if random.random() < 0.35:
            noise = torch.randn_like(wav) * random.uniform(0.001, 0.01)
            wav = wav + noise

        if random.random() < self.background_noise_prob:
            # Vary the background floor relative to each clip's signal level.
            signal_level = wav.abs().mean().clamp_min(1e-3)
            noise_level = signal_level * random.uniform(0.01, 0.08)
            wav = wav + torch.randn_like(wav) * noise_level

        return wav.clamp(-1.0, 1.0)

    def _features(self, wav: torch.Tensor) -> torch.Tensor:
        spectra = torch.stft(
            wav,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=self.window.to(wav.device),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spectra.abs().square()
        mel_power = self.mel_scale(power)
        mel_db = self.to_db(mel_power)

        common_reference = mel_db.max()
        log_mel = ((mel_db - common_reference).clamp(-80.0, 0.0) + 80.0) / 40.0 - 1.0

        cross = spectra[0] * spectra[1].conj()
        cross_real = self.mel_scale(cross.real)
        cross_imag = self.mel_scale(cross.imag)
        cross_norm = torch.sqrt(cross_real.square() + cross_imag.square()).clamp_min(1e-6)
        cos_ipd = cross_real / cross_norm
        sin_ipd = cross_imag / cross_norm
        ild = (mel_db[0] - mel_db[1]).clamp(-30.0, 30.0) / 30.0

        feature_channels = [log_mel[0], log_mel[1], ild, cos_ipd, sin_ipd]
        if self.include_gcc:
            # Frame-averaged GCC-PHAT around the physically plausible binaural
            # lag range. Store the 64-vector in one column of a sixth plane so
            # the regular tensor/DataLoader interface remains unchanged.
            phat = cross / cross.abs().clamp_min(1e-6)
            gcc = torch.fft.irfft(phat.mean(dim=1), n=512)
            gcc = torch.cat((gcc[-32:], gcc[:32]))
            gcc = gcc / gcc.norm().clamp_min(1e-6)
            gcc_plane = torch.zeros_like(ild)
            gcc_plane[:, 0] = gcc
            feature_channels.append(gcc_plane)

        mel = torch.stack(feature_channels, dim=0)
        mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=-1.0).to(torch.float32)

        if self.train and random.random() < 0.5:
            augmented = self._spec_augment(mel[:5])
            mel[:5] = augmented

        return mel

    @staticmethod
    def _spec_augment(mel: torch.Tensor) -> torch.Tensor:
        _, freq, time = mel.shape
        if freq > 12:
            f = random.randint(0, min(10, freq // 5))
            f0 = random.randint(0, max(0, freq - f))
            mel[:, f0 : f0 + f, :] = 0
        if time > 12:
            t = random.randint(0, min(16, time // 6))
            t0 = random.randint(0, max(0, time - t))
            mel[:, :, t0 : t0 + t] = 0
        return mel

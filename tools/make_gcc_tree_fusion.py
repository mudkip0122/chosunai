from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.audio_dataset import AudioDataset
from src.model import build_model


GCC_ARCHES = {"gcc", "gcc_temporal", "gcc_class", "gcc_stats", "disentangled"}


class TestWithTreeFeatures(Dataset):
    def __init__(self, frame: pd.DataFrame, audio_dir: Path) -> None:
        self.frame = frame.reset_index(drop=True)
        self.base = AudioDataset(
            self.frame,
            audio_dir,
            sample_rate=16000,
            duration=2.0,
            n_mels=64,
            train=False,
            include_gcc=True,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        wav = self.base._load_audio(self.base.audio_dir / row["filename"])
        features = self.base._features(wav)
        # The cached training vector contains the 64-bin frame-averaged
        # GCC-PHAT representation plus global log left/right energy ratio.
        log_energy_ratio = torch.log(
            (wav[0].square().mean() + 1e-9) / (wav[1].square().mean() + 1e-9)
        )
        tree_features = torch.cat((features[5, :, 0], log_energy_ratio.view(1)))
        return features, tree_features, str(row["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gcc-features", type=Path, default=Path("cv_runs/gcc_features.npy"))
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--model-weights", type=float, nargs="+", default=(3.0, 2.0, 2.0))
    parser.add_argument("--class-model-weights", type=float, nargs="+", default=None)
    parser.add_argument("--azimuth-model-weights", type=float, nargs="+", default=None)
    parser.add_argument("--tree-weight", type=float, action="append", default=None)
    parser.add_argument("--conditional-margin-quantile", type=float, default=None)
    parser.add_argument("--circular-regression", action="store_true")
    parser.add_argument("--circular-global-weight", type=float, default=0.20)
    parser.add_argument("--circular-weight", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gcc_tree"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if len(args.checkpoint) != len(args.model_weights):
        raise ValueError("checkpoint and model-weights lengths must match")
    class_model_weights = args.class_model_weights or args.model_weights
    azimuth_model_weights = args.azimuth_model_weights or args.model_weights
    if len(args.checkpoint) != len(class_model_weights):
        raise ValueError("checkpoint and class-model-weights lengths must match")
    if len(args.checkpoint) != len(azimuth_model_weights):
        raise ValueError("checkpoint and azimuth-model-weights lengths must match")
    tree_weights = args.tree_weight or [0.0, 0.05, 0.1, 0.15]

    train = pd.read_csv(args.data_dir / "train.csv")
    angles = np.sort(train["azimuth"].astype(int).unique())
    angle_to_idx = {int(angle): index for index, angle in enumerate(angles)}
    targets = train["azimuth"].map(angle_to_idx).to_numpy()
    cached = np.load(args.gcc_features)
    train_features = np.concatenate((cached[:, :64], cached[:, 65:66]), axis=1)
    tree = ExtraTreesClassifier(
        n_estimators=600,
        max_features=1.0,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=5,
        n_jobs=-1,
    )
    tree.fit(train_features, targets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tree, args.out_dir / "gcc_azimuth_extratrees.joblib", compress=3)
    circular_regressor = None
    if args.circular_regression:
        angle_radians = np.deg2rad(train["azimuth"].astype(float).to_numpy())
        circular_targets = np.stack(
            (np.cos(angle_radians), np.sin(angle_radians)), axis=1
        )
        circular_regressor = RandomForestRegressor(
            n_estimators=500,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=5,
            n_jobs=-1,
        )
        circular_regressor.fit(train_features, circular_targets)
        joblib.dump(
            circular_regressor,
            args.out_dir / "gcc_circular_rf2.joblib",
            compress=3,
        )

    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoint]
    idx_to_class = {int(k): v for k, v in loaded[0]["idx_to_class"].items()}
    idx_to_azimuth = {int(k): int(v) for k, v in loaded[0]["idx_to_azimuth"].items()}
    arches = [checkpoint["config"].get("arch", "baseline") for checkpoint in loaded]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for checkpoint, arch in zip(loaded, arches):
        model = build_model(arch, len(idx_to_class), len(idx_to_azimuth)).to(device)
        model.load_state_dict(checkpoint["model"])
        models.append(model.eval())
    class_weights = torch.tensor(class_model_weights, dtype=torch.float32, device=device)
    class_weights /= class_weights.sum()
    azimuth_weights = torch.tensor(azimuth_model_weights, dtype=torch.float32, device=device)
    azimuth_weights /= azimuth_weights.sum()

    test = pd.read_csv(args.data_dir / "test.csv")
    loader = DataLoader(
        TestWithTreeFeatures(test, args.data_dir / "test_audio"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    ids: list[str] = []
    class_model_parts = [[] for _ in models]
    azimuth_model_parts = [[] for _ in models]
    tree_parts, tree_margin_parts, circular_parts = [], [], []
    with torch.no_grad():
        for features, tree_features, batch_ids in tqdm(loader):
            features = features.to(device)
            for model_index, (model, arch) in enumerate(zip(models, arches)):
                model_input = features if arch in GCC_ARCHES else features[:, :5]
                class_logits, azimuth_logits = model(model_input)[:2]
                class_model_parts[model_index].append(class_logits.cpu())
                azimuth_model_parts[model_index].append(azimuth_logits.cpu())
            ids.extend(batch_ids)
            probability = tree.predict_proba(tree_features.numpy())
            tree_parts.append(torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0))))
            sorted_probability = np.sort(probability, axis=1)
            tree_margin_parts.append(
                torch.from_numpy(sorted_probability[:, -1] - sorted_probability[:, -2])
            )
            if circular_regressor is not None:
                vector = circular_regressor.predict(tree_features.numpy())
                predicted_angle = np.arctan2(vector[:, 1], vector[:, 0])
                candidate_angles = np.deg2rad(
                    [idx_to_azimuth[index] for index in range(len(idx_to_azimuth))]
                )
                circular_parts.append(
                    torch.from_numpy(
                        np.cos(predicted_angle[:, None] - candidate_angles[None, :])
                    )
                )

    # Match validation exactly: standardize each model head over the full split,
    # then apply the independently tuned class and azimuth ensemble weights.
    class_logits = None
    azimuth_logits = None
    for model_index, (class_weight, azimuth_weight) in enumerate(
        zip(class_weights.cpu(), azimuth_weights.cpu())
    ):
        model_class_logits = torch.cat(class_model_parts[model_index])
        model_azimuth_logits = torch.cat(azimuth_model_parts[model_index])
        model_class_logits /= model_class_logits.std().clamp_min(1e-6)
        model_azimuth_logits /= model_azimuth_logits.std().clamp_min(1e-6)
        weighted_class = model_class_logits * class_weight
        weighted_azimuth = model_azimuth_logits * azimuth_weight
        class_logits = weighted_class if class_logits is None else class_logits + weighted_class
        azimuth_logits = weighted_azimuth if azimuth_logits is None else azimuth_logits + weighted_azimuth
    tree_logits = torch.cat(tree_parts).float()
    tree_margin = torch.cat(tree_margin_parts).float()
    azimuth_logits = (azimuth_logits - azimuth_logits.mean()) / azimuth_logits.std().clamp_min(1e-6)
    tree_logits = (tree_logits - tree_logits.mean()) / tree_logits.std().clamp_min(1e-6)
    class_pred = class_logits.argmax(1).tolist()

    for tree_weight in tree_weights:
        azimuth_pred = (azimuth_logits + tree_weight * tree_logits).argmax(1).tolist()
        output = pd.DataFrame(
            {
                "id": ids,
                "sound_class": [idx_to_class[index] for index in class_pred],
                "azimuth": [idx_to_azimuth[index] for index in azimuth_pred],
            }
        )
        suffix = f"{tree_weight:.2f}".replace(".", "")
        path = args.out_dir / f"submission_gcc_tree_w{suffix}.csv"
        output.to_csv(path, index=False)
        print(f"wrote {path}")

    if args.conditional_margin_quantile is not None:
        if not 0.0 <= args.conditional_margin_quantile <= 1.0:
            raise ValueError("conditional-margin-quantile must be between 0 and 1")
        pred_low = (azimuth_logits + 0.15 * tree_logits).argmax(1)
        pred_high = (azimuth_logits + 0.30 * tree_logits).argmax(1)
        changed = pred_low != pred_high
        threshold = torch.quantile(tree_margin[changed], args.conditional_margin_quantile)
        selected = changed & (tree_margin >= threshold)
        pred_conditional = pred_low.clone()
        pred_conditional[selected] = pred_high[selected]
        output = pd.DataFrame(
            {
                "id": ids,
                "sound_class": [idx_to_class[index] for index in class_pred],
                "azimuth": [idx_to_azimuth[index] for index in pred_conditional.tolist()],
            }
        )
        quantile_suffix = f"{args.conditional_margin_quantile:.2f}".replace(".", "")
        path = args.out_dir / f"submission_gcc_tree_condmargin_q{quantile_suffix}_w015_w030.csv"
        output.to_csv(path, index=False)
        print(
            f"wrote {path} (selected {int(selected.sum())}/{int(changed.sum())} "
            f"w015-to-w030 changes, margin threshold={float(threshold):.6f})"
        )

    if circular_regressor is not None:
        circular_logits = torch.cat(circular_parts).float()
        circular_logits = (
            circular_logits - circular_logits.mean()
        ) / circular_logits.std().clamp_min(1e-6)
        prediction = (
            azimuth_logits
            + args.circular_global_weight * tree_logits
            + args.circular_weight * circular_logits
        ).argmax(1)
        output = pd.DataFrame(
            {
                "id": ids,
                "sound_class": [idx_to_class[index] for index in class_pred],
                "azimuth": [idx_to_azimuth[index] for index in prediction.tolist()],
            }
        )
        global_suffix = f"{args.circular_global_weight:.2f}".replace(".", "")
        circular_suffix = f"{args.circular_weight:.2f}".replace(".", "")
        path = args.out_dir / (
            f"submission_gcc_tree_w{global_suffix}_circularrf2_w{circular_suffix}.csv"
        )
        output.to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

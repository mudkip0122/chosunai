# Chosun AI Audio Challenge Baseline

This is a scratch-trained stereo audio baseline for joint sound-class and azimuth prediction.

## Unified Competition Pipeline

`competition_pipeline.py` provides one entry point for training, MMAC checking,
and submission generation. The default `disentangled` model uses stereo
log-mel, ILD, cosine/sine IPD, and GCC-PHAT features with separate lightweight
sound-class and localization trunks.

Validation training:

```powershell
.\.conda\envs\chosun-audio\python.exe competition_pipeline.py train --out checkpoints\disentangled_valid.pt --seed 5
```

Full-data training for two seeds:

```powershell
.\.conda\envs\chosun-audio\python.exe competition_pipeline.py train --validation-fraction 0 --out checkpoints\full_disentangled_seed5.pt --seed 5
.\.conda\envs\chosun-audio\python.exe competition_pipeline.py train --validation-fraction 0 --out checkpoints\full_disentangled_seed17.pt --seed 17
```

Check the two-model inference budget and create the submission:

```powershell
.\.conda\envs\chosun-audio\python.exe competition_pipeline.py mmac --arch disentangled --models 2
.\.conda\envs\chosun-audio\python.exe competition_pipeline.py infer --checkpoint checkpoints\full_disentangled_seed5.pt --checkpoint checkpoints\full_disentangled_seed17.pt --out outputs\submission.csv
```

The inference command refuses configurations at or above 100 MMAC per audio
and validates the output row count, IDs, and null values before writing CSV.

## Expected Data

Place files under `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `class_map.csv`
- `train_audio.zip`
- `test_audio.zip`

If Kaggle API authentication is working, download competition files directly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_kaggle_data.ps1 -Competition <competition-slug>
```

Then extract audio:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/unzip_data.ps1
```

Expected extracted folders:

- `data/train_audio/*.wav`
- `data/test_audio/*.wav`

## Install

```powershell
.\.conda\envs\chosun-audio\python.exe -m pip install -r requirements.txt
```

## Train

```powershell
.\.conda\envs\chosun-audio\python.exe train.py --epochs 20 --batch-size 32 --num-workers 2
```

The best checkpoint is saved to `checkpoints/baseline.pt`.

Full-train a final submission seed without a validation split:

```powershell
.\.conda\envs\chosun-audio\python.exe train_full.py --arch baseline --epochs 30 --lr 1e-3 --seed 5 --batch-size 32 --num-workers 2 --out checkpoints\full_seed5.pt
```

## Inference

```powershell
.\.conda\envs\chosun-audio\python.exe infer.py --checkpoint checkpoints/baseline.pt --out outputs/submission.csv
```

Mirror TTA:

```powershell
.\.conda\envs\chosun-audio\python.exe infer.py --checkpoint checkpoints/baseline.pt --tta mirror --out outputs/submission_tta.csv
```

Two-seed ensemble:

```powershell
.\.conda\envs\chosun-audio\python.exe infer.py --checkpoint checkpoints/seed17.pt --checkpoint checkpoints/seed42.pt --out outputs/submission_ensemble.csv
```

## Validate Submission Format

```powershell
.\.conda\envs\chosun-audio\python.exe validate_submission.py --submission outputs/submission.csv
```

## Cross Validation

Run repeated K-fold validation before spending daily submissions:

```powershell
.\.conda\envs\chosun-audio\python.exe cross_validate.py --arch baseline --folds 5 --repeats 2 --epochs 30 --lr 1e-3 --batch-size 32 --num-workers 2 --out-dir cv_runs\baseline_5x2
```

The run writes `fold_summary.csv`, `oof_predictions.csv`, `summary.json`, and class/azimuth error reports under the chosen `cv_runs` directory.

For staged runs, use `--start-split` and `--max-splits`:

```powershell
.\.conda\envs\chosun-audio\python.exe cross_validate.py --arch baseline --folds 5 --repeats 2 --epochs 30 --lr 1e-3 --start-split 1 --max-splits 1 --out-dir cv_runs\baseline_5x2
```

## Check MMAC

```powershell
.\.conda\envs\chosun-audio\python.exe calculate_mmac.py
```

Check a planned inference scenario:

```powershell
.\.conda\envs\chosun-audio\python.exe calculate_mmac.py --models 2 --tta-views 1
```

## Baseline Notes

- Input feature: 5-channel spatial mel feature: left log-mel, right log-mel, ILD, cos(IPD), sin(IPD).
- Outputs: one head for `sound_class`, one head for `azimuth`.
- New `gcc` architecture: the baseline CNN plus a GCC-PHAT interaural-delay
  vector head. This preserves the physical time-delay cue that mel pooling loses.
- Loss: `CrossEntropy(sound_class) + CrossEntropy(azimuth)`.
- Validation split: 80/20 with combined `sound_class_azimuth` stratification when possible.
- Augmentation: random gain, time shift, low-level noise, and light SpecAugment.

Warm-start and validate the GCC-PHAT head while keeping a baseline fixed:

```powershell
.\.conda\envs\chosun-audio\python.exe train.py --arch gcc --init-checkpoint checkpoints\seed5.pt --freeze-base --no-augment --epochs 6 --lr 3e-3 --seed 5 --batch-size 128 --num-workers 2 --out checkpoints\gcc_seed5.pt
```

# chosunai

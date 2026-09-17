# Chosun AI Challenge Handoff

Last updated: 2026-09-17

For the immediate answer to "what should I do now?", read `NEXT_ACTION.md` first.

## Current Best

- Known best public LB: `0.72613`
- File: `outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv`
- `outputs/gcc_circnoise/submission_gcc_tree_w025.csv` scored only `0.71119`.
  Reject it; it changed 604 classes and exposed optimistic/inconsistent random-
  split validation.
- `outputs/gcc_tree/submission_gcc_tree_marginq050_confq050_and.csv` scored
  `0.72538`, below the `0.72613` best by `0.00075`; reject it. The 14 reverted
  azimuth corrections were not beneficial on the public hidden set.
- No evidence-backed unsubmitted candidate is currently ready. Stop nearby
  tree gates and retain the `0.72613` best while developing domain-shift-aware
  validation.
- `outputs/domain_robust/submission_gcc_tree_condmargin_q050_w015_w030.csv`
  scored `0.72320`, below the best by `0.00293`. Its apparent domain-validation
  gain (`0.724781 -> 0.737091`) did not transfer; reject it.
- Method: 3:2:2 deep ensemble plus conditional GCC/ILD ExtraTrees azimuth correction, no TTA
- GCC-PHAT improved the previous best `0.72190` by `0.00041`; useful, but not a path to `0.77` through small tuning alone.
- `outputs/submission_ens_gcc5_circ29_angle37_w322_anglescore050.csv` was
  submitted and tied the best at `0.72231`.
- `outputs/gcc_tree/submission_gcc_tree_w015.csv` scored `0.72573`, improving
  the former best by `+0.00342`.
- `outputs/gcc_tree/submission_gcc_tree_w030.csv` scored `0.72434`, below the best;
  unconditional stronger tree correction is rejected.
- `outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv` scored
  `0.72613`, improving the former best by `+0.00040`; it is the new best.
- Do not submit nearby weights or thresholds merely to use a slot.
- This exactly preserves the current-best class predictions and adds a small
  ExtraTrees GCC/ILD correction to azimuth logits. Weight `0.15` improved two
  validation splits (`0.73378 -> 0.73513`, `0.70724 -> 0.70971`) and changes
  only `5.56%` of test azimuths.
- New disentangled model separates class acoustics from spatial/GCC localization.
  With fixed settings it scored `0.80508` and `0.78884` on two different
  stratified validation splits (mean `0.79696`). The submission averages two
  full-data seeds and uses angle score weight `0.25` at `95.677888` MMAC.
  It scored only `0.71911` on public LB, below the `0.72231` best. The random
  row-wise validation protocol is therefore not trustworthy for model selection.
  It uses the new temporal-statistics class head aggressively while keeping
  azimuth changes small through GCC + circular-soft localization.
- Controlled class/azimuth fusion validation improved from GCC-only `0.73378`
  to `0.75656`. The final candidate changes class `13.82%`, azimuth `4.36%`,
  and joint `17.34%` versus the current best; format validated at `94.478368` MMAC.
- Latest attack work: trained `checkpoints/full_seed41_angle_soft8_ls003_aw025.pt` with final train joint F1 `0.8077`.
- `outputs/submission_ens_circ29_angle37_angle41_w233.csv` scored `0.71521`; high-shift angle-pair attacks should be deprioritized.
- `outputs/submission_ens_fulls5_circ29_angle37_w322_azsmooth8.csv` scored `0.72102`; broad azimuth smoothing should be deprioritized.
- Previous best `outputs/submission_ens_fulls5_fulls17_circ29_w322.csv` scored `0.72029`.
- Previous best `outputs/submission_ens_fulls5_fulls17_s23_w322.csv` scored `0.71707`.
- `outputs/submission_ens_fulls5_fulls17_s23_w321.csv` previously scored `0.71398`.
- `outputs/submission_ens_fulls5_fulls17_s23_w311.csv` previously scored `0.71060`.
- `outputs/submission_ens_s5_s17_s23_w211.csv` previously scored `0.70485`.
- `outputs/submission_v2_seed42.csv` was submitted and scored `0.66241`, despite validation F1 `0.72313`.
- `outputs/submission_v2_seed7_soft8_ls003.csv` scored `0.68034`, despite validation F1 `0.75245`.
- `outputs/submission_v3_seed42_tta.csv` scored `0.65812`.
- `outputs/submission_ensemble_seed17_seed42_seed23.csv` scored `0.70407`.

## Aggressive 0.77 Attack Prepared

- The former primary file,
  `outputs/submission_aggressive_gccstats47_gcc5_circ29_class521_az032.csv`,
  scored `0.71852`, below the `0.72231` best.
- Added `gcc_stats`, using temporal mean/std/max/attention for class recognition,
  GCC-PHAT localization, a richer azimuth head, and an auxiliary angle head.
- Controlled same-split fusion improved from GCC-only `0.73378` to `0.75656`.
- Full checkpoint: `checkpoints/full_gcc_stats_seed47.pt`; final class accuracy
  `0.8759`, azimuth accuracy `0.6404`, train joint F1 `0.8433`.
- Candidate members: gcc_stats47 + GCC5 + circ29. Standardized-logit task weights:
  class `5:2:1`, azimuth `0:3:2`.
- Difference versus confirmed best: class `13.82%`, azimuth `4.36%`, joint
  `17.34%`. Format validated; exact compute `94.478368` MMAC.
- Remaining candidates in order: `outputs/aggressive_gc/submission_aggressive_class321_az032.csv`,
  then `outputs/aggressive_gc/submission_aggressive_class221_az032.csv`.

## GCC-PHAT Result

- `outputs/submission_ens_gcc5_circ29_angle37_w322.csv`
  - Public LB: `0.72231`; submitted; current best.
  - Previous best: `0.72190` from `submission_ens_fulls5_circ29_angle37_w322.csv`.
  - Controlled validation improved from `0.72931` to `0.73378`; azimuth
    accuracy improved from `0.64042` to `0.68500`.
  - `94.151136` MMAC; format validated.

## Rejected Class-Recognition Follow-ups

- `gcc_temporal` attention/statistics head: validation joint F1 `0.73161`, below
  GCC-only `0.73378`.
- `gcc_class` independent mono CNN: validation class accuracy `0.77333`, below
  baseline `0.77708`.
- Baseline class-only fine-tuning: validation class accuracy `0.77958`; gain too
  small and joint score lower, so no full-train submission was made.
- Time-frequency ExtraTrees/RBF-SVM: best validation class accuracy `0.65833`.
- Do not submit these variants without new evidence. The current public best
  remains GCC-PHAT w322 at `0.72231`.

## Candidates

- `outputs/submission_ens_s5_s23_s42_eq.csv`
  - Three baseline checkpoints, logits averaged, no TTA
  - Members: seed5 + seed23 + seed42
  - seed5 validation joint F1: `0.72931`
  - `94.116192` MMAC
  - Submission format validated

- `outputs/submission_ens_s5_s17_s23_eq.csv`
  - Three baseline checkpoints, logits averaged, no TTA
  - Members: seed5 + seed17 + seed23
  - `94.116192` MMAC
  - Submission format validated

- `outputs/submission_ens_s5_s17_s42_eq.csv`
  - Three baseline checkpoints, logits averaged, no TTA
  - Members: seed5 + seed17 + seed42
  - `94.116192` MMAC
  - Submission format validated

- `outputs/submission_ens_s5_s23_s42_w122.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: seed5:seed23:seed42 = `1:2:2`
  - `94.116192` MMAC
  - Submission format validated

- `outputs/submission_ens_s17_s23_s42_w122.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: seed17:seed23:seed42 = `1:2:2`
  - Public LB: `0.69537`
  - `94.116192` MMAC
  - Submission format validated
  - Submitted; worse than the current equal seed17+seed23+seed42 ensemble

- `outputs/submission_ens_s11_s23_s42_eq.csv`
  - Three baseline checkpoints, logits averaged, no TTA
  - Members: seed11 + seed23 + seed42
  - seed11 validation joint F1: `0.71814`
  - `94.116192` MMAC
  - Submission format validated

- `outputs/submission_v2_seed42.csv`
  - v2 wider CNN, Squeeze-Excitation, average/max pooling
  - Validation joint F1: `0.72313`
  - Public LB: `0.66241`
  - No TTA; `68.893248` MMAC
  - Not recommended over the current baseline TTA fallback

- `outputs/submission_v3_seed42_tta.csv`
  - Compact v2-style CNN with mirror TTA
  - Validation joint F1: `0.69416`
  - Public LB: `0.65812`
  - `79.539264` MMAC including TTA
  - Submitted; not recommended over baseline TTA

- `outputs/submission_v2_seed7_soft8_ls003.csv`
  - v2 with azimuth soft targets (`sigma=8`) and class label smoothing (`0.03`)
  - Validation joint F1: `0.75245`
  - Public LB: `0.68034`
  - No TTA; `68.893248` MMAC
  - Submission format validated
  - Submitted; not recommended over baseline TTA

- `outputs/submission_ensemble_seed17_seed42_seed23.csv`
  - Three baseline checkpoints, logits averaged, no TTA
  - `94.116192` MMAC
  - Public LB: `0.70407`
  - Previous baseline ensemble best before seed5/full-train variants

- `outputs/submission_ens_s5_s23_s42_w211.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: seed5:seed23:seed42 = `2:1:1`
  - Public LB: `0.70132`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs `submission_ensemble_seed17_seed42_seed23.csv`: class `9.98%`, azimuth `18.84%`, joint `27.06%`
  - Submitted; close to `0.70407` but below it

- `outputs/submission_ens_s5_s17_s23_w211.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: seed5:seed17:seed23 = `2:1:1`
  - Public LB: `0.70485`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs `submission_ensemble_seed17_seed42_seed23.csv`: class `9.84%`, azimuth `18.76%`, joint `26.80%`
  - Submitted; previous best before full-train ensemble

- `outputs/submission_ens_s5_s17_s42_w211.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: seed5:seed17:seed42 = `2:1:1`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs `submission_ensemble_seed17_seed42_seed23.csv`: class `9.56%`, azimuth `18.74%`, joint `26.60%`

- `outputs/submission_ens_fulls5_fulls17_s23_w311.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:1:1`
  - Public LB: `0.71060`
  - `94.116192` MMAC
  - Submission format validated
  - Submitted; previous best before w321

- `outputs/submission_ens_fulls5_fulls17_s23_w321.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:2:1`
  - Public LB: `0.71398`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w311: class `2.58%`, azimuth `5.16%`, joint `7.62%`
  - Submitted; previous best before w322

- `outputs/submission_ens_fulls5_fulls17_s23_w312.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:1:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w311: class `3.08%`, azimuth `4.80%`, joint `7.68%`

- `outputs/submission_ens_fulls5_fulls17_s23_w212.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `2:1:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w311: class `4.80%`, azimuth `7.96%`, joint `12.38%`

- `outputs/submission_ens_fulls5_fulls17_s23_w421.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `4:2:1`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w311: class `1.58%`, azimuth `3.28%`, joint `4.80%`

- `outputs/submission_ens_fulls5_fulls17_s23_w511.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `5:1:1`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w311: class `2.68%`, azimuth `4.04%`, joint `6.54%`

- `outputs/submission_ens_fulls5_fulls17_s23_w431.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `4:3:1`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w321: class `0.90%`, azimuth `1.66%`, joint `2.54%`

- `outputs/submission_ens_fulls5_fulls17_s23_w331.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:3:1`
  - Public LB: `0.71312`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w321: class `1.80%`, azimuth `3.48%`, joint `5.16%`
  - Submitted; slightly below w321

- `outputs/submission_ens_fulls5_fulls17_s23_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:2:2`
  - Public LB: `0.71707`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w321: class `2.20%`, azimuth `4.08%`, joint `6.20%`
  - Submitted; previous best before circular-soft seed29 improved to `0.72029`

- `checkpoints/full_seed23.pt`
  - Full-train baseline checkpoint, seed `23`
  - Epochs: 30, LR: `1e-3`
  - Final train joint F1: `0.7951`
  - Created to replace validation-split `checkpoints/seed23.pt` in the best ensemble family

- `outputs/submission_ens_fulls5_fulls17_fulls23_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:full_seed23 = `3:2:2`
  - Public LB: `0.71570`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322 with non-full seed23: class `5.72%`, azimuth `10.54%`, joint `15.64%`
  - Submitted; below current best, so full_seed23 replacement is lower priority

- `outputs/submission_ens_fulls5_fulls17_fulls23_w433.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:full_seed23 = `4:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322 with non-full seed23: class `5.82%`, azimuth `10.96%`, joint `16.16%`

- `outputs/submission_ens_fulls5_fulls17_fulls23_w323.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:full_seed23 = `3:2:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322 with non-full seed23: class `6.56%`, azimuth `12.66%`, joint `18.30%`

- `outputs/submission_ens_fulls5_fulls17_fulls23_w423.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:full_seed23 = `4:2:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322 with non-full seed23: class `6.52%`, azimuth `11.86%`, joint `17.58%`

- `outputs/submission_ens_fulls5_fulls17_fulls23_eq.csv`
  - Three full-train baseline checkpoints, logits averaged, no TTA
  - Members: full_seed5 + full_seed17 + full_seed23
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322 with non-full seed23: class `6.20%`, azimuth `12.14%`, joint `17.54%`

- `outputs/submission_ens_fulls5_fulls17_s23_w323.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:2:3`
  - Public LB: `0.71332`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `1.54%`, azimuth `2.88%`, joint `4.38%`
  - Submitted; below w322, so seed23 above the `3:2:2` ratio appears worse

- `checkpoints/full_seed42.pt`
  - Full-train baseline checkpoint, seed `42`
  - Epochs: 30, LR: `1e-3`
  - Final train joint F1: `0.7938`
  - Created after daily submission slots were exhausted

- `outputs/submission_ens_fulls5_s23_fulls42_w332.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:seed23:full_seed42 = `3:3:2`
  - Public LB: `0.71439`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `5.62%`, azimuth `12.10%`, joint `16.82%`
  - Submitted; below current best, so move to the next full_seed42 candidate

- `outputs/submission_ens_fulls5_fulls17_fulls42_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:full_seed42 = `3:2:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `6.12%`, azimuth `11.94%`, joint `17.16%`
  - Deprioritized after `outputs/submission_ens_fulls5_s23_fulls42_w332.csv` scored only `0.71439`

- `outputs/submission_ens_fulls5_fulls17_fulls42_w433.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:full_seed42 = `4:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `6.28%`, azimuth `12.36%`, joint `17.66%`

- `outputs/submission_ens_fulls5_s23_fulls42_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:seed23:full_seed42 = `3:2:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `6.06%`, azimuth `12.42%`, joint `17.66%`

- `outputs/submission_ens_fulls17_s23_fulls42_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed17:seed23:full_seed42 = `3:2:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `7.58%`, azimuth `16.62%`, joint `22.94%`

- `outputs/submission_ens_fulls5_fulls17_s23_w423.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `4:2:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `1.26%`, azimuth `2.52%`, joint `3.76%`

- `outputs/submission_ens_fulls5_fulls17_s23_w332.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `3:3:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `1.54%`, azimuth `3.20%`, joint `4.64%`

- `outputs/submission_ens_fulls5_fulls17_s23_w433.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `4:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `0.58%`, azimuth `0.86%`, joint `1.44%`
  - Very close probe around w322

- `outputs/submission_ens_fulls5_fulls17_s23_w533.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `5:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `0.54%`, azimuth `1.12%`, joint `1.64%`
  - Very close probe around w322

- `outputs/submission_ens_fulls5_fulls17_s23_w654.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `6:5:4`
  - Public LB: `0.71564`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `0.80%`, azimuth `1.62%`, joint `2.42%`
  - Submitted; below current best, suggesting nearby weight-only probes have low expected value

- `outputs/submission_ens_fulls5_fulls17_s23_w765.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `7:6:5`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `0.94%`, azimuth `1.74%`, joint `2.68%`

- `outputs/submission_ens_fulls5_fulls17_s23_w976.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `9:7:6`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `0.66%`, azimuth `1.18%`, joint `1.84%`

- `outputs/submission_ens_fulls5_fulls17_s23_w875.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `8:7:5`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `0.94%`, azimuth `2.22%`, joint `3.16%`

- `outputs/submission_ens_fulls5_fulls17_s23_w322_quota_joint.csv`
  - Current-best weighted baseline logits average, then joint quota assignment
  - Members and weights: full_seed5:full_seed17:seed23 = `3:2:2`
  - Public LB: `0.70680`
  - Quota rule: exactly 25 predictions for each of 200 sound_class/azimuth joint labels
  - Rationale: train has exactly 60 rows for each joint label; test has 5,000 rows, so the matching generated distribution is likely 25 rows per joint label
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `10.42%`, azimuth `13.22%`, joint `21.92%`
  - Submitted; much worse than current best, so quota assignment is not a good path

- `outputs/submission_ens_fulls5_fulls17_s23_w322_quota_ind.csv`
  - Current-best weighted baseline logits average, then independent class and azimuth quota assignment
  - Members and weights: full_seed5:full_seed17:seed23 = `3:2:2`
  - Output distribution: each class has 625 predictions, each azimuth has 200 predictions, joint counts range from 11 to 40
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: joint `19.42%`
  - Deprioritized because joint quota scored only `0.70680`

- `checkpoints/full_seed29_circsoft8_ls003.pt`
  - Full-train baseline checkpoint, seed `29`
  - Trained with circular azimuth soft targets (`sigma=8`) and class label smoothing (`0.03`)
  - Epochs: 30, LR: `1e-3`, batch size: `64`
  - Final train joint F1: `0.81698`
  - Created as new model diversity after weight-only and quota variants failed

- `outputs/submission_ens_fulls5_fulls17_circ29_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circular_soft_seed29 = `3:2:2`
  - Public LB: `0.72029`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w322: class `5.64%`, azimuth `9.44%`, joint `14.42%`
  - Submitted; previous best before angle-head seed37 improved to `0.72190`

- `outputs/submission_ens_fulls5_circ29_angle37_w322.csv`
  - Weighted logits average, no TTA
  - Members and weights: full_seed5:circular_soft_seed29:angle_seed37 = `3:2:2`
  - Public LB: `0.72190`
  - `94.117152` MMAC
  - Submission format validated
  - Difference vs previous best w322: class `6.08%`, azimuth `13.02%`, joint `17.96%`
  - Submitted; previous best before GCC-PHAT (`0.72231`)

- `outputs/submission_ens_fulls5_circ29_angle37_w533.csv`
  - Weighted logits average, no TTA
  - Members and weights: full_seed5:circular_soft_seed29:angle_seed37 = `5:3:3`
  - `94.117152` MMAC
  - Submission format validated
  - Difference vs previous best w322: class `5.94%`, azimuth `13.00%`, joint `17.86%`
  - Recommended first next-slot candidate after angle37 w322 scored `0.72190`

- `checkpoints/full_seed41_angle_soft8_ls003_aw025.pt`
  - Full-train `baseline_angle` checkpoint, seed `41`
  - Trained with circular azimuth soft targets (`sigma=8`), class label smoothing (`0.03`), and auxiliary angle loss weight `0.25`
  - Epochs: 30, LR: `1e-3`, batch size: `64`
  - Final train joint F1: `0.8077`
  - Created for aggressive 0.77 attack diversity

- `outputs/submission_ens_circ29_angle37_angle41_w233.csv`
  - Weighted logits average, no TTA
  - Members and weights: circular_soft_seed29:angle_seed37:angle_seed41 = `2:3:3`
  - `94.117152` MMAC
  - Submission format validated
  - Difference vs current best: class `9.32%`, azimuth `36.52%`, joint `42.00%`
  - Public LB: `0.71521`; submitted and worse than current best

- `outputs/submission_ens_circ29_angle37_angle41_eq.csv`
  - Logits average, no TTA
  - Members: circular_soft_seed29 + angle_seed37 + angle_seed41
  - `94.117152` MMAC
  - Submission format validated
  - Difference vs current best: class `9.00%`, azimuth `36.24%`, joint `41.70%`
  - Backup 0.77 attack candidate

- `make_decode_sweep.py`
  - New inference script for azimuth decoding logic
  - Supports `smooth`, `local_mean`, and `circular_mean` decoding
  - Also supports `--angle-score-weight` to add trained angle-head cosine scores to azimuth logits
  - Created because high-shift model swaps failed and the current best likely needs better azimuth decoding rather than more model churn

- `outputs/submission_ens_fulls5_circ29_angle37_w322_azsmooth8.csv`
  - Same model trio and weights as current best, but azimuth logits are probability-smoothed with sigma `8` before decoding
  - `94.117152` MMAC
  - Submission format validated
  - Difference vs current best: class `0.00%`, azimuth `16.76%`, joint `16.76%`
  - Public LB: `0.72102`; submitted and worse than current best

- `outputs/submission_ens_fulls5_circ29_angle37_w322_anglescore050.csv`
  - Same model trio and weights as current best, plus `0.50 * angle_head_cosine_score` added to azimuth logits
  - `94.117152` MMAC
  - Submission format validated
  - Difference vs current best: class `0.00%`, azimuth `0.38%`, joint `0.38%`
  - Recommended next candidate

- `checkpoints/full_seed31_circsoft8_ls003.pt`
  - Full-train baseline checkpoint, seed `31`
  - Trained with circular azimuth soft targets (`sigma=8`) and class label smoothing (`0.03`)
  - Epochs: 30, LR: `1e-3`, batch size: `64`
  - Final train joint F1: `0.80534`
  - Created as additional circular-soft diversity after seed29 improved public LB

- `checkpoints/full_circ29_31_soup_a07.pt`
  - Weight soup checkpoint: `0.7*full_seed29_circsoft8_ls003 + 0.3*full_seed31_circsoft8_ls003`
  - Created by `tools/make_weight_soup.py`
  - Keeps stronger seed29 signal while adding seed31 diversity under the 3-model MMAC limit

- `outputs/submission_ens_fulls5_fulls17_circ2931soup_a07_w533.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circ29_31_soup = `5:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `5.78%`, azimuth `4.70%`, joint `10.02%`
  - Safer non-angle backup after angle37 w322 scored `0.72190`

- `outputs/submission_ens_fulls5_fulls17_circ2931soup_a07_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circ29_31_soup = `3:2:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `5.76%`, azimuth `4.16%`, joint `9.50%`
  - Backup soup candidate

- `outputs/submission_ens_fulls5_fulls17_circ31_w322.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circular_soft_seed31 = `3:2:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `5.72%`, azimuth `3.92%`, joint `9.36%`
  - Backup circular-soft candidate

- `outputs/submission_ens_fulls5_fulls17_circ31_w533.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circular_soft_seed31 = `5:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `5.68%`, azimuth `4.00%`, joint `9.40%`
  - Backup circular-soft candidate

- `outputs/submission_ens_fulls5_fulls17_circ29_w533.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circular_soft_seed29 = `5:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `0.62%`, azimuth `1.02%`, joint `1.62%`
  - Lower-upside backup because it is very close to the current best

- `outputs/submission_ens_fulls5_fulls17_circ29_w433.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:circular_soft_seed29 = `4:3:3`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs current best w322: class `0.56%`, azimuth `1.38%`, joint `1.90%`
  - Lower-upside backup because it is very close to the current best

- `outputs/submission_full_seed29_circsoft8_ls003_tta.csv`
  - Single circular-soft seed29 with mirror TTA
  - `62.744128` MMAC
  - Submission format validated
  - Difference vs w322: class `14.60%`, azimuth `41.96%`, joint `50.16%`
  - Very risky; do not submit before the 3-model circular-soft ensemble

- `outputs/submission_ens_fulls5_fulls17_s23_w532.csv`
  - Weighted baseline logits average, no TTA
  - Members and weights: full_seed5:full_seed17:seed23 = `5:3:2`
  - `94.116192` MMAC
  - Submission format validated
  - Difference vs w321: class `0.58%`, azimuth `1.26%`, joint `1.82%`
  - Very close probe around w321

- `outputs/submission_ens_fulls5_fulls17_s23_w642.csv`
  - Same normalized weights as w321 (`6:4:2` = `3:2:1`)
  - Submission format validated, but predictions are identical to w321
  - Do not submit

## Rejected Experiments

- `v2_seed42_bg_w15`: background noise floor + azimuth weight 1.5; validation F1 `0.6810`
- `v2_seed42_w05`: original augmentation + azimuth weight 0.5; validation F1 `0.5921` during early stopping

## Code State

- `src/model.py` has `baseline`, `v2`, and `v3` architectures.
- `train.py` accepts `--arch`, `--azimuth-loss-weight`, and `--background-noise-prob`.
- `train.py` also accepts `--class-label-smoothing` and `--azimuth-soft-sigma`.
- `train.py` now uses circular angular distance for azimuth soft targets.
- `train_full.py` trains on all 12,000 training rows without a validation split for final submission checkpoints.
- Background noise floor defaults to disabled (`0.0`) so existing experiments remain reproducible.
- `infer.py` reads the architecture from each checkpoint config and supports mirror TTA.
- Existing baseline checkpoints remain compatible.

## Rules

- Train data only; scratch training; automated inference only.
- No external pretrained models or manual test labeling/listening.
- Stay below 100 MMAC.
- Two-model + mirror TTA is invalid (`125.488256` MMAC).

## Validation Workflow

Use repeated K-fold validation before spending daily submission slots:

```powershell
.\.conda\envs\chosun-audio\python.exe cross_validate.py --arch baseline --folds 5 --repeats 2 --epochs 30 --lr 1e-3 --batch-size 32 --num-workers 2 --out-dir cv_runs\baseline_5x2
```

This writes `summary.json`, `fold_summary.csv`, `oof_predictions.csv`, `error_by_class.csv`, `error_by_azimuth.csv`, and `error_by_class_azimuth.csv`.

For staged runs, add `--start-split N --max-splits 1`. Reports are written after each completed split, so a long CV run can be split across sessions.

First staged baseline split completed:

- Run dir: `cv_runs/baseline_5x2`
- Split: repeat 1, fold 1
- Epochs: 20
- LR: `3e-4`
- Best valid/OOF joint F1: `0.64836`
- Weakest classes in `error_by_class.csv`: `footsteps`, `alarm`, `speech`
- Note: known good baseline seeds used `epochs=30` and `lr=1e-3`, so rerun staged CV with those settings before making a submission decision.

## Next Action

The circular-regression candidate
`outputs/gcc_circular/submission_gcc_tree_w020_circularrf2_w025.csv` scored
`0.72295`, below the confirmed best by `0.00318`; it is submitted and rejected.
The confirmed public best remains
`outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv` at `0.72613`.
No evidence-backed next file is ready. Develop a genuinely new model or feature
axis and do not resubmit circular regression or nearby tree thresholds.

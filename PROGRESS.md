# Chosun AI Challenge Progress

Last updated: 2026-09-15

For the immediate answer to "what should I do now?", read `NEXT_ACTION.md` first.

## Current Best

- Best Public LB so far: `0.72613`
- Best submission file: `outputs/submission_ens_gcc5_circ29_angle37_w322.csv`
- Method: weighted logits average, GCC-seed5:circular_soft_seed29:angle_seed37 = `3:2:2`, no TTA
- MMAC: `94.151136`, under the 100 MMAC limit

## Tried Submissions

1. Kaggle provided baseline
   - Public LB: `0.60704`

2. `outputs/submission_seed17_tta.csv`
   - Public LB: `0.68754`
   - Validation joint F1: `0.7072`
   - Rule status: OK

3. `outputs/submission_ensemble_seed17_seed42.csv`
   - Public LB: `0.68523`
   - Method: seed17 + seed42 logits average, no TTA
   - MMAC: `62.744128`
   - Rule status: OK, but worse than seed17 TTA

4. `outputs/submission_seed42_tta.csv`
   - Public LB: `0.68885`
   - Method: seed42 single model + mirror TTA
   - Validation joint F1: `0.7213`
   - MMAC: `62.744128`
   - Rule status: OK

5. `outputs/submission_seed23_tta.csv`
   - Public LB: `0.68962`
   - Method: seed23 single model + mirror TTA
   - Validation joint F1: `0.7210`
   - MMAC: `62.744128`
   - Rule status: OK

6. `outputs/submission_v2_seed42.csv`
   - Public LB: `0.66241`
   - Method: v2 wider CNN + squeeze-excitation, no TTA
   - Validation joint F1: `0.7231`
   - MMAC: `68.893248`
   - Rule status: OK, but public score was well below validation and baseline TTA

7. `v2_seed42_bg_w15` experiment
   - Added relative background-noise-floor augmentation
   - Used `azimuth_loss_weight=1.5`
   - Best validation joint F1: `0.6810`
   - Rejected because it was below the original v2 (`0.7231`)

8. `v2_seed42_w05` experiment
   - Kept the original augmentation and used `azimuth_loss_weight=0.5`
   - Best validation joint F1: `0.5921` during early stopping
   - Rejected; `weight=1.0` remains the best v2 setting

9. `outputs/submission_v3_seed42_tta.csv`
   - Public LB: `0.65812`
   - Method: compact v2-style CNN + squeeze-excitation + dual pooling + mirror TTA
   - Validation joint F1: `0.6942`
   - MMAC: `79.539264`
   - Rule status: OK, but public score was below baseline TTA

10. `outputs/submission_v2_seed7_soft8_ls003.csv`
   - Public LB: `0.68034`
   - Method: v2 + azimuth soft targets (`sigma=8`) + class label smoothing (`0.03`), no TTA
   - Validation joint F1: `0.7524`
   - MMAC: `68.893248`
   - Rule status: OK
   - Caution: v2 validation did not transfer well to public LB; below `submission_seed23_tta.csv`

11. `outputs/submission_ensemble_seed17_seed42_seed23.csv`
   - Public LB: `0.70407`
   - Method: seed17 + seed42 + seed23 baseline logits average, no TTA
   - MMAC: `94.116192`
   - Rule status: OK
   - Previous best; baseline seed diversity transfers better than v2/v3 local gains

12. `outputs/submission_ens_s5_s17_s23_w211.csv`
   - Public LB: `0.70485`
   - Method: weighted baseline logits average, seed5:seed17:seed23 = `2:1:1`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK
   - Previous best

13. `outputs/submission_ens_fulls5_fulls17_s23_w311.csv`
   - Public LB: `0.71060`
   - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:1:1`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Previous best before w321

14. `outputs/submission_ens_fulls5_fulls17_s23_w321.csv`
   - Public LB: `0.71398`
   - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:2:1`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Previous best before w322

15. `outputs/submission_ens_fulls5_fulls17_s23_w331.csv`
   - Public LB: `0.71312`
   - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:3:1`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Slightly below w321, so increasing full_seed17 beyond the w321 ratio appears worse

16. `outputs/submission_ens_fulls5_fulls17_s23_w322.csv`
   - Public LB: `0.71707`
   - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:2:2`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Previous best before circular-soft ensemble

17. `checkpoints/full_seed23.pt`
   - Full-train baseline checkpoint, seed `23`
   - Epochs: 30, LR: `1e-3`
   - Final train joint F1: `0.7951`
   - Created because current best benefits from seed23 weight but previously used validation-split `seed23.pt`

18. `outputs/submission_ens_fulls5_fulls17_fulls23_w322.csv`
   - Public LB: `0.71570`
   - Method: weighted baseline logits average, full_seed5:full_seed17:full_seed23 = `3:2:2`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs current best w322 with non-full seed23: joint `15.64%`
   - Submitted; below current best, so full_seed23 replacement is lower priority

19. `outputs/submission_ens_fulls5_fulls17_fulls23_w433.csv`
   - Method: weighted baseline logits average, full_seed5:full_seed17:full_seed23 = `4:3:3`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs current best w322 with non-full seed23: joint `16.16%`

20. `outputs/submission_ens_fulls5_fulls17_fulls23_w423.csv`
   - Method: weighted baseline logits average, full_seed5:full_seed17:full_seed23 = `4:2:3`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs current best w322 with non-full seed23: joint `17.58%`

17. `outputs/submission_ens_fulls5_fulls17_s23_w323.csv`
   - Public LB: `0.71332`
   - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:2:3`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Below w322, so seed23 above the `3:2:2` ratio appears worse

18. `checkpoints/full_seed42.pt`
   - Full-train baseline checkpoint, seed `42`
   - Epochs: 30, LR: `1e-3`
   - Final train joint F1: `0.7938`
   - Created after daily submission slots were exhausted; use only for next-day candidates

19. `outputs/submission_ens_fulls5_s23_fulls42_w332.csv`
   - Public LB: `0.71439`
   - Method: weighted baseline logits average, full_seed5:seed23:full_seed42 = `3:3:2`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs current best w322: class `5.62%`, azimuth `12.10%`, joint `16.82%`
   - Submitted; below current best, so move to the next full_seed42 candidate

20. `outputs/submission_ens_fulls5_fulls17_s23_w654.csv`
   - Public LB: `0.71564`
   - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `6:5:4`, no TTA
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs current best w322: class `0.80%`, azimuth `1.62%`, joint `2.42%`
   - Submitted; below current best, suggesting nearby weight-only probes have low expected value

21. `outputs/submission_ens_fulls5_fulls17_s23_w322_quota_joint.csv`
   - Public LB: `0.70680`
   - Method: current-best weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:2:2`, then joint quota assignment
   - Quota assumption was wrong or too destructive for public LB despite uniform train distribution
   - Output distribution: 200 joint labels, exactly 25 predictions each
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs current best w322: class `10.42%`, azimuth `13.22%`, joint `21.92%`
   - Submitted; much worse than current best, so do not prioritize quota variants

22. `outputs/submission_ens_fulls5_fulls17_circ29_w322.csv`
   - Public LB: `0.72029`
   - Method: weighted baseline logits average, full_seed5:full_seed17:circular_soft_seed29 = `3:2:2`, no TTA
   - New checkpoint: `checkpoints/full_seed29_circsoft8_ls003.pt`
   - Training change: circular azimuth soft targets with `sigma=8` and class label smoothing `0.03`
   - Final train joint F1: `0.81698`
   - MMAC: `94.116192`
   - Rule status: OK; submission format validated
   - Difference vs previous best w322: class `5.64%`, azimuth `9.44%`, joint `14.42%`
   - Submitted; previous best, confirms circular-soft model diversity transfers to public LB

23. `checkpoints/full_seed31_circsoft8_ls003.pt`
   - Full-train baseline checkpoint, seed `31`
   - Training change: circular azimuth soft targets with `sigma=8` and class label smoothing `0.03`
   - Epochs: 30, LR: `1e-3`, batch size: `64`
   - Final train joint F1: `0.80534`
   - Created as additional circular-soft diversity after seed29 improved public LB

24. `checkpoints/full_circ29_31_soup_a07.pt`
   - Weight soup checkpoint: `0.7*full_seed29_circsoft8_ls003 + 0.3*full_seed31_circsoft8_ls003`
   - Created to fit seed29+seed31 diversity into a 3-model ensemble under the 100 MMAC limit

25. `outputs/submission_ens_fulls5_circ29_angle37_w322.csv`
  - Method: weighted logits average, full_seed5:circular_soft_seed29:angle_seed37 = `3:2:2`, no TTA
  - New checkpoint: `checkpoints/full_seed37_angle_soft8_ls003_aw025.pt`
  - Training change: circular azimuth soft targets with `sigma=8`, class label smoothing `0.03`, and auxiliary cos/sin angle head with loss weight `0.25`
  - Final train joint F1: `0.81157`
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Public LB: `0.72190`
  - Difference vs previous best w322: class `6.08%`, azimuth `13.02%`, joint `17.96%`
  - Submitted; previous best before GCC-PHAT, validates angle-head diversity on public LB

26. `checkpoints/full_seed41_angle_soft8_ls003_aw025.pt`
   - Full-train `baseline_angle` checkpoint, seed `41`
   - Training change: circular azimuth soft targets with `sigma=8`, class label smoothing `0.03`, and auxiliary cos/sin angle head with loss weight `0.25`
   - Epochs: 30, LR: `1e-3`, batch size: `64`
   - Final train joint F1: `0.8077`
   - Created for aggressive 0.77 attack diversity after angle37 improved public LB

27. `outputs/submission_ens_circ29_angle37_angle41_w233.csv`
   - Public LB: `0.71521`
   - Method: weighted logits average, circular_soft_seed29:angle_seed37:angle_seed41 = `2:3:3`, no TTA
   - Difference vs current best: class `9.32%`, azimuth `36.52%`, joint `42.00%`
   - Submitted; worse than current best, so large high-shift angle-pair attacks are deprioritized

28. `make_decode_sweep.py`
   - New inference logic for azimuth decoding
   - Supports `smooth`, `local_mean`, and `circular_mean` azimuth decoding instead of hard argmax
   - Now also supports `--angle-score-weight`, which adds the trained angle head's cosine score to azimuth logits
   - Created because soft azimuth targets and angle-head training should expose useful probability-neighborhood information at inference time

29. `outputs/submission_ens_fulls5_circ29_angle37_w322_azsmooth8.csv`
   - Public LB: `0.72102`
   - Same model trio and weights as current best, but azimuth probabilities were smoothed with sigma `8` before decoding
   - Difference vs current best: class `0.00%`, azimuth `16.76%`, joint `16.76%`
   - Submitted; worse than current best, so broad azimuth smoothing is deprioritized

## Earlier Prepared But Not Yet Scored

- `outputs/submission_ens_fulls5_circ29_angle37_w322_anglescore050.csv`
  - Same model trio and weights as current best, plus `0.50 * angle_head_cosine_score` added to azimuth logits
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best: class `0.00%`, azimuth `0.38%`, joint `0.38%`
  - Recommended next candidate; this uses the angle head directly at inference time

- `outputs/submission_ens_fulls5_circ29_angle37_w322_anglescore100.csv`
  - Same model trio and weights as current best, plus `1.00 * angle_head_cosine_score` added to azimuth logits
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best: class `0.00%`, azimuth `0.70%`, joint `0.70%`

- `outputs/submission_ens_fulls5_circ29_angle37_w322_azlocal3.csv`
  - Same model trio and weights as current best, but azimuth is decoded by top-3 local probability mean
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best: class `0.00%`, azimuth `22.06%`, joint `22.06%`

- `outputs/submission_ens_circ29_angle37_angle41_w233.csv`
  - Method: weighted logits average, circular_soft_seed29:angle_seed37:angle_seed41 = `2:3:3`, no TTA
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best: class `9.32%`, azimuth `36.52%`, joint `42.00%`
  - Public LB: `0.71521`; submitted and deprioritized

- `outputs/submission_ens_circ29_angle37_angle41_eq.csv`
  - Method: logits average, circular_soft_seed29:angle_seed37:angle_seed41, no TTA
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best: class `9.00%`, azimuth `36.24%`, joint `41.70%`

- `outputs/submission_ens_fulls17_circ29_angle41_w533.csv`
  - Method: weighted logits average, full_seed17:circular_soft_seed29:angle_seed41 = `5:3:3`, no TTA
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best: class `10.92%`, azimuth `30.12%`, joint `37.52%`

- `outputs/submission_ens_fulls5_circ29_angle37_w533.csv`
  - Method: weighted logits average, full_seed5:circular_soft_seed29:angle_seed37 = `5:3:3`, no TTA
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs previous best w322: class `5.94%`, azimuth `13.00%`, joint `17.86%`
  - Recommended first next-slot candidate after angle37 w322 scored `0.72190`

- `outputs/submission_ens_fulls5_fulls17_angle37_w433.csv`
  - Method: weighted logits average, full_seed5:full_seed17:angle_seed37 = `4:3:3`, no TTA
  - MMAC: `94.117152`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `6.04%`, azimuth `4.06%`, joint `9.80%`

- `outputs/submission_ens_fulls5_fulls17_circ2931soup_a07_w533.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:circ29_31_soup = `5:3:3`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `5.78%`, azimuth `4.70%`, joint `10.02%`
  - Safer non-angle backup after angle37 w322 scored `0.72190`

- `outputs/submission_ens_fulls5_fulls17_circ2931soup_a07_w322.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:circ29_31_soup = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `5.76%`, azimuth `4.16%`, joint `9.50%`
  - Recommended backup after the soup w533 candidate

- `outputs/submission_ens_fulls5_fulls17_circ31_w322.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:circular_soft_seed31 = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `5.72%`, azimuth `3.92%`, joint `9.36%`

- `outputs/submission_ens_fulls5_fulls17_circ31_w533.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:circular_soft_seed31 = `5:3:3`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `5.68%`, azimuth `4.00%`, joint `9.40%`

- `outputs/submission_ens_fulls5_circ29_circ31_w322.csv`
  - Method: weighted baseline logits average, full_seed5:circular_soft_seed29:circular_soft_seed31 = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `6.16%`, azimuth `13.38%`, joint `18.50%`
  - Higher-risk candidate; larger azimuth movement

- `outputs/submission_ens_fulls17_circ29_circ31_w322.csv`
  - Method: weighted baseline logits average, full_seed17:circular_soft_seed29:circular_soft_seed31 = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `7.88%`, azimuth `20.16%`, joint `26.16%`
  - Higher-risk candidate; do not submit before safer soup/circ31 candidates

- `outputs/submission_ens_fulls5_fulls17_circ29_w533.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:circular_soft_seed29 = `5:3:3`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `0.62%`, azimuth `1.02%`, joint `1.62%`
  - Lower-upside backup because it is very close to the current best

- `outputs/submission_ens_fulls5_fulls17_circ29_w433.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:circular_soft_seed29 = `4:3:3`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `0.56%`, azimuth `1.38%`, joint `1.90%`
  - Lower-upside backup because it is very close to the current best

- `outputs/submission_full_seed29_circsoft8_ls003_tta.csv`
  - Method: single circular-soft seed29 with mirror TTA
  - MMAC: `62.744128`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `14.60%`, azimuth `41.96%`, joint `50.16%`
  - Very risky; not recommended before the 3-model circular-soft ensemble

- `outputs/submission_ens_fulls5_fulls17_s23_w322_quota_ind.csv`
  - Method: current-best weighted baseline logits average, full_seed5:full_seed17:seed23 = `3:2:2`, then independent class and azimuth quota assignment
  - Output distribution: every class has 625 predictions, every azimuth has 200 predictions, joint counts range from 11 to 40
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: joint `19.42%`
  - Deprioritized because joint quota scored only `0.70680`

- `outputs/submission_ens_fulls5_fulls17_s23_w765.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `7:6:5`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `0.94%`, azimuth `1.74%`, joint `2.68%`

- `outputs/submission_ens_fulls5_fulls17_s23_w976.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `9:7:6`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `0.66%`, azimuth `1.18%`, joint `1.84%`

- `outputs/submission_ens_fulls5_fulls17_s23_w875.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:seed23 = `8:7:5`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `0.94%`, azimuth `2.22%`, joint `3.16%`

- `outputs/submission_ens_fulls5_fulls17_fulls42_w322.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:full_seed42 = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `6.12%`, azimuth `11.94%`, joint `17.16%`
  - Deprioritized after `outputs/submission_ens_fulls5_s23_fulls42_w332.csv` scored only `0.71439`

- `outputs/submission_ens_fulls5_fulls17_fulls42_w433.csv`
  - Method: weighted baseline logits average, full_seed5:full_seed17:full_seed42 = `4:3:3`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `6.28%`, azimuth `12.36%`, joint `17.66%`

- `outputs/submission_ens_fulls5_s23_fulls42_w322.csv`
  - Method: weighted baseline logits average, full_seed5:seed23:full_seed42 = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `6.06%`, azimuth `12.42%`, joint `17.66%`

- `outputs/submission_ens_fulls17_s23_fulls42_w322.csv`
  - Method: weighted baseline logits average, full_seed17:seed23:full_seed42 = `3:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Difference vs current best w322: class `7.58%`, azimuth `16.62%`, joint `22.94%`

- `outputs/submission_ens_s5_s23_s42_eq.csv`
  - Method: seed5 + seed23 + seed42 baseline logits average, no TTA
  - seed5 validation joint F1: `0.7293`
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated

- `outputs/submission_ens_s5_s17_s23_eq.csv`
  - Method: seed5 + seed17 + seed23 baseline logits average, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated

- `outputs/submission_ens_s5_s17_s42_eq.csv`
  - Method: seed5 + seed17 + seed42 baseline logits average, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated

- `outputs/submission_ens_s5_s23_s42_w122.csv`
  - Method: weighted baseline logits average, seed5:seed23:seed42 = `1:2:2`, no TTA
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated

- `outputs/submission_ens_s17_s23_s42_w122.csv`
  - Method: weighted baseline logits average, seed17:seed23:seed42 = `1:2:2`, no TTA
  - Public LB: `0.69537`
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Submitted; worse than the equal seed17+seed23+seed42 ensemble

- `outputs/submission_ens_s11_s23_s42_eq.csv`
  - Method: seed11 + seed23 + seed42 baseline logits average, no TTA
  - seed11 validation joint F1: `0.7181`
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated

- `outputs/submission_ens_s5_s23_s42_w211.csv`
  - Method: weighted baseline logits average, seed5:seed23:seed42 = `2:1:1`, no TTA
  - Motivation: seed5 has the best local validation among baseline seeds (`0.7293`)
  - Public LB: `0.70132`
  - Difference vs `submission_ensemble_seed17_seed42_seed23.csv`: class `9.98%`, azimuth `18.84%`, joint `27.06%`
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Submitted; close to `0.70407` but below it

- `outputs/submission_ens_s5_s17_s23_w211.csv`
  - Method: weighted baseline logits average, seed5:seed17:seed23 = `2:1:1`, no TTA
  - Public LB: `0.70485`
  - Difference vs `submission_ensemble_seed17_seed42_seed23.csv`: class `9.84%`, azimuth `18.76%`, joint `26.80%`
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated
  - Submitted; previous best before full-train ensemble

- `outputs/submission_ens_s5_s17_s42_w211.csv`
  - Method: weighted baseline logits average, seed5:seed17:seed42 = `2:1:1`, no TTA
  - Difference vs `submission_ensemble_seed17_seed42_seed23.csv`: class `9.56%`, azimuth `18.74%`, joint `26.60%`
  - MMAC: `94.116192`
  - Rule status: OK; submission format validated

- `outputs/submission_seed42_tta.csv`
  - Method: seed42 single model + mirror TTA
  - Validation joint F1: `0.7213`
  - MMAC: `62.744128`
  - Rule status: OK

- `outputs/submission_seed23_tta.csv`
  - Method: seed23 single model + mirror TTA
  - Validation joint F1: `0.7210`
  - MMAC: `62.744128`
  - Rule status: OK

- `outputs/submission_ensemble_seed17_seed42_seed23.csv`
  - Method: seed17 + seed42 + seed23 logits average, no TTA
  - Public LB: `0.70407`
  - MMAC: `94.116192`
  - Rule status: OK
  - Submitted; previous baseline ensemble best before seed5/full-train variants

- `outputs/submission_v2_seed42.csv`
  - Method: v2 wider CNN + squeeze-excitation, no TTA
  - Validation joint F1: `0.7231`
  - Public LB: `0.66241`
  - MMAC: `68.893248`
  - Rule status: OK, but not recommended over `submission_seed23_tta.csv`

- `outputs/submission_v2_seed7_soft8_ls003.csv`
  - Method: v2 + azimuth soft targets (`sigma=8`) + class label smoothing (`0.03`), no TTA
  - Validation joint F1: `0.7524`
  - Public LB: `0.68034`
  - MMAC: `68.893248`
  - Rule status: OK
  - Submission format validated
  - Submitted; not recommended over `submission_seed23_tta.csv`

- `checkpoints/v2_seed42_bg_w15.pt`
  - Experimental checkpoint with background noise-floor augmentation and azimuth loss weight `1.5`
  - Validation joint F1: `0.6810`
  - Rejected; do not submit

- `checkpoints/v2_seed42_w05.pt`
  - Original augmentation with azimuth loss weight `0.5`
  - Validation joint F1: `0.5921` during early stopping
  - Rejected; do not submit

- `outputs/submission_v3_seed42_tta.csv`
  - Method: compact v2-style CNN + squeeze-excitation + dual pooling + mirror TTA
  - Validation joint F1: `0.6942`
  - Public LB: `0.65812`
  - MMAC: `79.539264`
  - Rule status: OK
  - Submitted; not recommended over `submission_seed23_tta.csv`

## Data And Environment

- Competition slug: `2026-csu-sw-ai-challenge`
- Data directory: `data/`
- Train rows: 12,000
- Test rows: 5,000
- Local conda env: `.conda/envs/chosun-audio`
- CUDA verified on RTX 5080
- Kaggle API works

## Model Setup

- Input feature shape: `(5, 64, 201)`
- Feature channels:
  - left log-mel
  - right log-mel
  - ILD
  - cos(IPD)
  - sin(IPD)
- Models:
  - baseline: small depthwise separable CNN
  - v2: wider depthwise CNN with Squeeze-Excitation and average/max pooling
  - v3: compact v2-style model sized for mirror TTA
- Heads:
  - `sound_class`: 8-class classification
  - `azimuth`: 25-class classification
- Loss:
  - CrossEntropy(sound_class) + CrossEntropy(azimuth)
  - Optional: class label smoothing and azimuth soft-label CE based on angular distance
- No external pretrained model.
- Trained from scratch only on provided training data.

## Important Rule Notes

- OK:
  - train data only
  - scratch training
  - automated test inference
  - mirror TTA when using one model
  - two-model ensemble without TTA

- Do not submit:
  - `2 models + mirror TTA`, because MMAC is `125.488256`
  - any external pretrained model
  - any manually inspected/listened/labeled test audio result

## Useful Commands

Train a new seed:

```powershell
.\.conda\envs\chosun-audio\python.exe train.py --epochs 30 --seed 23 --lr 1e-3 --batch-size 32 --num-workers 2 --out checkpoints\seed23.pt
```

Train a final seed on all training rows:

```powershell
.\.conda\envs\chosun-audio\python.exe train_full.py --arch baseline --epochs 30 --lr 1e-3 --seed 5 --batch-size 32 --num-workers 2 --out checkpoints\full_seed5.pt
```

Single model + mirror TTA inference:

```powershell
.\.conda\envs\chosun-audio\python.exe infer.py --checkpoint checkpoints\seed42.pt --tta mirror --batch-size 128 --num-workers 2 --out outputs\submission_seed42_tta.csv
```

Validate a submission:

```powershell
.\.conda\envs\chosun-audio\python.exe validate_submission.py --submission outputs\submission_seed42_tta.csv
```

Check MMAC:

```powershell
.\.conda\envs\chosun-audio\python.exe calculate_mmac.py --models 1 --tta-views 2
```

Run repeated K-fold validation and OOF error analysis:

```powershell
.\.conda\envs\chosun-audio\python.exe cross_validate.py --arch baseline --folds 5 --repeats 2 --epochs 30 --lr 1e-3 --batch-size 32 --num-workers 2 --out-dir cv_runs\baseline_5x2
```

For staged runs, add `--start-split N --max-splits 1`. Reports are written after each completed split.

## Next Steps

### 2026-09-15 GCC-PHAT dual-path model

- Added `gcc`, a new architecture that fuses the existing mel CNN with an
  explicit 64-lag GCC-PHAT localization head.
- Controlled seed-5 validation: `0.73378` joint macro-F1 versus `0.72931` for
  the unchanged baseline; azimuth accuracy rose from `0.64042` to `0.68500`.
- GCC scale sweep peaked at multiplier `1.0` (`0.733779`); the gain was stable
  from `0.8` to `1.4`.
- Full-data checkpoint: `checkpoints/full_gcc_seed5.pt`.
- New submission candidate:
  `outputs/submission_ens_gcc5_circ29_angle37_w322.csv`.
- The candidate keeps every class prediction from the current public best and
  changes `16.70%` of azimuth predictions; mean absolute change is `1.43` degrees.
- Format validation passed; three-forward compute is `94.151136` MMAC.
- Public leaderboard score: `0.72231`; this improves the previous best `0.72190`
  by `0.00041` and validates GCC-PHAT, but remains far below `0.77`.

### 2026-09-15 class-recognition follow-up

- Added and validated two time-aware class-head variants after GCC-PHAT.
- `gcc_temporal`: learned attention plus mean/max temporal pooling; best validation
  joint F1 `0.73161`, below GCC-only `0.73378`. Rejected.
- `gcc_class`: independent mono-spectrogram CNN branch; best validation class
  accuracy `0.77333`, below baseline `0.77708`. Rejected.
- Class-only fine-tuning of the baseline reached `0.77958`, only `+0.00250` and
  without enough joint/public-LB evidence to justify a submission.
- Handcrafted time-frequency classifiers were also rejected: ExtraTrees class
  accuracy `0.62917`, best RBF-SVM accuracy `0.65833`.
- No new submission was produced from these rejected experiments. Keep the
  confirmed public best `submission_ens_gcc5_circ29_angle37_w322.csv` (`0.72231`).

### GCC angle-score 0.50 result and prepared next submission

- Submitted file: `outputs/submission_ens_gcc5_circ29_angle37_w322_anglescore050.csv`.
- Starts from the confirmed GCC best and adds angle-head cosine score weight `0.50`.
- Difference versus current best: class `0.00%`, azimuth/joint `0.26%` (13 rows),
  mean absolute azimuth change `0.122` degrees across all test rows.
- Submission format validated; compute remains under 100 MMAC.
- Public leaderboard score: `0.72231`, tied with the current best and did not
  improve the displayed score.
- At that time the next file was
  `outputs/submission_ens_fulls5_circ29_angle37_w322_anglescore050.csv`; this is
  now superseded by the aggressive GCC-statistics candidate below.

### 2026-09-15 aggressive GCC-statistics class attack

- Added `gcc_stats`: temporal mean/std/max/attention class pooling, a richer
  azimuth head, GCC-PHAT fusion, and an auxiliary unit-circle angle head.
- Controlled seed-5 fusion validation reached joint F1 `0.75656`, compared with
  GCC-only `0.73378` on the same split (`+0.02278`).
- Full-data checkpoint: `checkpoints/full_gcc_stats_seed47.pt`.
- Final full-train metrics: class accuracy `0.8759`, azimuth accuracy `0.6404`,
  joint F1 `0.8433`.
- Primary aggressive candidate:
  `outputs/submission_aggressive_gccstats47_gcc5_circ29_class521_az032.csv`.
- Three inference members: gcc_stats_seed47 + GCC_seed5 + circular_soft_seed29.
- Task-specific standardized-logit weights: class `5:2:1`, azimuth `0:3:2`.
- Difference from current best: class `13.82%`, azimuth `4.36%`, joint `17.34%`.
- Exact compute: `31.722592 + 31.383712 + 31.372064 = 94.478368` MMAC.
- Submission format validated with 5,000 rows.
- Public leaderboard score: `0.71852`, below the confirmed best `0.72231`.
- The single-split fusion gain did not transfer to the public leaderboard; do
  not resubmit this file or treat its tuned class weights as robust evidence.

### 2026-09-16 disentangled two-split candidate

- Added the `disentangled` architecture with independent acoustic-class and
  spatial/GCC localization trunks.
- Fixed hyperparameters reproduced across two stratified validation splits:
  seed 5 joint F1 `0.805083`, seed 17 joint F1 `0.788844`, mean `0.796964`.
- Mirror TTA was rejected because it sharply reduced exact azimuth accuracy.
- Uniform class/azimuth quotas were rejected because they did not improve both
  validation splits.
- Angle score weight `0.25` improved both splits slightly (`0.805656` and
  `0.789395`) and was retained.
- Full checkpoints: `checkpoints/full_disentangled_seed5.pt` and
  `checkpoints/full_disentangled_seed17.pt`.
- Next submission: `outputs/submission_disentangled_s5_s17_eq_angle025.csv`.
- Exact compute: `95.677888` MMAC. Format validated with 5,000 rows, no duplicate
  IDs or nulls.
- SHA-256: `89DC49EB3806E2A46F07339D143235EA2A0A45901668F7253EAE29E1AC138F5C`.
- Public leaderboard score: `0.71911`, below the confirmed best `0.72231` by
  `0.00320`.
- The large gap from two-split random validation (`0.796964` mean) shows that
  row-wise stratification is not a reliable proxy for the hidden test set.
- Do not submit the raw no-angle variant; it changes only `0.88%` of azimuth
  predictions relative to the scored file.

### 2026-09-16 GCC/ILD tree correction

- Trained a 600-tree ExtraTrees azimuth classifier on 64-bin GCC-PHAT plus
  global log-energy-ratio features.
- Whole-azimuth accuracy alone was not used for selection; replacing the neural
  azimuth outright reduced joint F1 and was rejected.
- Low-weight logit fusion improved joint F1 on two independent splits. At tree
  weight `0.15`: seed 5 `0.733779 -> 0.735134`; seed 17
  `0.707239 -> 0.709709`.
- The zero-weight test output exactly reproduces the confirmed `0.72231` best,
  verifying that the base ensemble implementation is unchanged.
- Candidate: `outputs/gcc_tree/submission_gcc_tree_w015.csv`.
- Test changes versus current best: class `0%`, azimuth/joint `5.56%`; `86.3%`
  of changed angles are within 20 degrees.
- Compute estimate: deep ensemble `94.151136` MMAC plus at most `0.0378`
  million tree comparisons per audio. Format validated.
- SHA-256: `653C7DC6B7B04448825647B87C9C98B2144C15B18E83B18A9671C9F9A12546DE`.

### 2026-09-16 GCC/ILD public result and next strength

- `outputs/gcc_tree/submission_gcc_tree_w015.csv` scored `0.72573` on the public
  leaderboard, improving the former `0.72231` best by `+0.00342`.
- Generated weights `0.20`, `0.25`, and `0.30`. Weight `0.30` remains stronger
  than `0.15` on both validation splits.
- Versus `w015`, `w030` changes no sound classes and changes azimuth on `4.54%`
  of test rows. Median circular change is 5 degrees; `88.1%` are within 20 degrees.
- Next candidate: `outputs/gcc_tree/submission_gcc_tree_w030.csv`.

### 2026-09-16 GCC/ILD stronger-weight result and conditional correction

- `outputs/gcc_tree/submission_gcc_tree_w030.csv` scored `0.72434`, down
  `0.00139` from the `0.72573` best. Global strengthening is rejected.
- Evaluated conditional `w015 -> w030` changes using tree confidence, tree margin,
  and neural margin on seed 5 and seed 17 validation splits.
- The robust rule was to apply `w030` only to changed samples in the upper half
  of ExtraTrees top-two probability margin. It improved both splits over `w015`
  by `+0.00139` and `+0.00071`.
- Test candidate changes zero sound classes and 114/5,000 azimuths (`2.28%`)
  versus the best; median movement is 5 degrees and `92.1%` are within 20 degrees.
- Next candidate:
  `outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv`.
- SHA-256: `693CCF2B6633148F0CE7AD2F60192F7F928454365024238FE9D0F4ED4A09B50E`.

### 2026-09-16 conditional GCC/ILD public result

- `outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv` scored
  `0.72613`, improving `w015` (`0.72573`) by `+0.00040`; this is the new best.
- Tested additional tree-confidence/tree-margin AND/OR gates and neural-margin
  gates. None robustly beat the submitted median tree-margin rule on both splits.
- Stop local tree-weight/threshold submissions and move to a new modeling or
  feature axis for the next material improvement.

### 2026-09-17 circular GCC/ILD regression candidate

- Tested two new axes and rejected both because they failed to beat the current
  conditional-tree baseline on both validation splits: class-conditional GCC
  trees and diversified classification forests.
- Added circular angle regression: a RandomForest predicts `(cos, sin)` from
  the 64-bin GCC-PHAT vector plus log energy ratio.
- The selected fixed blend uses global ExtraTrees weight `0.20` and circular-RF
  weight `0.25` (500 trees, `min_samples_leaf=2`).
- It beat the submitted conditional-tree baseline on both splits: seed 5
  `0.736525 -> 0.736988` and seed 17 `0.711086 -> 0.713472`.
- New candidate:
  `outputs/gcc_circular/submission_gcc_tree_w020_circularrf2_w025.csv`.
- Keeps all class predictions; changes 374/5,000 azimuths (`7.48%`) versus the
  `0.72613` best. Format validation passed.
- SHA-256: `C1758A1D67FF2E06011EE83F84A37AC356722051C40B3E93F5A5F50110C3B147`.
- Deep compute remains `94.151136` MMAC. The additional 500-tree regressor has
  maximum observed depth 55 (at most `0.0275` million comparisons), so total
  remains safely below 100 MMAC.
- Public leaderboard score: `0.72295`, below the confirmed `0.72613` best by
  `0.00318`. The two-split local gain did not transfer; reject this candidate
  and do not resubmit it.

1. Keep `outputs/submission_ens_gcc5_circ29_angle37_w322.csv` as the current best public-LB submission at `0.72231`.
2. Daily submission slots are exhausted; do not submit again until the next reset.
3. `outputs/submission_ens_fulls5_s23_fulls42_w332.csv` scored `0.71439`, below the current best.
4. `outputs/submission_ens_fulls5_fulls17_s23_w654.csv` scored `0.71564`, below the current best.
5. `outputs/submission_ens_fulls5_fulls17_s23_w322_quota_joint.csv` scored `0.70680`, so do not prioritize quota variants.
6. Do not prioritize more nearby weight-only tweaks; `w322` appears to be a local public-LB sweet spot.
7. Circular-soft checkpoint `checkpoints/full_seed29_circsoft8_ls003.pt` reached final train joint F1 `0.81698` and improved public LB.
8. New circular-soft checkpoint `checkpoints/full_seed31_circsoft8_ls003.pt` reached final train joint F1 `0.80534`.
9. `outputs/submission_ens_fulls5_circ29_angle37_w322_azsmooth8.csv` scored `0.72102`, below current best.
10. The former new-logic candidate
    `outputs/submission_ens_fulls5_circ29_angle37_w322_anglescore050.csv` is now
    behind the aggressive GCC-statistics candidates in submission order.
11. Backup new-logic candidate: `outputs/submission_ens_fulls5_circ29_angle37_w322_anglescore100.csv`.
12. `outputs/submission_ens_fulls5_fulls17_circ29_w533.csv` and `outputs/submission_ens_fulls5_fulls17_circ29_w433.csv` are now lower-upside backups because they differ from current best by only joint `1.62%` and `1.90%`.
13. Do not prioritize remaining full_seed23 replacement files; `outputs/submission_ens_fulls5_fulls17_fulls23_w322.csv` scored only `0.71570`.
14. Deprioritize remaining full_seed42 candidates.
15. `outputs/submission_ens_fulls5_fulls17_s23_w321.csv` is the previous best at `0.71398`; `outputs/submission_ens_fulls5_fulls17_s23_w311.csv` scored `0.71060`; `outputs/submission_ens_s5_s17_s23_w211.csv` was earlier best at `0.70485`.
16. Do not prioritize `outputs/submission_v2_seed42.csv`; public LB was only `0.66241`.

Do not submit `outputs/submission_ens_fulls5_fulls17_s23_w642.csv`; its normalized weights equal w321 and predictions are identical.

Do not use `2 models + mirror TTA`; it exceeds the 100 MMAC limit.

### 2026-09-17 robust GCC circular-soft/noise candidate

- Extracted simple audio-domain diagnostics and found a measurable train/test
  shift (domain-classifier AUC about `0.68`), motivating robustness training.
- Trained GCC checkpoints with circular-soft azimuth targets (`sigma=8`), class
  label smoothing `0.03`, and background-noise augmentation probability `0.20`.
- Validation checkpoints: `checkpoints/gcc_circnoise_seed5.pt` and
  `checkpoints/gcc_circnoise_seed17.pt`. Full-data checkpoint:
  `checkpoints/full_gcc_circnoise_seed17.pt`.
- Fixed standardized-logit fusion with the old GCC model uses class weights
  `1:1.8` and azimuth weights `1:1`. Joint macro-F1 was `0.768897` on seed 5
  and `0.750893` on seed 17.
- A global GCC/ILD ExtraTrees correction at weight `0.25` improved both splits
  to `0.770305` and `0.751598`; lower/higher nearby weights were less robust.
- Generated primary candidate:
  `outputs/gcc_circnoise/submission_gcc_tree_w025.csv`.
- Submission format passed: 5,000 rows in exact sample order, valid labels,
  integer azimuths, no duplicate IDs, and no nulls.
- Deep compute is `62.767424` MMAC; 600-tree worst-case traversal adds about
  `0.0378` million comparisons, remaining below 100 MMAC.
- Versus the `0.72613` best, it changes 604 classes, 464 azimuths, and 997 joint
  rows. SHA-256:
  `ACE57A17AA5ED35DAA33A46DD2C06E172046CCDC669F679E11D17EADFFBCF8DF`.
- The public score is unknown. Validation clears `0.75` on both splits, but this
  is not a guarantee because prior random-split gains have sometimes failed to
  transfer to the leaderboard.

### 2026-09-17 robust GCC public failure and conservative recovery

- `outputs/gcc_circnoise/submission_gcc_tree_w025.csv` scored `0.71119`, a drop
  of `0.01494` from the `0.72613` best. It is rejected.
- Root-cause audit: the file changed 604 sound classes and 997 joint rows under
  a measured train/test shift. In addition, seed-5 fusion validation used an old
  GCC checkpoint while seed-17 used an old baseline checkpoint, so the reported
  two-split numbers did not exactly reproduce final inference.
- Tested robust-azimuth override gates with classes fixed. A tree-agreement gate
  improved both random splits, but combining it with the existing three-model
  best would require four deep models and exceed 100 MMAC; the generated file in
  `outputs/robust_gate` is invalid and must not be submitted.
- Returned to the proven three-model GCC/ILD path. The fixed rule requiring both
  tree-margin and tree-confidence median gates improved over `w015` on both
  splits: `+0.000982` and `+0.000841`.
- New conservative candidate:
  `outputs/gcc_tree/submission_gcc_tree_marginq050_confq050_and.csv`.
- It changes zero classes and only 14 azimuths versus the `0.72613` best; format
  validation passed. Compute stays below 100 MMAC.
- SHA-256:
  `9E0F2DF46C625AB2769263E5FC2CFABC28053775DC527D62FD3C9BB78DFEAF59`.

### 2026-09-17 conservative gate public result

- `outputs/gcc_tree/submission_gcc_tree_marginq050_confq050_and.csv` scored
  `0.72538`, which is `0.00075` below the `0.72613` best and `0.00035` below
  `w015=0.72573`; reject it.
- The file preserved all classes and reverted only 14 azimuth decisions from
  the best. The decline shows those conditional corrections transferred better
  to the public set than the extra confidence filter predicted.
- Do not spend more slots on nearby GCC-tree thresholds. No new submission is
  ready; the next research axis must use domain-shift-aware validation.

### 2026-09-17 domain-shift-aware ensemble candidate

- Built `cv_runs/domain_valid_indices.npy`: the most test-like 20% within every
  class/azimuth stratum. Domain OOF AUC is `0.680006`; selected-row mean test
  propensity is `0.431477` versus `0.267767` for all training rows.
- Retrained all three public-ensemble members on the exact split and compared
  valid 3-model combinations under identical standardized-logit/tree decoding.
- Public-like domain ensemble: joint `0.724781`, class accuracy `0.761667`,
  azimuth accuracy `0.685833`.
- Robust-GCC replacement: joint `0.737091`, class accuracy `0.772500`, azimuth
  accuracy `0.725000`; improvements are `+0.012310`, `+0.010833`, `+0.039167`.
- Trained exact full checkpoint `checkpoints/full_gcc_domainrobust_seed5.pt` and
  generated `outputs/domain_robust/submission_gcc_tree_condmargin_q050_w015_w030.csv`.
- Three deep forwards cost `94.151136` MMAC; tree worst case adds about `0.0378`
  million comparisons. Format validation passed.
- Versus the `0.72613` best: 478 class, 1,509 azimuth, 1,831 joint-row changes.
- SHA-256:
  `6671B56085EB131C886DEA13C8AEFC22DD0CF4BA764C4ACC0F792E90A78A2D32`.

### 2026-09-17 domain-robust public result

- `outputs/domain_robust/submission_gcc_tree_condmargin_q050_w015_w030.csv`
  scored `0.72320`, which is `0.00293` below the `0.72613` best; reject it.
- The matched domain validation gain (`0.724781 -> 0.737091`) did not transfer
  to the public set. Selecting high test-propensity training rows captured
  simple acoustic shift but not the hidden label/error distribution.
- The confirmed best remains
  `outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv` at
  `0.72613`. No new evidence-backed submission is ready.

### 2026-09-17 acoustic-content Group validation

- Extracted shift-sensitive, gain/EQ-normalized mono log-mel fingerprints to
  `cv_runs/source_fingerprint_direct.npy`. High-confidence duplicate-like pairs
  covered too few rows for useful Group K-fold validation.
- Clustered fingerprints within each sound class into 320 acoustic-content
  groups and created five leakage-resistant folds in
  `cv_runs/acoustic_group_folds`.
- Fold 0: GCC base `0.731855`, robust GCC `0.748425` (`+0.016570`).
- Fold 1: GCC base `0.717766`, robust GCC `0.707036` (`-0.010730`).
- Two-fold means are base `0.724811`, robust `0.727731`, but the negative
  worst-fold result means the gain is not reproducible. This agrees with the
  robust ensemble's public failure at `0.72320`; reject the recipe and do not
  create another submission from it.

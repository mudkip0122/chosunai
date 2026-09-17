# What To Submit Next

Last updated: 2026-09-17

## Authoritative Answer

No new evidence-backed file is ready. Keep the confirmed best:

`outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv` = `0.72613`

Do not resubmit it if duplicate submissions waste a slot.

## Latest Public Results

- `outputs/domain_robust/submission_gcc_tree_condmargin_q050_w015_w030.csv`:
  `0.72320` — rejected (`-0.00293` versus best).
- `outputs/gcc_tree/submission_gcc_tree_marginq050_confq050_and.csv`:
  `0.72538` — rejected (`-0.00075` versus best).
- `outputs/gcc_circnoise/submission_gcc_tree_w025.csv`:
  `0.71119` — rejected (`-0.01494` versus best).
- `outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv`:
  `0.72613` — confirmed best.
- `outputs/gcc_tree/submission_gcc_tree_w015.csv`:
  `0.72573`.

## Next Research Requirement

Stop nearby tree gates and random-split-selected class changes. The next file
must come from domain-shift-aware validation or a materially new model, reproduce
its validation and inference recipe exactly, and remain below 100 MMAC.

The domain-robust candidate improved matched validation from `0.724781` to
`0.737091` but scored only `0.72320` publicly. Do not submit it again and do not
use the single domain-propensity split as the sole selection criterion.

Acoustic Group validation is now available in `cv_runs/acoustic_group_folds`.
The robust GCC recipe won fold 0 but lost fold 1, so it is rejected. A future
candidate must improve every tested group fold, not merely the mean.

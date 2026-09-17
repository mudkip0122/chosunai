# Next Action

Last updated: 2026-09-17

## Immediate Action

The domain-shift-validated candidate has now been scored and rejected:

`outputs/domain_robust/submission_gcc_tree_condmargin_q050_w015_w030.csv` = `0.72320`

The confirmed public best remains:

`outputs/gcc_tree/submission_gcc_tree_condmargin_q050_w015_w030.csv` = `0.72613`

The latest candidates are rejected:

- `outputs/domain_robust/submission_gcc_tree_condmargin_q050_w015_w030.csv` = `0.72320`
- `outputs/gcc_circnoise/submission_gcc_tree_w025.csv` = `0.71119`
- `outputs/gcc_tree/submission_gcc_tree_marginq050_confq050_and.csv` = `0.72538`

The second result is `0.00075` below the best and `0.00035` below the original
`w015=0.72573`. Its 14 reverts show that the public-best conditional corrections
were useful on the hidden set even when random validation preferred filtering
some of them out.

## Conclusions

- Built five acoustic-content Group folds from gain/EQ-normalized mono log-mel
  fingerprints. Similar-content clusters never cross train/validation within a
  fold. Exact duplicate-like groups alone covered too few rows, so 320 within-
  class acoustic clusters were used for the group split.
- Controlled GCC comparison is inconsistent: fold 0 base `0.731855` versus
  robust `0.748425` (`+0.016570`), but fold 1 base `0.717766` versus robust
  `0.707036` (`-0.010730`). Mean gain is positive but worst-fold gain is
  negative, matching the public failure. Reject the robust recipe.

- A new validation split selects the 20% most test-like training rows within
  every class/azimuth stratum (domain OOF AUC `0.6800`; mean test propensity
  `0.4315` versus `0.2678` over all train rows).
- On this exact split, replacing the original GCC member with the robust GCC
  member improves the matched 3-model conditional-tree ensemble from `0.724781`
  to `0.737091`. Class accuracy improves `0.761667 -> 0.772500`; azimuth
  accuracy improves `0.685833 -> 0.725000`.
- Final inference exactly matches the validated architecture/weights:
  robust GCC + circ29 + angle37 at `3:2:2`, conditional tree q50, no TTA.
- Compute is about `94.151136 + 0.0378` MMAC, below 100. Format validated.
- SHA-256:
  `6671B56085EB131C886DEA13C8AEFC22DD0CF4BA764C4ACC0F792E90A78A2D32`.

- Do not change the current-best sound classes: the robust-GCC candidate's 604
  class changes caused a large public drop.
- Do not make another nearby GCC-tree threshold/confidence submission. Public
  results now show `w015 < conditional margin q50`, while the extra confidence
  filter reverses the gain.
- Random row-wise validation is not reliable enough for further submission
  selection under the measured train/test domain shift.
- This candidate changes 478 classes and 1,509 azimuths versus the public best;
  it is a material model replacement supported by the new domain validation,
  but its public score was only `0.72320` (`-0.00293` versus best). The
  domain-propensity split also failed to transfer and must not be used alone.

## Invalid/Rejected

- Do not submit any already scored file above.
- Do not submit `outputs/robust_gate/submission_bestclass_robustaz_treeagree.csv`;
  reproducing it requires four deep models and exceeds 100 MMAC.
- Do not submit `outputs/gcc_circnoise/submission_gcc_tree_w000.csv`.

## When A New Score Is Reported

Update this file, `WHAT_TO_SUBMIT_NEXT.md`, `HANDOFF.md`, and `PROGRESS.md`.

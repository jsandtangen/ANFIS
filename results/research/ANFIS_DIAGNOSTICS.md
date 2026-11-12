# ANFIS failure and stability diagnostics

These are measured diagnostics of the executable model. They do not establish the mechanism of unavailable historical runs. Test analyses describe already selected models; none change the main-study selection.

## Optimization evidence

The cached/vectorized replay of original seed-42 weighted GD differed from the unchanged implementation's fitted coefficients by at most 2.776e-17. It recorded epoch 0 through 500, including weighted validation BCE and gradient norms (`original_gradient_trajectory.csv`). Training BCE changed from 0.69330282 to 0.68490366; validation BCE changed from 0.69276741 to 0.69161727. Gradient L2 changed from 0.03118502 to 0.02689078. A nonzero terminal gradient and continuing loss change indicate the fixed 500 steps did not prove stationarity. Small per-rule normalized weights attenuate coefficient gradients; this is an algebraic mechanism, not proof of a unique cause of held-out errors.

Same initialization, same 500 steps, same fixed premises and weights, changing learning rate only (validation outputs):

| lr | train_loss_last_recorded | validation_bce | validation_balanced_accuracy | validation_recall | validation_f1 |
| --- | --- | --- | --- | --- | --- |
| 0.020000 | 0.684918 | 0.691617 | 0.538690 | 0.229167 | 0.289474 |
| 0.200000 | 0.656212 | 0.690892 | 0.586310 | 0.895833 | 0.500000 |
| 2.000000 | 0.622260 | 0.690748 | 0.584821 | 0.875000 | 0.497041 |


This ablation evaluates no test thresholds and does not retrospectively replace selected configurations. Changes in fit and validation losses support an optimizer-budget diagnosis when they coincide; they do not show that higher learning rates always generalize better. Adam comparisons also change regularization and stopping, so attribute their combined effects cautiously.

## Coverage, normalization and score separation

Seed 42 weighted, held-out test diagnostics:

| tag | normalized_sum_min | rows_raw_sum_below_1e10 | mean_effective_rules | coefficient_l2 | positive_rate_default | good_score_mean | bad_score_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0.008688 | 1 | 327.160424 | 0.993494 | 0.445000 | 0.501894 | 0.512181 |
| fixed_adam | 1.000000 | 1 | 27.120686 | 1.504760 | 0.505000 | 0.498120 | 0.530189 |
| trainable_adam | 1.000000 | 1 | 25.926077 | 1.495203 | 0.485000 | 0.498269 | 0.531140 |


Changing only original normalization to a log-domain exact normalization, retaining the fitted original premises and coefficients, changed a seed-42 weighted test logit by at most 0.3199822. This explicitly tests epsilon-related attenuation; compare its scale with scores and margins before calling it a dominant performance cause. `diagnostics.json` records all split/condition minima and undercoverage counts, Gaussian centres/widths, coincident centre pairs, coefficient magnitudes and mean rule weights. Low feature coverage means max Gaussian grade below .01; saturated membership means grade below 1e-8. These operational cutoffs are diagnostics, not calibrated danger thresholds.

The new fixed and trainable models normalize in log space. This avoids product underflow and ensures unit-sum normalized weights even in tails, but relative normalization can still yield confident extrapolation where absolute membership is tiny. Thus numerical stability does not ensure adequate data support.

## Threshold and initialization sensitivity

Of 15 original-architecture default-threshold split-condition results, 2 have a predicted-positive fraction <=.05 or >=.95. Their complete matrices and recalls are in per_split_metrics.csv. Selected validation thresholds are reported separately; a threshold can change recall and F1 without changing ROC-AUC or average precision. AUC and AP assess ranking and are unchanged between the two threshold rows of any one fitted model.

With split seed fixed to 42, original weighted GD was rerun using five consequent initialization seeds:

| initialization_seed | threshold_mode | threshold | balanced_accuracy | recall | f1 | roc_auc | positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | default | 0.500000 | 0.585714 | 0.850000 | 0.495146 | 0.690119 | 0.730000 |
| 0 | validation | 0.510000 | 0.642857 | 0.450000 | 0.490909 | 0.690119 | 0.250000 |
| 1 | default | 0.500000 | 0.609524 | 0.633333 | 0.487179 | 0.665000 | 0.480000 |
| 1 | validation | 0.520000 | 0.602381 | 0.283333 | 0.386364 | 0.665000 | 0.140000 |
| 2 | default | 0.500000 | 0.632143 | 0.700000 | 0.515337 | 0.680952 | 0.515000 |
| 2 | validation | 0.500000 | 0.632143 | 0.700000 | 0.515337 | 0.680952 | 0.515000 |
| 3 | default | 0.500000 | 0.615476 | 0.816667 | 0.513089 | 0.688929 | 0.655000 |
| 3 | validation | 0.500000 | 0.615476 | 0.816667 | 0.513089 | 0.688929 | 0.655000 |
| 4 | default | 0.500000 | 0.630952 | 0.583333 | 0.500000 | 0.686190 | 0.400000 |
| 4 | validation | 0.490000 | 0.604762 | 0.966667 | 0.517857 | 0.686190 | 0.820000 |


This separates initialization variability from changes in train/test composition. Both matter, but neither set of five runs establishes a universal instability rate. The original architecture's frozen premises cannot adapt regions to label structure; improvement from trainable premises must be assessed empirically, and may be negative. New comparisons preserve every result.

## Figures and artifacts

- `plots/original_optimization_and_memberships.png`: original training/validation BCE, gradient norm, and scaled membership functions.
- `plots/diagnostics_seed42_weighted.png`: test ROC, PR, threshold sensitivity, original score histograms, tuned validation objectives and gradient trajectories. Penalized objectives and original BCE are labelled and are not directly identical losses.
- `plots/rule_concentration.png`: sorted mean held-out normalized rule weights.
- `diagnostic_summary.csv`, `diagnostics.json`: numerical diagnostics for all selected ANFIS fits.
- `histories/`: all tuning trajectories, including rejected candidates.

The original main suppresses all Python warnings on import, and that global filter also applies to the extension importing it. Numerical tests, finite checks and convergence trajectories were used, but the absence of printed warnings must not be interpreted as a convergence certificate. This should be removed in a future separately recorded run if warning-level provenance is needed.

## Isolated normalization intervention

Holding the fitted original coefficients and premises constant, replace only the firing normalization at inference. Thresholds remain .5 or the original validation-selected threshold; no threshold is selected on these test outcomes. This is a diagnostic intervention, not a retuned performance claim.

| normalization | threshold_mode | threshold | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_epsilon | default | 0.500000 | 0.646429 | 0.650000 | 0.523490 | 0.690357 | 0.471485 |
| original_epsilon | original_validation | 0.510000 | 0.635714 | 0.400000 | 0.470588 | 0.690357 | 0.471485 |
| exact_log_domain | default | 0.500000 | 0.646429 | 0.650000 | 0.523490 | 0.695357 | 0.474269 |
| exact_log_domain | original_validation | 0.510000 | 0.644048 | 0.416667 | 0.485437 | 0.695357 | 0.474269 |

Per-row original/stable logits, scores and absolute firing sums are in normalization_only_predictions.csv.

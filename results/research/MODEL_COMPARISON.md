# New matched-split model comparison

Five stratified holdouts; mean ± sample SD. Positive class is bad credit. The grids and partitions were fixed before these new runs.

## none: default thresholds

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.538095 ± 0.033077 | 0.113333 ± 0.059395 | 0.186179 ± 0.091679 | 0.599333 ± 0.053290 | 0.415089 ± 0.060609 |
| fixed_adam | 0.530952 ± 0.026527 | 0.093333 ± 0.050827 | 0.157447 ± 0.078953 | 0.647476 ± 0.040371 | 0.449626 ± 0.083711 |
| trainable_adam | 0.545238 ± 0.027932 | 0.133333 ± 0.061237 | 0.211906 ± 0.087800 | 0.653905 ± 0.045023 | 0.462045 ± 0.076696 |
| mlp | 0.520238 ± 0.027523 | 0.093333 ± 0.069322 | 0.145092 ± 0.105490 | 0.657440 ± 0.055076 | 0.442856 ± 0.074025 |
| logistic | 0.551429 ± 0.028013 | 0.140000 ± 0.056026 | 0.225318 ± 0.079713 | 0.679214 ± 0.019629 | 0.477622 ± 0.042360 |
| dummy | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |

## none: validation thresholds

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.540714 ± 0.042769 | 0.410000 ± 0.398365 | 0.301246 ± 0.192921 | 0.599333 ± 0.053290 | 0.415089 ± 0.060609 |
| fixed_adam | 0.561905 ± 0.041836 | 0.556667 ± 0.337598 | 0.391358 ± 0.154884 | 0.647476 ± 0.040371 | 0.449626 ± 0.083711 |
| trainable_adam | 0.613333 ± 0.049711 | 0.626667 ± 0.301984 | 0.459281 ± 0.138446 | 0.653905 ± 0.045023 | 0.462045 ± 0.076696 |
| mlp | 0.619524 ± 0.059181 | 0.633333 ± 0.116070 | 0.495399 ± 0.068138 | 0.657440 ± 0.055076 | 0.442856 ± 0.074025 |
| logistic | 0.595714 ± 0.031096 | 0.590000 ± 0.298515 | 0.446252 ± 0.094911 | 0.679214 ± 0.019629 | 0.477622 ± 0.042360 |
| dummy | 0.500000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |

## weighted: default thresholds

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.618571 ± 0.051243 | 0.700000 ± 0.081650 | 0.504647 ± 0.044243 | 0.656857 ± 0.053678 | 0.462107 ± 0.053358 |
| fixed_adam | 0.627857 ± 0.031995 | 0.680000 ± 0.061689 | 0.509220 ± 0.033107 | 0.687405 ± 0.025717 | 0.489580 ± 0.057202 |
| trainable_adam | 0.623095 ± 0.036987 | 0.663333 ± 0.102334 | 0.501189 ± 0.046920 | 0.672238 ± 0.057764 | 0.482022 ± 0.069747 |
| mlp | 0.630238 ± 0.021488 | 0.673333 ± 0.076920 | 0.511336 ± 0.017787 | 0.680214 ± 0.006127 | 0.480514 ± 0.025601 |
| logistic | 0.611190 ± 0.020733 | 0.586667 ± 0.039791 | 0.481843 ± 0.019639 | 0.678762 ± 0.019305 | 0.479657 ± 0.043382 |
| dummy | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |

## weighted: validation thresholds

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.600000 ± 0.036383 | 0.380000 ± 0.150647 | 0.411869 ± 0.083567 | 0.656857 ± 0.053678 | 0.462107 ± 0.053358 |
| fixed_adam | 0.594524 ± 0.038379 | 0.583333 ± 0.346210 | 0.433369 ± 0.113459 | 0.687405 ± 0.025717 | 0.489580 ± 0.057202 |
| trainable_adam | 0.590476 ± 0.027303 | 0.666667 ± 0.302536 | 0.452288 ± 0.094368 | 0.672238 ± 0.057764 | 0.482022 ± 0.069747 |
| mlp | 0.601905 ± 0.021372 | 0.606667 ± 0.196709 | 0.469669 ± 0.042563 | 0.680214 ± 0.006127 | 0.480514 ± 0.025601 |
| logistic | 0.597381 ± 0.028663 | 0.623333 ± 0.269928 | 0.460607 ± 0.070658 | 0.678762 ± 0.019305 | 0.479657 ± 0.043382 |
| dummy | 0.500000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |

## oversampled: default thresholds

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.609524 ± 0.038346 | 0.663333 ± 0.094575 | 0.491204 ± 0.035325 | 0.658476 ± 0.054345 | 0.460695 ± 0.049735 |
| fixed_adam | 0.618333 ± 0.024909 | 0.656667 ± 0.064118 | 0.498280 ± 0.022949 | 0.685500 ± 0.026817 | 0.485878 ± 0.057196 |
| trainable_adam | 0.610952 ± 0.032092 | 0.653333 ± 0.092346 | 0.490311 ± 0.037898 | 0.667476 ± 0.054473 | 0.475514 ± 0.062783 |
| mlp | 0.627143 ± 0.040181 | 0.660000 ± 0.097610 | 0.505390 ± 0.043989 | 0.658619 ± 0.036785 | 0.460689 ± 0.063346 |
| logistic | 0.617143 ± 0.028317 | 0.610000 ± 0.054772 | 0.491185 ± 0.025915 | 0.675667 ± 0.026636 | 0.476602 ± 0.048701 |
| dummy | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |

## oversampled: validation thresholds

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.594762 ± 0.040429 | 0.446667 ± 0.250389 | 0.411109 ± 0.108077 | 0.658476 ± 0.054345 | 0.460695 ± 0.049735 |
| fixed_adam | 0.593810 ± 0.027139 | 0.613333 ± 0.329436 | 0.443615 ± 0.098749 | 0.685500 ± 0.026817 | 0.485878 ± 0.057196 |
| trainable_adam | 0.599524 ± 0.018645 | 0.633333 ± 0.321671 | 0.453682 ± 0.091129 | 0.667476 ± 0.054473 | 0.475514 ± 0.062783 |
| mlp | 0.607619 ± 0.045188 | 0.646667 ± 0.209298 | 0.479116 ± 0.074479 | 0.658619 ± 0.036785 | 0.460689 ± 0.063346 |
| logistic | 0.599286 ± 0.031459 | 0.580000 ± 0.269361 | 0.452953 ± 0.079466 | 0.675667 ± 0.026636 | 0.476602 ± 0.048701 |
| dummy | 0.500000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |

## Paired differences, weighted validation thresholds

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus logistic | -0.002857 | -0.040000 | -0.027237 | 0.008643 | 0.009923 |
| fixed_adam minus mlp | -0.007381 | -0.023333 | -0.036300 | 0.007190 | 0.009066 |
| fixed_adam minus original | -0.005476 | 0.203333 | 0.021500 | 0.030548 | 0.027474 |
| trainable_adam minus fixed_adam | -0.004048 | 0.083333 | 0.018919 | -0.015167 | -0.007558 |

Positive differences favour the first model. `paired_summary.csv` includes SD, median and range; `paired_differences.csv` retains each seed. Overlapping holdouts are dependent, so these are descriptive differences. No independent-split standard error or t-test is asserted.

Conditional seed-42 stratified paired bootstrap (fixed Adam minus MLP; percentile 95% intervals, 2,000 resamples):

| metric | lower | upper |
| --- | --- | --- |
| balanced_accuracy | -0.088125 | 0.050000 |
| recall | 0.100000 | 0.300000 |
| f1 | -0.059178 | 0.064314 |
| roc_auc | -0.061313 | 0.118812 |
| average_precision | -0.107325 | 0.116057 |

These intervals condition on trained models, selected thresholds, the chosen partition and its class counts. They omit tuning/training uncertainty and are not a global test of superiority. See the full assumptions in EXPERIMENTAL_DESIGN.md.

## Local timing

| model | selected_fit_seconds_mean | selected_fit_seconds_min | selected_fit_seconds_max | search_seconds_mean | search_seconds_min | search_seconds_max |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.050681 | 0.038917 | 0.067080 | 0.050701 | 0.038937 | 0.067101 |
| fixed_adam | 0.360688 | 0.277498 | 0.487637 | 0.686197 | 0.530758 | 0.824290 |
| logistic | 0.051712 | 0.039526 | 0.074604 | 0.095346 | 0.076841 | 0.122368 |
| mlp | 8.154806 | 4.685182 | 15.875217 | 14.601156 | 8.337754 | 24.249855 |
| original | 4.055237 | 3.520729 | 4.808973 | 4.059346 | 3.523909 | 4.813018 |
| trainable_adam | 0.436996 | 0.339154 | 0.566307 | 0.862258 | 0.710779 | 1.119710 |

The pipeline completed 150 candidate fits/evaluations, selected 90 model-condition-split checkpoints, and produced 180 metric rows and 18000 held-out scores. The measured completion-run wall time was 297.282 s; scope: completion run only: repeated dimensionality fits plus initialization sensitivity; initial main-study total wall time unavailable. Initial full-pipeline wall time is unavailable because its process ended during dimension work. Complete per-trial main-study elapsed times survived. Trials include the no-fit dummy reference. All candidate outcomes remain in all_trials.csv. No failed candidate is silently replaced.

`per_split_metrics.csv` additionally contains accuracy, specificity, precision, Brier score, predicted-positive rates and every confusion matrix. `aggregate_metrics.csv` gives mean, SD, median and range for these metrics. Scores are uncalibrated, especially with modified fitting priors. Dummy at .5 predicts the majority class; validation threshold tie-breaking may instead select all-positive predictions, both with BA .5.

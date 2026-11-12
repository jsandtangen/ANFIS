# Final ACIT4620 research summary

Prepared 12 November 2026 from the completed local experiments. This is the evidence package for updating **Credit Risk Classification with a Simplified ANFIS and an MLP**, not an edited final paper or a grade prediction. The existing report and official examination brief are absent from this checkout. Section names below therefore identify the sections that should be changed; their existing numbers, exact wording and layout cannot be verified. Working RQs follow the supplied request.

The original source and saved top-level results remain intact. The baseline reproduction, new comparative study, diagnostic interventions and dimension sensitivity are distinguished throughout. Every numeric result below comes from a saved CSV/JSON or a calculation on those artifacts; no missing historical results have been estimated.

## Evidence boundary and original reproduction

The inspected original repository contained main.py, README.md, Run.md, unpinned requirements, one CSV, four result files and ten figures. No tests, report, notebooks, saved models, repeated-run logs, 5/6-input code, exam instructions or paper PDFs were present. Original Git HEAD was df2b4e97ecc5b111ff8cd231227f302746068de4. The old virtual environment pointed to a missing interpreter. A new Python 3.11.13 environment was installed without altering the old environment.

Executing unchanged main.py through reproduce_original.py reproduced every saved metric exactly:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| ANFIS | 0.730000 | 0.571429 | 0.400000 | 0.470588 | 0.690357 |
| Neural Network | 0.690000 | 0.485294 | 0.550000 | 0.515625 | 0.692738 |

Original ANFIS used weighted BCE; original MLP used random oversampling and an internal early-stopping split. Thresholds were selected on external validation data: .51 for ANFIS and .50 for MLP. The split was 640 fitting / 160 validation / 200 test, with class counts 448/192, 112/48 and 140/60 good/bad. ANFIS confusion matrix was [[122,18],[36,24]], giving balanced accuracy 0.635714. MLP was [[105,35],[27,33]], giving balanced accuracy 0.650000. These are single-split results. See [REPRODUCTION_REPORT.md](results/research/REPRODUCTION_REPORT.md) and reproduction_check.csv.

The task's previous repeated class-weighted means (ANFIS/MLP BA .536±.044/.605±.028; recall .417±.417/.707±.125; F1 .304±.167/.492±.024; AUC .545±.082/.671±.032; AP .361±.065/.466±.043) have no corresponding original code, per-split artifacts or settings in this checkout. They remain unverified. The new results below must not be presented as reproducing those historical means.

## 1. RQ1: predictive performance under class imbalance

### Protocol and metric interpretation

Five paired stratified holdouts used seeds 42,7,19,73,101. Every model within a split saw the same fitting, validation and test source rows. StandardScaler fitted on the 640 fitting observations only. Good=0 and bad=1. No test set entered epoch, configuration, threshold or feature selection. This extension was designed after inspecting historical results, so it remains exploratory at the project level.

Conditions: **none**, **weighted** (good=1,bad=2.33), **oversampled** (448 fitting observations of each class, duplicates confined to fitting). Gaussian initialization uses the fitting inputs available in that condition. Standardization always precedes resampling and uses the original 640 rows. All non-dummy models get matched fitting conditions; the dummy remains the same constant-prior reference.

Models: **original** means unchanged fixed-premise SimpleANFIS, 500 full-batch GD steps at .02; **fixed_adam** retains full-grid first-order rules and fixed quantile Gaussians but uses stable exact normalization, Adam, L2 and validation stopping; **trainable_adam** additionally learns Gaussian centres and log widths; **mlp** retains (16,8) ReLU layers with externally controlled stopping; **logistic** is L2 logistic regression; **dummy** has constant bad score .3 and predicts majority good at .5.

Two candidates per tuned family: ANFIS (lr=.01,L2=.001) or (.03,.01); MLP (lr=.001,alpha=.0001) or (.003,.01); logistic C=.1 or 1. ANFIS/MLP cap at 400 epochs, patience 50. ANFIS stopping restores minimum validation penalized BCE; MLP restores maximum validation balanced accuracy after threshold search. Final trial selection uses validation balanced accuracy for all tuned families. Threshold search checks .20–.80 by .01, first maximum wins. Model search costs and stopping criteria differ transparently; this is a bounded comparison, not equal FLOPs or proof of optimal tuning. All 150 candidate evaluations, including 15 no-fit dummy references, are recorded in all_trials.csv.

The following tables are mean ± **sample SD (ddof=1)** over five overlapping holdouts. Accuracy, balanced accuracy (BA), positive precision/recall/F1, specificity, ROC-AUC, average precision (AP), Brier score, predicted-positive rate and times are all included. Brier is lower-is-better; all other predictive metrics are higher-is-better subject to tradeoffs. Confusion-count means/SDs are included as descriptive counts, with true good/bad rows and predicted good/bad columns. AUC/AP/Brier are repeated between threshold modes because changing a decision threshold does not change scores. Full precision per-split observations remain in per_split_metrics.csv and test_predictions.csv.

### Complete comparison tables

#### none; fixed 0.5 threshold

| model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.708000 ± 0.022804 | 0.538095 ± 0.033077 | 0.537534 ± 0.173495 | 0.113333 ± 0.059395 | 0.186179 ± 0.091679 | 0.962857 ± 0.009313 |
| fixed_adam | 0.706000 ± 0.018507 | 0.530952 ± 0.026527 | 0.548052 ± 0.219482 | 0.093333 ± 0.050827 | 0.157447 ± 0.078953 | 0.968571 ± 0.014812 |
| trainable_adam | 0.710000 ± 0.017678 | 0.545238 ± 0.027932 | 0.558615 ± 0.134846 | 0.133333 ± 0.061237 | 0.211906 ± 0.087800 | 0.957143 ± 0.019562 |
| mlp | 0.691000 ± 0.022192 | 0.520238 ± 0.027523 | 0.342879 ± 0.240635 | 0.093333 ± 0.069322 | 0.145092 ± 0.105490 | 0.947143 ± 0.039318 |
| logistic | 0.716000 ± 0.018507 | 0.551429 ± 0.028013 | 0.610513 ± 0.145800 | 0.140000 ± 0.056026 | 0.225318 ± 0.079713 | 0.962857 ± 0.014639 |
| dummy | 0.700000 ± 0.000000 | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |

| model | roc_auc | average_precision | brier | positive_rate | threshold |
| --- | --- | --- | --- | --- | --- |
| original | 0.599333 ± 0.053290 | 0.415089 ± 0.060609 | 0.243684 ± 0.000989 | 0.060000 ± 0.014142 | 0.500000 ± 0.000000 |
| fixed_adam | 0.647476 ± 0.040371 | 0.449626 ± 0.083711 | 0.203111 ± 0.010769 | 0.050000 ± 0.018371 | 0.500000 ± 0.000000 |
| trainable_adam | 0.653905 ± 0.045023 | 0.462045 ± 0.076696 | 0.197275 ± 0.009827 | 0.070000 ± 0.027157 | 0.500000 ± 0.000000 |
| mlp | 0.657440 ± 0.055076 | 0.442856 ± 0.074025 | 0.198830 ± 0.008863 | 0.065000 ± 0.043445 | 0.500000 ± 0.000000 |
| logistic | 0.679214 ± 0.019629 | 0.477622 ± 0.042360 | 0.192847 ± 0.005226 | 0.068000 ± 0.020797 | 0.500000 ± 0.000000 |
| dummy | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 0.210000 ± 0.000000 | 0.000000 ± 0.000000 | 0.500000 ± 0.000000 |

| model | tn | fp | fn | tp | selected_fit_seconds | search_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| original | 134.800000 ± 1.303840 | 5.200000 ± 1.303840 | 53.200000 ± 3.563706 | 6.800000 ± 3.563706 | 3.768065 ± 0.535063 | 3.772280 ± 0.536466 |
| fixed_adam | 135.600000 ± 2.073644 | 4.400000 ± 2.073644 | 54.400000 ± 3.049590 | 5.600000 ± 3.049590 | 0.553522 ± 0.192398 | 0.998521 ± 0.280886 |
| trainable_adam | 134.000000 ± 2.738613 | 6.000000 ± 2.738613 | 52.000000 ± 3.674235 | 8.000000 ± 3.674235 | 0.521082 ± 0.065762 | 1.023794 ± 0.106109 |
| mlp | 132.600000 ± 5.504544 | 7.400000 ± 5.504544 | 54.400000 ± 4.159327 | 5.600000 ± 4.159327 | 5.607503 ± 1.604154 | 12.847630 ± 3.673036 |
| logistic | 134.800000 ± 2.049390 | 5.200000 ± 2.049390 | 51.600000 ± 3.361547 | 8.400000 ± 3.361547 | 0.051146 ± 0.010962 | 0.110804 ± 0.024086 |
| dummy | 140.000000 ± 0.000000 | 0.000000 ± 0.000000 | 60.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.050424 ± 0.012291 | 0.050449 ± 0.012296 |

#### none; validation-selected threshold

| model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.593000 ± 0.136042 | 0.540714 ± 0.042769 | 0.369367 ± 0.052204 | 0.410000 ± 0.398365 | 0.301246 ± 0.192921 | 0.671429 ± 0.358640 |
| fixed_adam | 0.564000 ± 0.107958 | 0.561905 ± 0.041836 | 0.359290 ± 0.034665 | 0.556667 ± 0.337598 | 0.391358 ± 0.154884 | 0.567143 ± 0.291924 |
| trainable_adam | 0.608000 ± 0.099411 | 0.613333 ± 0.049711 | 0.436063 ± 0.075854 | 0.626667 ± 0.301984 | 0.459281 ± 0.138446 | 0.600000 ± 0.258775 |
| mlp | 0.614000 ± 0.064750 | 0.619524 ± 0.059181 | 0.410708 ± 0.058881 | 0.633333 ± 0.116070 | 0.495399 ± 0.068138 | 0.605714 ± 0.101192 |
| logistic | 0.598000 ± 0.126915 | 0.595714 ± 0.031096 | 0.438942 ± 0.097860 | 0.590000 ± 0.298515 | 0.446252 ± 0.094911 | 0.601429 ± 0.304658 |
| dummy | 0.300000 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.000000 ± 0.000000 |

| model | roc_auc | average_precision | brier | positive_rate | threshold |
| --- | --- | --- | --- | --- | --- |
| original | 0.599333 ± 0.053290 | 0.415089 ± 0.060609 | 0.243684 ± 0.000989 | 0.353000 ± 0.368927 | 0.490000 ± 0.012247 |
| fixed_adam | 0.647476 ± 0.040371 | 0.449626 ± 0.083711 | 0.203111 ± 0.010769 | 0.470000 ± 0.303933 | 0.380000 ± 0.068191 |
| trainable_adam | 0.653905 ± 0.045023 | 0.462045 ± 0.076696 | 0.197275 ± 0.009827 | 0.468000 ± 0.268621 | 0.340000 ± 0.095656 |
| mlp | 0.657440 ± 0.055076 | 0.442856 ± 0.074025 | 0.198830 ± 0.008863 | 0.466000 ± 0.090926 | 0.320000 ± 0.038079 |
| logistic | 0.679214 ± 0.019629 | 0.477622 ± 0.042360 | 0.192847 ± 0.005226 | 0.456000 ± 0.301484 | 0.306000 ± 0.083247 |
| dummy | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 0.210000 ± 0.000000 | 1.000000 ± 0.000000 | 0.200000 ± 0.000000 |

| model | tn | fp | fn | tp | selected_fit_seconds | search_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| original | 94.000000 ± 50.209561 | 46.000000 ± 50.209561 | 35.400000 ± 23.901883 | 24.600000 ± 23.901883 | 3.768065 ± 0.535063 | 3.772280 ± 0.536466 |
| fixed_adam | 79.400000 ± 40.869304 | 60.600000 ± 40.869304 | 26.600000 ± 20.255863 | 33.400000 ± 20.255863 | 0.553522 ± 0.192398 | 0.998521 ± 0.280886 |
| trainable_adam | 84.000000 ± 36.228442 | 56.000000 ± 36.228442 | 22.400000 ± 18.119051 | 37.600000 ± 18.119051 | 0.521082 ± 0.065762 | 1.023794 ± 0.106109 |
| mlp | 84.800000 ± 14.166863 | 55.200000 ± 14.166863 | 22.000000 ± 6.964194 | 38.000000 ± 6.964194 | 5.607503 ± 1.604154 | 12.847630 ± 3.673036 |
| logistic | 84.200000 ± 42.652081 | 55.800000 ± 42.652081 | 24.600000 ± 17.910891 | 35.400000 ± 17.910891 | 0.051146 ± 0.010962 | 0.110804 ± 0.024086 |
| dummy | 0.000000 ± 0.000000 | 140.000000 ± 0.000000 | 0.000000 ± 0.000000 | 60.000000 ± 0.000000 | 0.050424 ± 0.012291 | 0.050449 ± 0.012296 |

#### weighted; fixed 0.5 threshold

| model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.586000 ± 0.065803 | 0.618571 ± 0.051243 | 0.397423 ± 0.045109 | 0.700000 ± 0.081650 | 0.504647 ± 0.044243 | 0.537143 ± 0.102195 |
| fixed_adam | 0.607000 ± 0.034749 | 0.627857 ± 0.031995 | 0.408190 ± 0.028487 | 0.680000 ± 0.061689 | 0.509220 ± 0.033107 | 0.575714 ± 0.053595 |
| trainable_adam | 0.607000 ± 0.034387 | 0.623095 ± 0.036987 | 0.405581 ± 0.029671 | 0.663333 ± 0.102334 | 0.501189 ± 0.046920 | 0.582857 ± 0.067876 |
| mlp | 0.613000 ± 0.051430 | 0.630238 ± 0.021488 | 0.416959 ± 0.040604 | 0.673333 ± 0.076920 | 0.511336 ± 0.017787 | 0.587143 ± 0.102941 |
| logistic | 0.621000 ± 0.033053 | 0.611190 ± 0.020733 | 0.410587 ± 0.028770 | 0.586667 ± 0.039791 | 0.481843 ± 0.019639 | 0.635714 ± 0.057366 |
| dummy | 0.700000 ± 0.000000 | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |

| model | roc_auc | average_precision | brier | positive_rate | threshold |
| --- | --- | --- | --- | --- | --- |
| original | 0.656857 ± 0.053678 | 0.462107 ± 0.053358 | 0.248619 ± 0.001329 | 0.534000 ± 0.084291 | 0.500000 ± 0.000000 |
| fixed_adam | 0.687405 ± 0.025717 | 0.489580 ± 0.057202 | 0.236507 ± 0.006003 | 0.501000 ± 0.047880 | 0.500000 ± 0.000000 |
| trainable_adam | 0.672238 ± 0.057764 | 0.482022 ± 0.069747 | 0.238024 ± 0.005838 | 0.491000 ± 0.072232 | 0.500000 ± 0.000000 |
| mlp | 0.680214 ± 0.006127 | 0.480514 ± 0.025601 | 0.228912 ± 0.005366 | 0.491000 ± 0.093835 | 0.500000 ± 0.000000 |
| logistic | 0.678762 ± 0.019305 | 0.479657 ± 0.043382 | 0.228622 ± 0.005842 | 0.431000 ± 0.049168 | 0.500000 ± 0.000000 |
| dummy | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 0.210000 ± 0.000000 | 0.000000 ± 0.000000 | 0.500000 ± 0.000000 |

| model | tn | fp | fn | tp | selected_fit_seconds | search_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| original | 75.200000 ± 14.307341 | 64.800000 ± 14.307341 | 18.000000 ± 4.898979 | 42.000000 ± 4.898979 | 4.055237 ± 0.620347 | 4.059346 ± 0.620743 |
| fixed_adam | 80.600000 ± 7.503333 | 59.400000 ± 7.503333 | 19.200000 ± 3.701351 | 40.800000 ± 3.701351 | 0.360688 ± 0.080217 | 0.686197 ± 0.116776 |
| trainable_adam | 81.600000 ± 9.502631 | 58.400000 ± 9.502631 | 20.200000 ± 6.140033 | 39.800000 ± 6.140033 | 0.436996 ± 0.095358 | 0.862258 ± 0.159582 |
| mlp | 82.200000 ± 14.411801 | 57.800000 ± 14.411801 | 19.600000 ± 4.615192 | 40.400000 ± 4.615192 | 8.154806 ± 4.565428 | 14.601156 ± 6.072871 |
| logistic | 89.000000 ± 8.031189 | 51.000000 ± 8.031189 | 24.800000 ± 2.387467 | 35.200000 ± 2.387467 | 0.051712 ± 0.015519 | 0.095346 ± 0.021674 |
| dummy | 140.000000 ± 0.000000 | 0.000000 ± 0.000000 | 60.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.050681 ± 0.012367 | 0.050701 ± 0.012367 |

#### weighted; validation-selected threshold

| model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.688000 ± 0.043243 | 0.600000 ± 0.036383 | 0.488254 ± 0.068517 | 0.380000 ± 0.150647 | 0.411869 ± 0.083567 | 0.820000 ± 0.111506 |
| fixed_adam | 0.599000 ± 0.116372 | 0.594524 ± 0.038379 | 0.429764 ± 0.089916 | 0.583333 ± 0.346210 | 0.433369 ± 0.113459 | 0.605714 ± 0.308733 |
| trainable_adam | 0.560000 ± 0.095131 | 0.590476 ± 0.027303 | 0.404051 ± 0.094459 | 0.666667 ± 0.302536 | 0.452288 ± 0.094368 | 0.514286 ± 0.263270 |
| mlp | 0.600000 ± 0.082991 | 0.601905 ± 0.021372 | 0.406762 ± 0.044903 | 0.606667 ± 0.196709 | 0.469669 ± 0.042563 | 0.597143 ± 0.199528 |
| logistic | 0.587000 ± 0.119509 | 0.597381 ± 0.028663 | 0.419242 ± 0.077530 | 0.623333 ± 0.269928 | 0.460607 ± 0.070658 | 0.571429 ± 0.282437 |
| dummy | 0.300000 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.000000 ± 0.000000 |

| model | roc_auc | average_precision | brier | positive_rate | threshold |
| --- | --- | --- | --- | --- | --- |
| original | 0.656857 ± 0.053678 | 0.462107 ± 0.053358 | 0.248619 ± 0.001329 | 0.240000 ± 0.120000 | 0.508000 ± 0.004472 |
| fixed_adam | 0.687405 ± 0.025717 | 0.489580 ± 0.057202 | 0.236507 ± 0.006003 | 0.451000 ± 0.318500 | 0.512000 ± 0.048683 |
| trainable_adam | 0.672238 ± 0.057764 | 0.482022 ± 0.069747 | 0.238024 ± 0.005838 | 0.540000 ± 0.274500 | 0.486000 ± 0.033615 |
| mlp | 0.680214 ± 0.006127 | 0.480514 ± 0.025601 | 0.228912 ± 0.005366 | 0.464000 ± 0.197718 | 0.498000 ± 0.101587 |
| logistic | 0.678762 ± 0.019305 | 0.479657 ± 0.043382 | 0.228622 ± 0.005842 | 0.487000 ± 0.277502 | 0.488000 ± 0.087864 |
| dummy | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 0.210000 ± 0.000000 | 1.000000 ± 0.000000 | 0.200000 ± 0.000000 |

| model | tn | fp | fn | tp | selected_fit_seconds | search_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| original | 114.800000 ± 15.610894 | 25.200000 ± 15.610894 | 37.200000 ± 9.038805 | 22.800000 ± 9.038805 | 4.055237 ± 0.620347 | 4.059346 ± 0.620743 |
| fixed_adam | 84.800000 ± 43.222679 | 55.200000 ± 43.222679 | 25.000000 ± 20.772578 | 35.000000 ± 20.772578 | 0.360688 ± 0.080217 | 0.686197 ± 0.116776 |
| trainable_adam | 72.000000 ± 36.857835 | 68.000000 ± 36.857835 | 20.000000 ± 18.152135 | 40.000000 ± 18.152135 | 0.436996 ± 0.095358 | 0.862258 ± 0.159582 |
| mlp | 83.600000 ± 27.933850 | 56.400000 ± 27.933850 | 23.600000 ± 11.802542 | 36.400000 ± 11.802542 | 8.154806 ± 4.565428 | 14.601156 ± 6.072871 |
| logistic | 80.000000 ± 39.541118 | 60.000000 ± 39.541118 | 22.600000 ± 16.195678 | 37.400000 ± 16.195678 | 0.051712 ± 0.015519 | 0.095346 ± 0.021674 |
| dummy | 0.000000 ± 0.000000 | 140.000000 ± 0.000000 | 0.000000 ± 0.000000 | 60.000000 ± 0.000000 | 0.050681 ± 0.012367 | 0.050701 ± 0.012367 |

#### oversampled; fixed 0.5 threshold

| model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.588000 ± 0.055295 | 0.609524 ± 0.038346 | 0.394202 ± 0.038687 | 0.663333 ± 0.094575 | 0.491204 ± 0.035325 | 0.555714 ± 0.101695 |
| fixed_adam | 0.603000 ± 0.041773 | 0.618333 ± 0.024909 | 0.404153 ± 0.031296 | 0.656667 ± 0.064118 | 0.498280 ± 0.022949 | 0.580000 ± 0.078311 |
| trainable_adam | 0.594000 ± 0.042042 | 0.610952 ± 0.032092 | 0.395705 ± 0.034952 | 0.653333 ± 0.092346 | 0.490311 ± 0.037898 | 0.568571 ± 0.082931 |
| mlp | 0.614000 ± 0.043932 | 0.627143 ± 0.040181 | 0.412718 ± 0.036646 | 0.660000 ± 0.097610 | 0.505390 ± 0.043989 | 0.594286 ± 0.078311 |
| logistic | 0.620000 ± 0.044441 | 0.617143 ± 0.028317 | 0.414220 ± 0.037970 | 0.610000 ± 0.054772 | 0.491185 ± 0.025915 | 0.624286 ± 0.077030 |
| dummy | 0.700000 ± 0.000000 | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |

| model | roc_auc | average_precision | brier | positive_rate | threshold |
| --- | --- | --- | --- | --- | --- |
| original | 0.658476 ± 0.054345 | 0.460695 ± 0.049735 | 0.248381 ± 0.001278 | 0.510000 ± 0.093207 | 0.500000 ± 0.000000 |
| fixed_adam | 0.685500 ± 0.026817 | 0.485878 ± 0.057196 | 0.238755 ± 0.007079 | 0.491000 ± 0.070746 | 0.500000 ± 0.000000 |
| trainable_adam | 0.667476 ± 0.054473 | 0.475514 ± 0.062783 | 0.240055 ± 0.006433 | 0.498000 ± 0.080669 | 0.500000 ± 0.000000 |
| mlp | 0.658619 ± 0.036785 | 0.460689 ± 0.063346 | 0.235558 ± 0.011994 | 0.482000 ± 0.076125 | 0.500000 ± 0.000000 |
| logistic | 0.675667 ± 0.026636 | 0.476602 ± 0.048701 | 0.229693 ± 0.007409 | 0.446000 ± 0.066182 | 0.500000 ± 0.000000 |
| dummy | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 0.210000 ± 0.000000 | 0.000000 ± 0.000000 | 0.500000 ± 0.000000 |

| model | tn | fp | fn | tp | selected_fit_seconds | search_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| original | 77.800000 ± 14.237275 | 62.200000 ± 14.237275 | 20.200000 ± 5.674504 | 39.800000 ± 5.674504 | 4.233243 ± 0.381260 | 4.236277 ± 0.381278 |
| fixed_adam | 81.200000 ± 10.963576 | 58.800000 ± 10.963576 | 20.600000 ± 3.847077 | 39.400000 ± 3.847077 | 0.440205 ± 0.063546 | 0.829157 ± 0.089770 |
| trainable_adam | 79.600000 ± 11.610340 | 60.400000 ± 11.610340 | 20.800000 ± 5.540758 | 39.200000 ± 5.540758 | 0.553931 ± 0.218404 | 1.065788 ± 0.316725 |
| mlp | 83.200000 ± 10.963576 | 56.800000 ± 10.963576 | 20.400000 ± 5.856620 | 39.600000 ± 5.856620 | 6.687502 ± 5.321556 | 10.763486 ± 5.278244 |
| logistic | 87.400000 ± 10.784248 | 52.600000 ± 10.784248 | 23.400000 ± 3.286335 | 36.600000 ± 3.286335 | 0.060836 ± 0.018515 | 0.107524 ± 0.023173 |
| dummy | 140.000000 ± 0.000000 | 0.000000 ± 0.000000 | 60.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.050391 ± 0.010348 | 0.050413 ± 0.010349 |

#### oversampled; validation-selected threshold

| model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.654000 ± 0.054704 | 0.594762 ± 0.040429 | 0.441836 ± 0.046911 | 0.446667 ± 0.250389 | 0.411109 ± 0.108077 | 0.742857 ± 0.180702 |
| fixed_adam | 0.586000 ± 0.116372 | 0.593810 ± 0.027139 | 0.415574 ± 0.066236 | 0.613333 ± 0.329436 | 0.443615 ± 0.098749 | 0.574286 ± 0.304415 |
| trainable_adam | 0.586000 ± 0.115942 | 0.599524 ± 0.018645 | 0.431789 ± 0.102046 | 0.633333 ± 0.321671 | 0.453682 ± 0.091129 | 0.565714 ± 0.302177 |
| mlp | 0.592000 ± 0.079498 | 0.607619 ± 0.045188 | 0.398801 ± 0.049182 | 0.646667 ± 0.209298 | 0.479116 ± 0.074479 | 0.568571 ± 0.187668 |
| logistic | 0.607000 ± 0.105155 | 0.599286 ± 0.031459 | 0.423215 ± 0.065180 | 0.580000 ± 0.269361 | 0.452953 ± 0.079466 | 0.618571 ± 0.260033 |
| dummy | 0.300000 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.000000 ± 0.000000 |

| model | roc_auc | average_precision | brier | positive_rate | threshold |
| --- | --- | --- | --- | --- | --- |
| original | 0.658476 ± 0.054345 | 0.460695 ± 0.049735 | 0.248381 ± 0.001278 | 0.314000 ± 0.200730 | 0.506000 ± 0.005477 |
| fixed_adam | 0.685500 ± 0.026817 | 0.485878 ± 0.057196 | 0.238755 ± 0.007079 | 0.482000 ± 0.311139 | 0.506000 ± 0.029665 |
| trainable_adam | 0.667476 ± 0.054473 | 0.475514 ± 0.062783 | 0.240055 ± 0.006433 | 0.494000 ± 0.307681 | 0.508000 ± 0.032711 |
| mlp | 0.658619 ± 0.036785 | 0.460689 ± 0.063346 | 0.235558 ± 0.011994 | 0.496000 ± 0.189947 | 0.494000 ± 0.055498 |
| logistic | 0.675667 ± 0.026636 | 0.476602 ± 0.048701 | 0.229693 ± 0.007409 | 0.441000 ± 0.261281 | 0.506000 ± 0.078613 |
| dummy | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 | 0.210000 ± 0.000000 | 1.000000 ± 0.000000 | 0.200000 ± 0.000000 |

| model | tn | fp | fn | tp | selected_fit_seconds | search_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| original | 104.000000 ± 25.298221 | 36.000000 ± 25.298221 | 33.200000 ± 15.023315 | 26.800000 ± 15.023315 | 4.233243 ± 0.381260 | 4.236277 ± 0.381278 |
| fixed_adam | 80.400000 ± 42.618071 | 59.600000 ± 42.618071 | 23.200000 ± 19.766133 | 36.800000 ± 19.766133 | 0.440205 ± 0.063546 | 0.829157 ± 0.089770 |
| trainable_adam | 79.200000 ± 42.304846 | 60.800000 ± 42.304846 | 22.000000 ± 19.300259 | 38.000000 ± 19.300259 | 0.553931 ± 0.218404 | 1.065788 ± 0.316725 |
| mlp | 79.600000 ± 26.273561 | 60.400000 ± 26.273561 | 21.200000 ± 12.557866 | 38.800000 ± 12.557866 | 6.687502 ± 5.321556 | 10.763486 ± 5.278244 |
| logistic | 86.600000 ± 36.404670 | 53.400000 ± 36.404670 | 25.200000 ± 16.161683 | 34.800000 ± 16.161683 | 0.060836 ± 0.018515 | 0.107524 ± 0.023173 |
| dummy | 0.000000 ± 0.000000 | 140.000000 ± 0.000000 | 0.000000 ± 0.000000 | 60.000000 ± 0.000000 | 0.050391 ± 0.010348 | 0.050413 ± 0.010349 |

### Which model performs best for which objective?

| objective | threshold_mode | model | condition | mean |
| --- | --- | --- | --- | --- |
| accuracy | default | logistic | none | 0.716000 |
| accuracy | validation | original | weighted | 0.688000 |
| balanced_accuracy | default | mlp | weighted | 0.630238 |
| balanced_accuracy | validation | mlp | none | 0.619524 |
| precision | default | logistic | none | 0.610513 |
| precision | validation | original | weighted | 0.488254 |
| recall | default | original | weighted | 0.700000 |
| recall | validation | trainable_adam | weighted | 0.666667 |
| f1 | default | mlp | weighted | 0.511336 |
| f1 | validation | mlp | none | 0.495399 |
| specificity | default | fixed_adam | none | 0.968571 |
| specificity | validation | original | weighted | 0.820000 |
| roc_auc | default | fixed_adam | weighted | 0.687405 |
| roc_auc | validation | fixed_adam | weighted | 0.687405 |
| average_precision | default | fixed_adam | weighted | 0.489580 |
| average_precision | validation | fixed_adam | weighted | 0.489580 |
| brier | default | logistic | none | 0.192847 |
| brier | validation | logistic | none | 0.192847 |

This is a retrospective description of observed means, not a selection of a final model on test data. It is not a statistical winner table. Dummy is excluded because all-positive prediction trivially attains recall 1 and all-negative prediction specificity 1. The selected-threshold dummy predicts every observation bad: the validation BA ties at .5 and the first threshold .20 wins. Its F1 .461538 is a useful warning that F1 alone can reward a degenerate policy on this dataset.

For overall accuracy and uncalibrated Brier score, unadjusted logistic regression is the strongest observed reference; its low bad-credit recall at .5 shows why accuracy alone is inadequate. Under weighted default-threshold evaluation, MLP and fixed Adam are close in BA/F1, while fixed Adam has the highest observed weighted AUC/AP. Allowing trainable premises does not improve the weighted means relative to fixed Adam. The unadjusted MLP with a validation-selected threshold has the highest observed selected-threshold BA. None of these facts establishes a universally best classifier.

Training costs include fitting plus candidate-specific validation computations; search_seconds includes all trials in that model family, while selected_fit_seconds measures the selected candidate's computation. Timing is local, one-thread elapsed time. Some initial development diagnostics overlapped the first incomplete dimension run; final dimension benchmark timings come from its separate completion run. No CPU-independent efficiency claim is made.

### Differences and uncertainty

Paired differences: none, default; first model minus second, mean ± SD.

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus mlp | 0.010714 ± 0.024815 | 0.000000 ± 0.060093 | 0.012355 ± 0.095283 | -0.009964 ± 0.039157 | 0.006770 ± 0.033630 |
| trainable_adam minus fixed_adam | 0.014286 ± 0.009524 | 0.040000 ± 0.019003 | 0.054460 ± 0.027146 | 0.006429 ± 0.023200 | 0.012419 ± 0.040984 |
| fixed_adam minus original | -0.007143 ± 0.019867 | -0.020000 ± 0.047726 | -0.028732 ± 0.067003 | 0.048143 ± 0.054149 | 0.034537 ± 0.052939 |
| fixed_adam minus logistic | -0.020476 ± 0.030356 | -0.046667 ± 0.066039 | -0.067872 ± 0.091597 | -0.031738 ± 0.032251 | -0.027996 ± 0.055768 |

Paired differences: none, validation; first model minus second, mean ± SD.

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus mlp | -0.057619 ± 0.028104 | -0.076667 ± 0.235879 | -0.104041 ± 0.095874 | -0.009964 ± 0.039157 | 0.006770 ± 0.033630 |
| trainable_adam minus fixed_adam | 0.051429 ± 0.032323 | 0.070000 ± 0.241925 | 0.067923 ± 0.067167 | 0.006429 ± 0.023200 | 0.012419 ± 0.040984 |
| fixed_adam minus original | 0.021190 ± 0.022797 | 0.146667 ± 0.234047 | 0.090112 ± 0.116406 | 0.048143 ± 0.054149 | 0.034537 ± 0.052939 |
| fixed_adam minus logistic | -0.033810 ± 0.028380 | -0.033333 ± 0.111181 | -0.054894 ± 0.068346 | -0.031738 ± 0.032251 | -0.027996 ± 0.055768 |

Paired differences: weighted, default; first model minus second, mean ± SD.

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus mlp | -0.002381 ± 0.016517 | 0.006667 ± 0.096896 | -0.002116 ± 0.022952 | 0.007190 ± 0.022551 | 0.009066 ± 0.049826 |
| trainable_adam minus fixed_adam | -0.004762 ± 0.018133 | -0.016667 ± 0.048591 | -0.008031 ± 0.025779 | -0.015167 ± 0.037176 | -0.007558 ± 0.018472 |
| fixed_adam minus original | 0.009286 ± 0.049426 | -0.020000 ± 0.091591 | 0.004573 ± 0.037682 | 0.030548 ± 0.031240 | 0.027474 ± 0.013422 |
| fixed_adam minus logistic | 0.016667 ± 0.019029 | 0.093333 ± 0.052175 | 0.027377 ± 0.021573 | 0.008643 ± 0.012736 | 0.009923 ± 0.020742 |

Paired differences: weighted, validation; first model minus second, mean ± SD.

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus mlp | -0.007381 ± 0.029802 | -0.023333 ± 0.188414 | -0.036300 ± 0.079096 | 0.007190 ± 0.022551 | 0.009066 ± 0.049826 |
| trainable_adam minus fixed_adam | -0.004048 ± 0.019427 | 0.083333 ± 0.131233 | 0.018919 ± 0.050565 | -0.015167 ± 0.037176 | -0.007558 ± 0.018472 |
| fixed_adam minus original | -0.005476 ± 0.023759 | 0.203333 ± 0.246757 | 0.021500 ± 0.038088 | 0.030548 ± 0.031240 | 0.027474 ± 0.013422 |
| fixed_adam minus logistic | -0.002857 ± 0.042346 | -0.040000 ± 0.107108 | -0.027237 ± 0.072677 | 0.008643 ± 0.012736 | 0.009923 ± 0.020742 |

Paired differences: oversampled, default; first model minus second, mean ± SD.

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus mlp | -0.008810 ± 0.059765 | -0.003333 ± 0.090830 | -0.007109 ± 0.062966 | 0.026881 ± 0.043043 | 0.025189 ± 0.029005 |
| trainable_adam minus fixed_adam | -0.007381 ± 0.030098 | -0.003333 ± 0.046248 | -0.007969 ± 0.033995 | -0.018024 ± 0.034591 | -0.010364 ± 0.009845 |
| fixed_adam minus original | 0.008810 ± 0.049190 | -0.006667 ± 0.093986 | 0.007076 ± 0.041782 | 0.027024 ± 0.031059 | 0.025183 ± 0.014094 |
| fixed_adam minus logistic | 0.001190 ± 0.028159 | 0.046667 ± 0.044721 | 0.007095 ± 0.030096 | 0.009833 ± 0.013796 | 0.009276 ± 0.024336 |

Paired differences: oversampled, validation; first model minus second, mean ± SD.

| comparison | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| fixed_adam minus mlp | -0.013810 ± 0.036911 | -0.033333 ± 0.385501 | -0.035500 ± 0.114584 | 0.026881 ± 0.043043 | 0.025189 ± 0.029005 |
| trainable_adam minus fixed_adam | 0.005714 ± 0.009427 | 0.020000 ± 0.027386 | 0.010066 ± 0.015042 | -0.018024 ± 0.034591 | -0.010364 ± 0.009845 |
| fixed_adam minus original | -0.000952 ± 0.029480 | 0.166667 ± 0.290832 | 0.032507 ± 0.064836 | 0.027024 ± 0.031059 | 0.025183 ± 0.014094 |
| fixed_adam minus logistic | -0.005476 ± 0.008813 | 0.033333 ± 0.071686 | -0.009337 ± 0.020722 | 0.009833 ± 0.013796 | 0.009276 ± 0.024336 |

The saved paired_summary.csv additionally includes median and range; per-split effects are in paired_differences.csv. Repeated test sets overlap and models share many fitting observations. The SDs describe split sensitivity, not the standard error of five independent studies. No ordinary paired t-test, independent-split CI, or claim of statistical significance is justified here.

A supplementary seed-42 weighted, validation-threshold comparison used 2,000 stratified paired percentile bootstrap resamples of that test set; difference is fixed Adam minus MLP:

| metric | lower | upper |
| --- | --- | --- |
| balanced_accuracy | -0.088125 | 0.050000 |
| recall | 0.100000 | 0.300000 |
| f1 | -0.059178 | 0.064314 |
| roc_auc | -0.061313 | 0.118812 |
| average_precision | -0.107325 | 0.116057 |

These 95% intervals condition on those fitted models, chosen configurations, thresholds, test partition and class counts. They omit fitting/tuning uncertainty and do not generalize to all splits. In particular, seed-42 recall advantage can coexist with a different mean recall ordering across the five partitions. Many comparisons were inspected; all are exploratory. Each test partition contains only 60 bad cases, so recall moves in steps of 1/60.

Validation-selected thresholds sometimes reduce held-out BA or F1 relative to .5. This does not imply test leakage; it demonstrates limited transfer from a small, noisy validation set. The two threshold policies must remain separately labelled in the report. No operational costs or calibrated lending probabilities were estimated.

## 2. RQ2: rule complexity and dimensionality

### Exact structural counts

| inputs | rules | trained_consequent_parameters | fixed_premise_scalars | total_stored_fuzzy_scalars | test_three_matrix_workspace_bytes |
| --- | --- | --- | --- | --- | --- |
| 4 | 81 | 405 | 24 | 429 | 388800 |
| 5 | 243 | 1458 | 30 | 1488 | 1166400 |
| 6 | 729 | 5103 | 36 | 5139 | 3499200 |

Original/fixed models train only consequent coefficients, including biases. Their 6d premise scalars are frozen model state. If all premises were learned, the trainable count would be 429/1488/5139; only the four-input trainable variant was executed. The three-matrix workspace counts three N-by-R float64 test arrays, not peak RAM, interpreter overhead or every temporary. OS memory queries were denied, so no measured peak-memory result is claimed.

### Completed dimension results

All **30** planned dimension fits completed: 3 seeds × 5 feature-set specifications × 2 ANFIS protocols. They use the weighted condition, matched partitions, original GD .02/500 and fixed Adam .01/L2=.001 with validation stopping. No dimension-specific tuning was performed. Base inputs are month_duration, credit_amount, payment_to_income_ratio, age. Requested additions are residence_since (5) and then n_credits (6). Separate train_MI sensitivity adds the fitting-only top one/two discrete mutual-information-ranked codes among residence_since, n_credits, n_guarantors. Coinciding feature choices are still retained as separately labelled planned runs.

The following tables report mean ± sample SD over three overlapping holdouts, including every saved predictive metric, elapsed fitting time and relevant convergence/coverage measures. Raw seed-level outcomes are in dimension_metrics.csv.

| feature_set | d | model | rules | consequent_parameters | fit_seconds | epochs_run | normalized_sum_min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 4 | original | 81.000000 ± 0.000000 | 405.000000 ± 0.000000 | 2.429680 ± 0.046881 | 500.000000 ± 0.000000 | 0.197939 ± 0.320699 |
| base | 4 | fixed_adam | 81.000000 ± 0.000000 | 405.000000 ± 0.000000 | 0.169139 ± 0.037329 | 63.000000 ± 12.124356 | 1.000000 ± 0.000000 |
| original_additions_5 | 5 | original | 243.000000 ± 0.000000 | 1458.000000 ± 0.000000 | 9.562574 ± 0.589404 | 500.000000 ± 0.000000 | 0.235909 ± 0.374560 |
| original_additions_5 | 5 | fixed_adam | 243.000000 ± 0.000000 | 1458.000000 ± 0.000000 | 0.712249 ± 0.021002 | 56.333333 ± 2.309401 | 1.000000 ± 0.000000 |
| original_additions_6 | 6 | original | 729.000000 ± 0.000000 | 5103.000000 ± 0.000000 | 32.969713 ± 0.607942 | 500.000000 ± 0.000000 | 0.262828 ± 0.386285 |
| original_additions_6 | 6 | fixed_adam | 729.000000 ± 0.000000 | 5103.000000 ± 0.000000 | 2.162076 ± 0.038824 | 54.666667 ± 1.154701 | 1.000000 ± 0.000000 |
| train_MI_5 | 5 | original | 243.000000 ± 0.000000 | 1458.000000 ± 0.000000 | 9.323722 ± 0.073593 | 500.000000 ± 0.000000 | 0.221986 ± 0.339275 |
| train_MI_5 | 5 | fixed_adam | 243.000000 ± 0.000000 | 1458.000000 ± 0.000000 | 0.726957 ± 0.012797 | 57.666667 ± 3.055050 | 1.000000 ± 0.000000 |
| train_MI_6 | 6 | original | 729.000000 ± 0.000000 | 5103.000000 ± 0.000000 | 33.643987 ± 0.607305 | 500.000000 ± 0.000000 | 0.270971 ± 0.379023 |
| train_MI_6 | 6 | fixed_adam | 729.000000 ± 0.000000 | 5103.000000 ± 0.000000 | 2.366910 ± 0.142738 | 54.666667 ± 1.154701 | 1.000000 ± 0.000000 |

| feature_set | d | model | accuracy | balanced_accuracy | precision | recall | f1 | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 4 | original | 0.713333 ± 0.020817 | 0.606349 ± 0.040987 | 0.528571 ± 0.054917 | 0.338889 ± 0.091793 | 0.410893 ± 0.086968 | 0.873810 ± 0.010911 |
| base | 4 | fixed_adam | 0.608333 ± 0.138954 | 0.607540 ± 0.041314 | 0.449882 ± 0.095426 | 0.605556 ± 0.383454 | 0.450630 ± 0.122256 | 0.609524 ± 0.356165 |
| original_additions_5 | 5 | original | 0.641667 ± 0.079739 | 0.551984 ± 0.044133 | 0.402031 ± 0.066363 | 0.327778 ± 0.340479 | 0.296277 ± 0.178806 | 0.776190 ± 0.257968 |
| original_additions_5 | 5 | fixed_adam | 0.705000 ± 0.008660 | 0.601984 ± 0.036815 | 0.520883 ± 0.029555 | 0.344444 ± 0.141748 | 0.399132 ± 0.107826 | 0.859524 ± 0.069375 |
| original_additions_6 | 6 | original | 0.520000 ± 0.201308 | 0.538095 ± 0.064955 | 0.338647 ± 0.041560 | 0.583333 ± 0.508538 | 0.332047 ± 0.260888 | 0.492857 ± 0.492857 |
| original_additions_6 | 6 | fixed_adam | 0.606667 ± 0.056199 | 0.604762 ± 0.030792 | 0.402099 ± 0.028017 | 0.600000 ± 0.189297 | 0.470202 ± 0.062318 | 0.609524 ± 0.154248 |
| train_MI_5 | 5 | original | 0.648333 ± 0.053463 | 0.604365 ± 0.054740 | 0.459660 ± 0.083870 | 0.494444 ± 0.310167 | 0.421087 ± 0.149936 | 0.714286 ± 0.206032 |
| train_MI_5 | 5 | fixed_adam | 0.583333 ± 0.065256 | 0.621429 ± 0.020927 | 0.398934 ± 0.031321 | 0.716667 ± 0.160728 | 0.505260 ± 0.030078 | 0.526190 ± 0.158006 |
| train_MI_6 | 6 | original | 0.515000 ± 0.203654 | 0.537698 ± 0.054351 | 0.445173 ± 0.194884 | 0.594444 ± 0.501756 | 0.339845 ± 0.239896 | 0.480952 ± 0.497152 |
| train_MI_6 | 6 | fixed_adam | 0.611667 ± 0.066583 | 0.601984 ± 0.031074 | 0.405704 ± 0.037423 | 0.577778 ± 0.178211 | 0.465511 ± 0.058006 | 0.626190 ± 0.162934 |

| feature_set | d | model | roc_auc | average_precision | brier | positive_rate |
| --- | --- | --- | --- | --- | --- | --- |
| base | 4 | original | 0.691508 ± 0.017588 | 0.494198 ± 0.039490 | 0.247709 ± 0.000481 | 0.190000 ± 0.034641 |
| base | 4 | fixed_adam | 0.703056 ± 0.017337 | 0.517130 ± 0.055279 | 0.236754 ± 0.007912 | 0.455000 ± 0.362595 |
| original_additions_5 | 5 | original | 0.652302 ± 0.032982 | 0.454486 ± 0.065001 | 0.248744 ± 0.001066 | 0.255000 ± 0.282356 |
| original_additions_5 | 5 | fixed_adam | 0.677976 ± 0.020336 | 0.498074 ± 0.025265 | 0.240971 ± 0.002442 | 0.201667 ± 0.090875 |
| original_additions_6 | 6 | original | 0.557619 ± 0.109093 | 0.384591 ± 0.089441 | 0.249592 ± 0.000829 | 0.530000 ± 0.494039 |
| original_additions_6 | 6 | fixed_adam | 0.663413 ± 0.035387 | 0.484535 ± 0.059523 | 0.242612 ± 0.003367 | 0.453333 ± 0.163121 |
| train_MI_5 | 5 | original | 0.660198 ± 0.038181 | 0.459663 ± 0.067670 | 0.248859 ± 0.000919 | 0.348333 ± 0.236766 |
| train_MI_5 | 5 | fixed_adam | 0.702698 ± 0.036172 | 0.511916 ± 0.078143 | 0.237895 ± 0.005727 | 0.546667 ± 0.157665 |
| train_MI_6 | 6 | original | 0.605278 ± 0.026428 | 0.422164 ± 0.041989 | 0.249556 ± 0.000066 | 0.541667 ± 0.496043 |
| train_MI_6 | 6 | fixed_adam | 0.668889 ± 0.033966 | 0.492339 ± 0.056788 | 0.242652 ± 0.002991 | 0.435000 ± 0.165227 |

| feature_set | d | model | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- |
| base | 4 | original | 122.333333 ± 1.527525 | 17.666667 ± 1.527525 | 39.666667 ± 5.507571 | 20.333333 ± 5.507571 |
| base | 4 | fixed_adam | 85.333333 ± 49.863146 | 54.666667 ± 49.863146 | 23.666667 ± 23.007245 | 36.333333 ± 23.007245 |
| original_additions_5 | 5 | original | 108.666667 ± 36.115555 | 31.333333 ± 36.115555 | 40.333333 ± 20.428738 | 19.666667 ± 20.428738 |
| original_additions_5 | 5 | fixed_adam | 120.333333 ± 9.712535 | 19.666667 ± 9.712535 | 39.333333 ± 8.504901 | 20.666667 ± 8.504901 |
| original_additions_6 | 6 | original | 69.000000 ± 69.000000 | 71.000000 ± 69.000000 | 25.000000 ± 30.512293 | 35.000000 ± 30.512293 |
| original_additions_6 | 6 | fixed_adam | 85.333333 ± 21.594752 | 54.666667 ± 21.594752 | 24.000000 ± 11.357817 | 36.000000 ± 11.357817 |
| train_MI_5 | 5 | original | 100.000000 ± 28.844410 | 40.000000 ± 28.844410 | 30.333333 ± 18.610033 | 29.666667 ± 18.610033 |
| train_MI_5 | 5 | fixed_adam | 73.666667 ± 22.120880 | 66.333333 ± 22.120880 | 17.000000 ± 9.643651 | 43.000000 ± 9.643651 |
| train_MI_6 | 6 | original | 67.333333 ± 69.601245 | 72.666667 ± 69.601245 | 24.333333 ± 30.105371 | 35.666667 ± 30.105371 |
| train_MI_6 | 6 | fixed_adam | 87.666667 ± 22.810816 | 52.333333 ± 22.810816 | 25.333333 ± 10.692677 | 34.666667 ± 10.692677 |

Convergence was independently replayed for all 30 fits with the same initialization and data, using cached/vectorized mathematically equivalent original GD updates and the same fixed Adam implementation. Every saved dimension metric and validation-selected threshold agreed within 1e-10. Replay timings are **not** used as original-implementation benchmark timings. Replays add full trajectories and model states under dimension_replay/.

| feature_set | d | model | best_epoch | selected_training_bce | selected_validation_bce | selected_consequent_gradient_l2 | coefficient_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 4 | original | 500.000000 ± 0.000000 | 0.685901 ± 0.001354 | 0.691517 ± 0.000313 | 0.024560 ± 0.002625 | 0.993575 ± 0.019008 |
| base | 4 | fixed_adam | 13.000000 ± 12.124356 | 0.665136 ± 0.011524 | 0.680726 ± 0.014311 | 0.016017 ± 0.009354 | 2.233735 ± 1.283396 |
| original_additions_5 | 5 | original | 500.000000 ± 0.000000 | 0.689528 ± 0.001158 | 0.691587 ± 0.000886 | 0.017055 ± 0.001402 | 1.899443 ± 0.020763 |
| original_additions_5 | 5 | fixed_adam | 6.333333 ± 2.309401 | 0.669773 ± 0.005208 | 0.685006 ± 0.004797 | 0.011771 ± 0.003380 | 2.733764 ± 0.528797 |
| original_additions_6 | 6 | original | 500.000000 ± 0.000000 | 0.691858 ± 0.000608 | 0.692569 ± 0.001366 | 0.012403 ± 0.001190 | 3.554192 ± 0.015315 |
| original_additions_6 | 6 | fixed_adam | 4.666667 ± 1.154701 | 0.671871 ± 0.001996 | 0.686733 ± 0.004783 | 0.008988 ± 0.001787 | 3.688159 ± 0.151809 |
| train_MI_5 | 5 | original | 500.000000 ± 0.000000 | 0.689439 ± 0.001066 | 0.692402 ± 0.000313 | 0.017921 ± 0.002069 | 1.898984 ± 0.019581 |
| train_MI_5 | 5 | fixed_adam | 7.666667 ± 3.055050 | 0.664579 ± 0.004538 | 0.681207 ± 0.007690 | 0.011206 ± 0.004327 | 3.006531 ± 0.577925 |
| train_MI_6 | 6 | original | 500.000000 ± 0.000000 | 0.691719 ± 0.000390 | 0.692648 ± 0.001055 | 0.012009 ± 0.000745 | 3.554738 ± 0.013905 |
| train_MI_6 | 6 | fixed_adam | 4.666667 ± 1.154701 | 0.672043 ± 0.002245 | 0.686774 ± 0.004075 | 0.008561 ± 0.001288 | 3.702820 ± 0.175544 |

Original has a fixed 500th-epoch endpoint, not a validation-selected optimum. Adam restores a validation-selected epoch and may stop later after patience. BCE columns above are unpenalized weighted BCE at the restored state; Adam stopping used BCE plus its coefficient penalty. Gradient values are consequent-only gradients at that state (including the penalty for Adam), so they are not directly identical convergence criteria across models. No claim of reaching a global optimum is made.

Fitting-only feature ranks and exact code values:

| seed | feature | mutual_information | rank |
| --- | --- | --- | --- |
| 42 | n_credits | 0.001166 | 1 |
| 42 | n_guarantors | 0.000262 | 2 |
| 42 | residence_since | 0.000237 | 3 |
| 7 | n_credits | 0.003521 | 1 |
| 7 | residence_since | 0.001553 | 2 |
| 7 | n_guarantors | 0.000038 | 3 |
| 19 | n_credits | 0.002698 | 1 |
| 19 | residence_since | 0.002628 | 2 |
| 19 | n_guarantors | 0.000928 | 3 |
| seed | feature_set | features |
| --- | --- | --- |
| 42 | base | month_duration;credit_amount;payment_to_income_ratio;age |
| 42 | original_additions_5 | month_duration;credit_amount;payment_to_income_ratio;age;residence_since |
| 42 | original_additions_6 | month_duration;credit_amount;payment_to_income_ratio;age;residence_since;n_credits |
| 42 | train_MI_5 | month_duration;credit_amount;payment_to_income_ratio;age;n_credits |
| 42 | train_MI_6 | month_duration;credit_amount;payment_to_income_ratio;age;n_credits;n_guarantors |
| 7 | base | month_duration;credit_amount;payment_to_income_ratio;age |
| 7 | original_additions_5 | month_duration;credit_amount;payment_to_income_ratio;age;residence_since |
| 7 | original_additions_6 | month_duration;credit_amount;payment_to_income_ratio;age;residence_since;n_credits |
| 7 | train_MI_5 | month_duration;credit_amount;payment_to_income_ratio;age;n_credits |
| 7 | train_MI_6 | month_duration;credit_amount;payment_to_income_ratio;age;n_credits;residence_since |
| 19 | base | month_duration;credit_amount;payment_to_income_ratio;age |
| 19 | original_additions_5 | month_duration;credit_amount;payment_to_income_ratio;age;residence_since |
| 19 | original_additions_6 | month_duration;credit_amount;payment_to_income_ratio;age;residence_since;n_credits |
| 19 | train_MI_5 | month_duration;credit_amount;payment_to_income_ratio;age;n_credits |
| 19 | train_MI_6 | month_duration;credit_amount;payment_to_income_ratio;age;n_credits;residence_since |

### What the experiments actually demonstrate

Adding inputs deterministically multiplies the Cartesian rule count: 81→243→729. Consequent coefficients grow 405→1458→5103, a factor of 12.6 from four to six. This mathematical statement is separate from observed runtime and predictive quality. The original nested Python loops and larger dense arrays drive substantial runtime growth; vectorization/caching and stopping also contribute to the new implementation's time advantage. It would be incorrect to attribute the speed difference exclusively to Adam.

Predictive changes are the joint result of new information, coded feature semantics, rule count, initialization and fitting. A 4/5/6-input comparison alone does not isolate a causal effect of dimension on generalization. Train-only ranking provides a controlled sensitivity to the particular added codes, but uses only three seeds and a restricted candidate pool, not exhaustive feature selection. It does not justify choosing the best test subset.

The original epsilon normalization can make a_r sum below one. Therefore 1/sum(a_r²) can exceed the nominal number of rules: it is not a valid effective-rule count in affected original observations. Do not interpret that original diagnostic as genuine rule diversity. New exact-normalized activation concentration is meaningful as a relative concentration measure. Coincident Gaussian centres on coded inputs can create semantically duplicate sets even when the nominal Cartesian count is large.

Equal code spacing for installment burden, residence and number of credits is assumed by Gaussian/linear modeling, not validated. n_guarantors has an uncertain supplied semantic label; supplementary ranking treats it as a discrete code only. No categorical-encoding ablation, compact-rule implementation or reproduction of Cao et al. was completed. These remain limitations rather than invented results.

## 3. RQ3: real held-out explanations

Explained model: seed 42 weighted **fixed_adam**, using its validation-selected threshold. Before fitting, case criteria specified the lowest source-row ID among correctly predicted bad cases and the lowest ID among incorrectly predicted good cases. No case was chosen for visually appealing rules or large contributions. Both selected observations belong to the untouched test partition.

For standardized x, Gaussian grade mu_jk=exp(-0.5((x_j-c_jk)/sigma_jk)²), raw firing w_r=product_j(mu_j,r_j), normalized a_r=w_r/sum(w), first-order output f_r=beta_r·[x,1], contribution c_r=a_r f_r, and logit z=sum(c_r). Score=sigmoid(z), with a separate selected score threshold. Log-domain calculation implements exact relative normalization. Rule contributions are in logit units, and membership grades express set compatibility; neither is an outcome probability. All slopes/biases are in cases/explained_model_consequents.csv.

### Source row 1: correct bad-credit classification

Zero-based source row **1**, CSV line **3** including header. Source SHA256 is recorded in data_audit.json. True class=1, predicted class=1; good=0,bad=1. Original inputs in order (duration,amount,burden-code,age) are [48.0, 5951.0, 2.0, 22.0]. Standardized inputs are [2.240084374334162, 0.9855785906114148, -0.8573354355049446, -1.2226403295668138].

Score **0.575667125642**; selected threshold **0.470000000000**; direct logit **0.305011358523764**. All 12 input memberships:

| feature | set | original_value | scaled_value | center | width | grade |
| --- | --- | --- | --- | --- | --- | --- |
| month_duration | LOW | 48.000000 | 2.240084 | -0.743587 | 0.750000 | 0.000366 |
| month_duration | MEDIUM | 48.000000 | 2.240084 | -0.246308 | 0.750000 | 0.004106 |
| month_duration | HIGH | 48.000000 | 2.240084 | 0.250970 | 0.750000 | 0.029690 |
| credit_amount | LOW | 5951.000000 | 0.985579 | -0.689674 | 0.750000 | 0.082526 |
| credit_amount | MEDIUM | 5951.000000 | 0.985579 | -0.341362 | 0.750000 | 0.209061 |
| credit_amount | HIGH | 5951.000000 | 0.985579 | 0.265453 | 0.750000 | 0.630677 |
| payment_to_income_ratio | LOW | 2.000000 | -0.857335 | -0.857335 | 0.750000 | 1.000000 |
| payment_to_income_ratio | MEDIUM | 2.000000 | -0.857335 | 0.034851 | 0.750000 | 0.492849 |
| payment_to_income_ratio | HIGH | 2.000000 | -0.857335 | 0.927038 | 0.750000 | 0.059000 |
| age | LOW | 22.000000 | -1.222640 | -0.772157 | 0.750000 | 0.834947 |
| age | MEDIUM | 22.000000 | -1.222640 | -0.231577 | 0.750000 | 0.417666 |
| age | HIGH | 22.000000 | -1.222640 | 0.579294 | 0.750000 | 0.055788 |

Eight most influential rules by absolute signed contribution (zero-based IDs):

| rule_id | antecedent | firing | normalized_firing | consequent | contribution |
| --- | --- | --- | --- | --- | --- |
| 72 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=LOW AND age=LOW | 0.015634 | 0.244391 | 0.386425 | 0.094439 |
| 73 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=LOW AND age=MEDIUM | 0.007821 | 0.122252 | 0.523445 | 0.063992 |
| 75 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=MEDIUM AND age=LOW | 0.007705 | 0.120448 | 0.487659 | 0.058737 |
| 76 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=MEDIUM AND age=MEDIUM | 0.003854 | 0.060252 | 0.412144 | 0.024832 |
| 66 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=MEDIUM AND age=LOW | 0.002554 | 0.039927 | 0.561765 | 0.022429 |
| 45 | month_duration=MEDIUM AND credit_amount=HIGH AND payment_to_income_ratio=LOW AND age=LOW | 0.002162 | 0.033801 | 0.280053 | 0.009466 |
| 48 | month_duration=MEDIUM AND credit_amount=HIGH AND payment_to_income_ratio=MEDIUM AND age=LOW | 0.001066 | 0.016659 | 0.436476 | 0.007271 |
| 64 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=LOW AND age=MEDIUM | 0.002592 | 0.040525 | 0.100739 | 0.004082 |

The most positive term is 0.094438736953 (rule 72); the most negative is -0.002804391154 (rule 46). Positive terms sum to **0.311580480614867**; negative terms sum to **-0.006569122091103**. All 81 terms sum to **0.305011358523764**, with stored in-memory reconstruction error **0.000e+00**. sigmoid(z) gives the score above. Exported CSV reconstruction and independent checkpoint recomputation are tested within 1e-12.

Full rule terms: [cases/case_row_1_rules.csv](results/research/cases/case_row_1_rules.csv). Membership file: [cases/case_row_1_memberships.csv](results/research/cases/case_row_1_memberships.csv). Figure: [case_row_1.png](results/research/plots/case_row_1.png).

### Source row 6: incorrect good-credit classification

Zero-based source row **6**, CSV line **8** including header. Source SHA256 is recorded in data_audit.json. True class=0, predicted class=1; good=0,bad=1. Original inputs in order (duration,amount,burden-code,age) are [24.0, 2835.0, 3.0, 53.0]. Standardized inputs are [0.2509702576864149, -0.1497168598093262, 0.034851033963615635, 1.5703572684303753].

Score **0.489683527201**; selected threshold **0.470000000000**; direct logit **-0.041271748577546**. All 12 input memberships:

| feature | set | original_value | scaled_value | center | width | grade |
| --- | --- | --- | --- | --- | --- | --- |
| month_duration | LOW | 24.000000 | 0.250970 | -0.743587 | 0.750000 | 0.415099 |
| month_duration | MEDIUM | 24.000000 | 0.250970 | -0.246308 | 0.750000 | 0.802672 |
| month_duration | HIGH | 24.000000 | 0.250970 | 0.250970 | 0.750000 | 1.000000 |
| credit_amount | LOW | 2835.000000 | -0.149717 | -0.689674 | 0.750000 | 0.771700 |
| credit_amount | MEDIUM | 2835.000000 | -0.149717 | -0.341362 | 0.750000 | 0.967880 |
| credit_amount | HIGH | 2835.000000 | -0.149717 | 0.265453 | 0.750000 | 0.857946 |
| payment_to_income_ratio | LOW | 3.000000 | 0.034851 | -0.857335 | 0.750000 | 0.492849 |
| payment_to_income_ratio | MEDIUM | 3.000000 | 0.034851 | 0.034851 | 0.750000 | 1.000000 |
| payment_to_income_ratio | HIGH | 3.000000 | 0.034851 | 0.927038 | 0.750000 | 0.492849 |
| age | LOW | 53.000000 | 1.570357 | -0.772157 | 0.750000 | 0.007615 |
| age | MEDIUM | 53.000000 | 1.570357 | -0.231577 | 0.750000 | 0.055788 |
| age | HIGH | 53.000000 | 1.570357 | 0.579294 | 0.750000 | 0.417666 |

Eight most influential rules by absolute signed contribution (zero-based IDs):

| rule_id | antecedent | firing | normalized_firing | consequent | contribution |
| --- | --- | --- | --- | --- | --- |
| 68 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.404251 | 0.073460 | 0.241958 | 0.017774 |
| 59 | month_duration=HIGH AND credit_amount=LOW AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.322313 | 0.058571 | -0.258010 | -0.015112 |
| 41 | month_duration=MEDIUM AND credit_amount=MEDIUM AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.324481 | 0.058965 | -0.226816 | -0.013374 |
| 65 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=LOW AND age=HIGH | 0.199235 | 0.036205 | -0.237755 | -0.008608 |
| 32 | month_duration=MEDIUM AND credit_amount=LOW AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.258711 | 0.047013 | -0.174464 | -0.008202 |
| 74 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=LOW AND age=HIGH | 0.176605 | 0.032093 | 0.243573 | 0.007817 |
| 44 | month_duration=MEDIUM AND credit_amount=MEDIUM AND payment_to_income_ratio=HIGH AND age=HIGH | 0.159920 | 0.029061 | -0.234035 | -0.006801 |
| 77 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.358335 | 0.065117 | 0.099786 | 0.006498 |

The most positive term is 0.017774356037 (rule 68); the most negative is -0.015111808016 (rule 59). Positive terms sum to **0.053225152483263**; negative terms sum to **-0.094496901060809**. All 81 terms sum to **-0.041271748577546**, with stored in-memory reconstruction error **0.000e+00**. sigmoid(z) gives the score above. Exported CSV reconstruction and independent checkpoint recomputation are tested within 1e-12.

Full rule terms: [cases/case_row_6_rules.csv](results/research/cases/case_row_6_rules.csv). Membership file: [cases/case_row_6_memberships.csv](results/research/cases/case_row_6_memberships.csv). Figure: [case_row_6.png](results/research/plots/case_row_6.png).

### Meaning of the explanations

Row 1 is associated with a positive net logit and correctly classified bad credit. Its strongest displayed terms are local linear outputs gated by overlapping duration/amount/burden/age Gaussians. That states how the fitted model computes the score; it is not evidence that these attributes caused an actual default. Row 6 has a negative net logit but is still predicted bad because its score .489684 exceeds the validation threshold .47. That case exposes an understandable threshold-policy error and contribution cancellation that would be missed by displaying rule antecedents alone.

**Structural transparency** means the rule inventory, learned coefficients and arithmetic are accessible. **Faithful local explanation** is supported here because actual held-out logits and scores can be reconstructed exactly from all rules, with raw/scaled inputs, memberships, firing strengths, linear outputs and signed contributions exposed. **Meaningful human explanation** requires further evidence: whether a reader understands these terms, whether linguistic labels are appropriate, whether a small number of rules adequately summarizes the decision and whether explanations are stable. We have no user study, expert evaluation or cross-fit explanation-stability test. LOW/MEDIUM/HIGH are quantile labels rather than externally validated credit concepts; learned premises can also cross in the trainable variant. Top terms do not alone reconstruct the full decision, and contribution signs are relative to zero logit rather than a causal or counterfactual baseline.

RQ3 is therefore answered positively for verifiable local numerical decomposition, with clear limits for semantic and human interpretability. Fuzzy structure does not guarantee a meaningful explanation merely because rules can be printed.

## 4. Diagnosis of original ANFIS performance

### Verified observations

1. **Limited optimization progress at the original budget.** In the seed-42 weighted replay, training BCE moved from 0.69330282 to 0.68490366 over 500 updates; validation BCE moved from 0.69276741 to 0.69161727. The terminal gradient norm remained 0.02689078. Cached replay coefficients matched the original within 2.776e-17. Nonstationarity and slow loss change are observed; no global optimum was certified.

2. **Normalization attenuation exists.** With original coefficients and premises fixed, exact log-domain normalization changed a seed-42 test logit by up to 0.319982201. Original a_r values can have total below one when raw strengths are small, shrinking logits toward zero/score .5. This demonstrates a numerical mechanism, but not that it explains every classification error.

| normalization | threshold_mode | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| original_epsilon | default | 0.646429 | 0.650000 | 0.523490 | 0.690357 | 0.471485 |
| original_epsilon | original_validation | 0.635714 | 0.400000 | 0.470588 | 0.690357 | 0.471485 |
| exact_log_domain | default | 0.646429 | 0.650000 | 0.523490 | 0.695357 | 0.474269 |
| exact_log_domain | original_validation | 0.644048 | 0.416667 | 0.485437 | 0.695357 | 0.474269 |

This intervention changes inference only and locks .5 or the original validation-selected threshold; it is a diagnostic comparison, not a newly tuned model. Per-row firing sums/logits are in normalization_only_predictions.csv.

3. **Threshold and imbalance policy materially change recall.** Weighted original mean recall is .700000 at .5 and .380000 under validation-selected thresholds, although score-ranking metrics are unchanged. Its selected-threshold accuracy .688000 is accompanied by lower bad recall. Near-single-class prediction counts and per-split matrices are retained in ANFIS_DIAGNOSTICS.md and per_split_metrics.csv. Unadjusted models at .5 typically miss most bad observations; all-negative majority accuracy is already .7.

4. **Grid size dilutes and complicates the parameterization.** Fixed normalized weights multiply each rule's consequent gradient. Many overlapping rules share credit for each observation. Larger grids increase the number of parameters and original implementation runtime. Absolute coverage and relative concentration are measured, not assumed; diagnostics.json contains every selected ANFIS fit's centres, widths, low-coverage fractions, coefficient norms, normalized-sum minima and rule-weight distribution.

5. **Original comparison was procedurally asymmetric.** ANFIS had weighting and fixed epochs; MLP had oversampling and internal early stopping. Duplicates could straddle MLP's internal stopping split. New external stopping and matched conditions resolve that specific comparison flaw, but they do not retroactively validate the original comparison as fair.

### Controlled learning-rate ablation and modifications

Same split, initialization, fixed premises, weights and 500 GD steps; learning rate only changes, evaluated on validation data:

| lr | train_loss_last_recorded | validation_bce | validation_balanced_accuracy | validation_recall | validation_f1 |
| --- | --- | --- | --- | --- | --- |
| 0.020000 | 0.684918 | 0.691617 | 0.538690 | 0.229167 | 0.289474 |
| 0.200000 | 0.656212 | 0.690892 | 0.586310 | 0.895833 | 0.500000 |
| 2.000000 | 0.622260 | 0.690748 | 0.584821 | 0.875000 | 0.497041 |

Larger GD steps reduced fitting loss and modestly altered validation ranking/decision behaviour; this supports investigating optimization budget rather than attributing failure solely to fuzzy modeling. It does not show that the largest step size is generally best. No main-study model was replaced after this diagnostic.

**What helped in measured outcomes:** stable normalization removes attenuation and full-batch vectorized fitting reduces implementation overhead; new fixed Adam improves mean AUC/AP over original in all three matched conditions. The complete table shows whether decision metrics improve for the chosen threshold policy. Weighting and oversampling increase bad recall at .5 relative to unadjusted fitting. External MLP stopping avoids duplicate overlap in its stopping split. Reproducibility, transparency and test coverage improve independently of predictive scores.

**What did not reliably help:** premise learning did not improve weighted mean BA/AUC/AP over fixed Adam; validation-selected thresholds did not consistently beat .5 on held-out decision metrics; adding coded inputs did not establish monotonic predictive benefit. These negative/mixed results are retained, not filtered out.

**Plausible mechanisms, not demonstrated exclusive causes:** insufficient step size/epochs with distributed rule gradients; unadaptable fixed premises; high-variance threshold selection on 48 bad validation cases; redundant coded-feature quantiles; regularization and overfitting effects as grid size grows. Adam changes optimizer, L2, normalization and stopping simultaneously, so its performance difference cannot be assigned to one component alone. A full factorial ablation was not run. The original main globally suppresses warnings, including when imported by the extension; absence of printed warnings is not a convergence certificate. Numerical trajectory and consistency checks provide the actual evidence.

The unavailable historical ANFIS means cannot be diagnosed as though they were generated by this exact checkout. Our diagnosis applies to the executable original architecture and recorded new evaluations.

## 5. Academic conclusions and contribution

**RQ1:** On this restricted four-input benchmark, the relative performance of ANFIS and MLP depends on imbalance treatment, threshold policy and metric. Fixed Adam is competitive in ranking metrics, while the small MLP and logistic reference are competitive or better for several decision metrics. Five overlapping holdouts and conditional uncertainty estimates do not establish universal superiority. Accuracy must be interpreted against the .7 majority baseline, and recall/F1 against degenerate all-positive behaviour.

**RQ2:** Full-grid rule and coefficient growth is verified mathematically and in code; measured 4/5/6-input fitting costs grow strongly in the original implementation. Predictive consequences are heterogeneous and confounded with added information, encoding and fitting. The feature-ranking sensitivity controls selection to fitting data but does not isolate dimension as a causal variable or justify ordinal spacing.

**RQ3:** The simplified first-order model provides faithful rule-level decompositions of two predetermined real held-out observations, including an error. Exact reconstruction establishes computational fidelity and exposes threshold effects. It does not establish causal validity, cognitive simplicity or semantic adequacy of all rules.

The most defensible scientific contribution is an **auditable empirical connection between optimization, imbalance/threshold policy, grid complexity and faithful individual explanations in the same preserved credit-risk experiment**. This contribution is a carefully controlled diagnostic case study and reproducible evidence package, not a novel ANFIS architecture or a claim of state-of-the-art credit scoring.

Strengths: exact baseline recovery; matched partitions and fitting-only transformations; retained negative outcomes and rejected trials; logistic/dummy references; explicit timing/search budgets; full test scores and model states; analytic-gradient verification; real-case reconstruction tests; separation of mathematical growth, compute costs and predictive effects.

Weaknesses: one historical dataset, four base predictors, small validation/test minority counts, overlapping holdouts, limited two-trial searches, different stopping criteria, project-level adaptive dataset reuse, no external validation, uncertain coded semantics, uncalibrated scores, no human interpretability assessment and incomplete full-paper review. The original dataset provenance/export mapping, report and official exam instructions remain missing. No deployment, fairness, causal or grade claim is justified.

Verified literature metadata and cautious connections are in [LIMITATIONS_AND_DISCUSSION.md](results/research/LIMITATIONS_AND_DISCUSSION.md): Jang (1993), doi:10.1109/21.256541; Cao et al. (2025), doi:10.1109/TFUZZ.2025.3572937; Zhang et al. (2024), doi:10.1109/TFUZZ.2024.3442821; Lakra et al. (2025), doi:10.1109/ACCESS.2025.3585980. None of their methods is claimed as reproduced. Bibliographic metadata/repository abstracts were checked; full texts were not supplied. Links and source-supported descriptions are in that document.

## 6. Exact report-update instructions and assessment gaps

The existing report file is absent, so section numbers, original wording and exact text-level edits cannot be identified honestly. Apply the following changes to the corresponding named sections when the source is available. Do not silently replace earlier numbers.

| Report section | Required change | Evidence/table/figure to include |
| --- | --- | --- |
| Title/authors/front matter | Retain project identity; insert verified authors and OsloMet email addresses; use official required IEEE conference template. | Identities/template not provided; verification remains outstanding. |
| Abstract | Summarize preserved simplified architecture, bounded matched-split extension, conditional findings and verified two-case decomposition. Remove superiority or significance claims unsupported here. | Use exact RQ1/RQ2/RQ3 numbers from this document; distinguish single-split reproduction. |
| Introduction and RQs | Retain exact original questions if appropriate and reconcile them against working RQs; frame imbalance, grid complexity and local faithfulness as separate objectives. | Exact original RQ wording remains unavailable. |
| Related work | Correctly distinguish simplified frozen-premise TSK from Jang's hybrid ANFIS; discuss Cao's compact design concern and Zhang's ensemble imbalance work without claiming implementation. | Verified citations and bounded connections in LIMITATIONS_AND_DISCUSSION.md; read full papers before expanding claims. |
| Dataset/preprocessing | Give source hash, target good=0/bad=1, 700/300 counts, feature names, 640/160/200 splits and fitting-only scaling. Discuss coded ordinal spacing/provenance. | data_audit.json, split_assignments.csv, RESEARCH_AUDIT.md. |
| Model/optimization methods | State fixed/trainable parameters, quantile Gaussian initialization, original epsilon normalization, exact new normalization, logit/sigmoid distinction, loss, L2, learning rates, epochs and stopping. | EXPERIMENTAL_DESIGN.md, config.json, main.py and research_models.py. |
| Experimental design | Separate original asymmetric experiment from matched unadjusted/weighted/oversampled study. Document two trials/model, different epoch criteria, validation thresholds, all seeds and no fitting+validation refit. | all_trials.csv; both threshold tables in this document. |
| Original results/reproduction | Retain original single-split numbers and exact matching rerun. Mark unavailable repeated summaries and previous dimension outcomes as unverified until their artifacts are found. | reproduction_check.csv, original_reproduction/execution.log, original_snapshot/. |
| RQ1 results | Include full default/selected threshold comparison, meaningful minority metrics, paired descriptive differences and uncertainty caveats. | final_complete_comparison.csv, per_split_metrics.csv, paired_summary.csv, conditional_bootstrap.csv; plots/model_comparison.png. |
| ANFIS failure analysis | Show loss/gradient replay, validation-only LR ablation, inference-only normalization intervention, score distributions and threshold sensitivity. Separate observed mechanisms from causal hypotheses. | original_gradient_trajectory.csv, learning_rate_validation_ablation.csv, normalization_only_metrics.csv; plots/original_optimization_and_memberships.png and plots/diagnostics_seed42_weighted.png. |
| RQ2 results | Add exact rule/parameter counts, all 4/5/6-input times and metrics, restored epochs/gradient/BCE diagnostics, train-only feature sensitivity and ordinal limitations. | dimension_metrics.csv, dimension_convergence.csv, train_only_feature_ranking.csv; plots/dimension_runtime_performance.png. |
| RQ3 results | Replace rule inventory alone with both real cases, true/predicted labels, raw/scaled inputs, grades, firings, consequent outputs, signed terms and exact reconstruction. | cases/case_summaries.json, both complete rule/membership CSVs, explained_model_consequents.csv; plots/case_row_1.png and plots/case_row_6.png. |
| Discussion | Discuss small/no consistent gains, original optimizer/normalization weaknesses, thresholds, overlapping holdouts, code semantics, uncalibrated scores and transparency versus comprehension. | This document sections 4–5 and LIMITATIONS_AND_DISCUSSION.md. |
| Conclusion | Answer each RQ with conditional verified findings, not intrinsic ANFIS superiority or a promised grade. | Exact tables above; suggested prose in REPORT_READY_RESULTS.md. |
| Reproducibility appendix | Attach or link all source, pinned requirements, data/hash, README commands and artifact access as permitted by official brief. Exclude environment/cache folders. | README.md, source files, requirements-research.lock.txt, FILE_MANIFEST.csv and tests.log. |

### Status of preliminary criticisms

| Preliminary concern | What is now resolved in the evidence package | What remains unresolved |
| --- | --- | --- |
| Reproducibility / code / README | Exact baseline rerun; preserved original source/results; pinned environment; saved models/splits/scores; executable commands, tests and hash manifest. | Fresh independent-machine reproduction; data export provenance; actual appendix attachment and official requirements check. |
| RQ3 lacks real examples | Two predefined held-out cases with complete numerical rule decomposition, error case, raw/scaled inputs, membership/firing/consequent tables and tests. | Human/expert comprehension and cross-fit explanation stability. |
| Poor/unstable ANFIS | Loss/gradient, initialization, normalization, threshold and rule-coverage diagnostics; controlled improvements and retained mixed results. | No consistently superior ANFIS; limited search and no complete factorial causal attribution. |
| Unfair optimization | Matched data/conditions; logistic/dummy baselines; explicit two-trial budgets; external MLP stopping. | Different epoch stopping objectives/compute costs; small validation search; no broad nested protocol. |
| Uncontrolled feature/dimension choices | Three paired seeds at 4/5/6 inputs; both requested and fitting-only MI additions; actual counts, times and convergence replay. | Restricted candidate pool, coded spacing validity, no one-hot encoding control, no compact-rule alternative. |
| Descriptive statistics only | Paired differences, SD/median/range, conditional paired bootstrap and explicit dependence limitations. | No population-level superiority inference; five overlapping holdouts are not independent replicates. |
| Paper analysis / IEEE compliance | Traceable tables, figures, drafting paragraphs and this integration guide are supplied. | Report itself not edited; word count, official IEEE template, author emails, citation integration and final appendix not checked. |

The stated 7,000–9,000 words, IEEE conference template, OsloMet emails, code+README appendix and marking weights originate from the user-provided requirements; the official brief was not present. This summary is not claimed to satisfy the final report's word count or formatting. The actual document must be checked against the authoritative brief.

## 7. Reproducibility, completed experiments, files and tests

### Completed experiment ledger

| Experiment | Actual completed execution and retained evidence |
| --- | --- |
| Unchanged full program | One seed-42 ANFIS+MLP run with original EDA/evaluation; exact saved-table agreement; original_reproduction/. |
| Main matched comparison | Five seeds × three conditions × six models; 150 candidate evaluations (135 fitted candidates plus 15 dummy references), 90 selected bundles, 180 threshold-specific metric rows, 18,000 test-score rows. All trials and rejected trajectories retained. |
| Real-case explanation | Two selected weighted fixed-Adam test observations, 81 rule terms and 12 membership grades per case, full consequent coefficients, exact logit/sigmoid checks. |
| Original optimization replay | Seed-42 weighted original GD epoch 0–500; gradients and fit/validation BCE; coefficient agreement within 2.78e-17. |
| Learning-rate diagnostic | Three original GD fits at .02,.2,2., identical initialization/split/500 steps; validation-only outcomes. |
| Normalization-only diagnostic | Original model's two inference normalizations, same premises/coefficients and locked thresholds; no refitting or test tuning. |
| Initialization sensitivity | Five original weighted GD initializations with split seed 42 fixed; default and validation thresholds. |
| Dimension/feature sensitivity | All 30 planned original/fixed-Adam fits over 4/5/6 inputs and three split seeds; final per-fit metrics/times saved. |
| Dimension convergence replays | 30 deterministic replays, saved histories and models; all saved dimension metrics and thresholds checked within 1e-10; replay times excluded from original timing claims. |
| Conditional uncertainty | 2,000 stratified paired test-sample bootstrap resamples, fixed Adam minus MLP, seed42 weighted validation thresholds; all samples retained. |

Main-study and dimension results are new experiments, not the missing historical repeated runs. Repeated planned feature specifications may coincide after MI ranking; those labelled evaluations are retained rather than pretending to be distinct information sets.

### Tests and final verification

Final recorded unittest outcome: **6 tests, 0 failures, 0 errors, 0 skips**; successful=True. See tests.log and test_outcome.json.

1. `test_premise_and_consequent_gradients`: central finite differences for every slope/bias, Gaussian centre and log width on a small weighted problem.
2. `test_stable_inference_and_original_agreement`: ordinary-region original/stable agreement, finite extreme inputs and unit-sum stable weights.
3. `test_paired_split_and_resampling_integrity`: partition sizes/disjointness, stratified class counts and fitting-only resampling provenance.
4. `test_metrics_and_threshold`: known confusion matrix/metric values and deterministic threshold tie-breaking.
5. `test_case_reconstruction_against_checkpoint`: both real held-out rows, original data/scaler, exported rule weights/outputs/contributions, reconstructed logit and sigmoid score.
6. `test_saved_metrics_from_predictions`: all 180 main-study metric rows recomputed from saved test scores.

Delivery verification also reloaded **90** selected checkpoint bundles, regenerated every test score, checked source-row partitions and fitting-only scaler means, verified original source/results byte preservation, and checked all 30 dimension rule/parameter counts. Maximum score difference was **1.110e-16**; maximum scaler-mean difference was **0.000e+00**. Convergence replay adds independent dimension metric agreement. Structural/hash checks are evidence of consistency, not proof against all scientific design errors.

### Reproduction commands

Run sequentially from the repository root with Python 3.11.13 (PowerShell; a working Python 3.11 installation is required):

```powershell
py -3.11 -m venv .research-venv
.\.research-venv\Scripts\python.exe -m pip install -r requirements-research.lock.txt
.\.research-venv\Scripts\python.exe reproduce_original.py
.\.research-venv\Scripts\python.exe research_pipeline.py
.\.research-venv\Scripts\python.exe research_telemetry.py
.\.research-venv\Scripts\python.exe dimension_convergence.py
.\.research-venv\Scripts\python.exe build_research_reports.py
.\.research-venv\Scripts\python.exe verify_delivery.py
.\.research-venv\Scripts\python.exe build_final_summary.py
.\.research-venv\Scripts\python.exe verify_delivery.py
```

The first verification creates the checked test outcome used in this summary; the final verification refreshes the hash manifest after the summary is written. On Linux/macOS use python3.11 to create the environment and .research-venv/bin/python for the executable prefix. Platform-dependent numeric/timing differences remain possible. verify_delivery runs the full unittest suite; it can also be run directly with `python -m unittest test_research -v`.

If completing the already-saved main study after an interrupted dimension process, run finish_research.py instead of rerunning the main pipeline, then continue with telemetry/convergence/reports/verification. This helper recomputes all 30 dimension fits and five initialization fits; it does not invent their lost values. For a fresh complete pipeline the helper is unnecessary.

The current checkout's local interpreter was installed in .python/ and used to create .research-venv; neither directory nor .uv-cache/ belongs in the submission. Inputs are local CSV only, with no experiment-time data download. root README explains clean installation and safe original reproduction. requirements-research.lock.txt lists exact installed distributions; environment.json and original_reproduction/environment.json list runtime/hardware identifiers and data/source hashes. Models are version-dependent trusted local pickles; do not load untrusted files. Reruns overwrite corresponding research artifacts; retain a copy when comparing environments.

### Failed, interrupted, unavailable or incomplete work

- Supplied old `.venv` failed to launch because its base Python was missing. Resolved by a separate runtime/environment; the original environment was preserved.
- Initial runtime download was blocked by sandbox networking; the approved isolated installation succeeded. CPU/RAM CIM queries were denied; available CPU identifier/logical count was recorded, while peak-memory measurement remains unavailable.
- The initial main pipeline ended during dimension execution before it wrote a final dimension table/environment/initialization file. Its main-study outputs survived. pipeline_initial_incomplete.log and executed_source/research_pipeline_initial.py preserve that attempt; its logged timings are not substituted for missing metrics. The cause of process termination was not confirmed. A separate completion run reran all 30 dimension fits and completed initialization sensitivity with per-fit persistence.
- Initial pre-pipeline tests passed four numerical tests and skipped two artifact tests as designed. Final verification ran all six with no skips. No failed model trial was removed to improve reported results.
- Exact reproduction of supplied historical repeated means and previous 5/6-input findings remains impossible without their code/settings/results. Current experiments do not certify those missing claims.
- No report edit, official IEEE compliance/word-count check, verified author/email entry, full-paper methodological reproduction, compact-rule experiment, categorical encoding ablation, broad nested search, external dataset validation, calibration intervention or human interpretability study was completed. These are omissions, not negative experimental results.

The measured completion-run wall time was **297.282 seconds**, covering the repeated dimension fits and initialization sensitivity only. The original full-program reproduction took **9.447 seconds**, including EDA/figures. Initial full-pipeline total elapsed time is unavailable; do not sum overlapping processes and call that wall time. Per-trial/per-dimension timings are retained. Hardware string: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, 12 logical CPUs; numerical compute limited to one BLAS thread.

### Generated files and source inventory

The ten requested research documents are RESEARCH_AUDIT.md, REPRODUCTION_REPORT.md, EXPERIMENTAL_DESIGN.md, ANFIS_DIAGNOSTICS.md, MODEL_COMPARISON.md, INTERPRETABILITY_CASE_STUDIES.md, RULE_COMPLEXITY_ANALYSIS.md, REPORT_READY_RESULTS.md, LIMITATIONS_AND_DISCUSSION.md and README.md under results/research/. The root README and Run.md were updated; original copies remain in original_snapshot/. main.py, original result tables and original figures remain unchanged.

The complete path/size/SHA256 inventory is [FILE_MANIFEST.csv](results/research/FILE_MANIFEST.csv), refreshed by final verification. The explicit path inventory below covers generated research files and reproducibility sources; virtual environments, cache and Python bytecode are excluded. Full rule/trajectory/model files are included, not only aggregate tables.

<details><summary>Complete generated-artifact and reproduction-source path inventory</summary>

- `FINAL_RESEARCH_SUMMARY.md`
- `README.md`
- `Run.md`
- `build_final_summary.py`
- `build_research_reports.py`
- `data/german_credit_data.csv`
- `dimension_convergence.py`
- `finish_research.py`
- `main.py`
- `reproduce_original.py`
- `requirements-research.lock.txt`
- `requirements-research.txt`
- `requirements.txt`
- `research_models.py`
- `research_pipeline.py`
- `research_telemetry.py`
- `results/research/ANFIS_DIAGNOSTICS.md`
- `results/research/EXPERIMENTAL_DESIGN.md`
- `results/research/FILE_MANIFEST.csv`
- `results/research/INTERPRETABILITY_CASE_STUDIES.md`
- `results/research/LIMITATIONS_AND_DISCUSSION.md`
- `results/research/MODEL_COMPARISON.md`
- `results/research/README.md`
- `results/research/REPORT_READY_RESULTS.md`
- `results/research/REPRODUCTION_REPORT.md`
- `results/research/RESEARCH_AUDIT.md`
- `results/research/RULE_COMPLEXITY_ANALYSIS.md`
- `results/research/aggregate_metrics.csv`
- `results/research/all_trials.csv`
- `results/research/cases/case_row_1_memberships.csv`
- `results/research/cases/case_row_1_rules.csv`
- `results/research/cases/case_row_6_memberships.csv`
- `results/research/cases/case_row_6_rules.csv`
- `results/research/cases/case_summaries.json`
- `results/research/cases/explained_model_consequents.csv`
- `results/research/completion_execution.log`
- `results/research/conditional_bootstrap.csv`
- `results/research/conditional_bootstrap_samples.csv`
- `results/research/config.json`
- `results/research/data_audit.json`
- `results/research/delivery_verification.json`
- `results/research/diagnostic_summary.csv`
- `results/research/diagnostics.json`
- `results/research/dimension_convergence.csv`
- `results/research/dimension_convergence_execution.log`
- `results/research/dimension_metrics.csv`
- `results/research/dimension_replay/19_base_fixed_adam.pkl`
- `results/research/dimension_replay/19_base_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/19_base_original.pkl`
- `results/research/dimension_replay/19_base_original_trajectory.csv`
- `results/research/dimension_replay/19_original_additions_5_fixed_adam.pkl`
- `results/research/dimension_replay/19_original_additions_5_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/19_original_additions_5_original.pkl`
- `results/research/dimension_replay/19_original_additions_5_original_trajectory.csv`
- `results/research/dimension_replay/19_original_additions_6_fixed_adam.pkl`
- `results/research/dimension_replay/19_original_additions_6_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/19_original_additions_6_original.pkl`
- `results/research/dimension_replay/19_original_additions_6_original_trajectory.csv`
- `results/research/dimension_replay/19_train_MI_5_fixed_adam.pkl`
- `results/research/dimension_replay/19_train_MI_5_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/19_train_MI_5_original.pkl`
- `results/research/dimension_replay/19_train_MI_5_original_trajectory.csv`
- `results/research/dimension_replay/19_train_MI_6_fixed_adam.pkl`
- `results/research/dimension_replay/19_train_MI_6_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/19_train_MI_6_original.pkl`
- `results/research/dimension_replay/19_train_MI_6_original_trajectory.csv`
- `results/research/dimension_replay/42_base_fixed_adam.pkl`
- `results/research/dimension_replay/42_base_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/42_base_original.pkl`
- `results/research/dimension_replay/42_base_original_trajectory.csv`
- `results/research/dimension_replay/42_original_additions_5_fixed_adam.pkl`
- `results/research/dimension_replay/42_original_additions_5_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/42_original_additions_5_original.pkl`
- `results/research/dimension_replay/42_original_additions_5_original_trajectory.csv`
- `results/research/dimension_replay/42_original_additions_6_fixed_adam.pkl`
- `results/research/dimension_replay/42_original_additions_6_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/42_original_additions_6_original.pkl`
- `results/research/dimension_replay/42_original_additions_6_original_trajectory.csv`
- `results/research/dimension_replay/42_train_MI_5_fixed_adam.pkl`
- `results/research/dimension_replay/42_train_MI_5_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/42_train_MI_5_original.pkl`
- `results/research/dimension_replay/42_train_MI_5_original_trajectory.csv`
- `results/research/dimension_replay/42_train_MI_6_fixed_adam.pkl`
- `results/research/dimension_replay/42_train_MI_6_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/42_train_MI_6_original.pkl`
- `results/research/dimension_replay/42_train_MI_6_original_trajectory.csv`
- `results/research/dimension_replay/7_base_fixed_adam.pkl`
- `results/research/dimension_replay/7_base_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/7_base_original.pkl`
- `results/research/dimension_replay/7_base_original_trajectory.csv`
- `results/research/dimension_replay/7_original_additions_5_fixed_adam.pkl`
- `results/research/dimension_replay/7_original_additions_5_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/7_original_additions_5_original.pkl`
- `results/research/dimension_replay/7_original_additions_5_original_trajectory.csv`
- `results/research/dimension_replay/7_original_additions_6_fixed_adam.pkl`
- `results/research/dimension_replay/7_original_additions_6_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/7_original_additions_6_original.pkl`
- `results/research/dimension_replay/7_original_additions_6_original_trajectory.csv`
- `results/research/dimension_replay/7_train_MI_5_fixed_adam.pkl`
- `results/research/dimension_replay/7_train_MI_5_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/7_train_MI_5_original.pkl`
- `results/research/dimension_replay/7_train_MI_5_original_trajectory.csv`
- `results/research/dimension_replay/7_train_MI_6_fixed_adam.pkl`
- `results/research/dimension_replay/7_train_MI_6_fixed_adam_trajectory.csv`
- `results/research/dimension_replay/7_train_MI_6_original.pkl`
- `results/research/dimension_replay/7_train_MI_6_original_trajectory.csv`
- `results/research/dimension_summary.csv`
- `results/research/environment.json`
- `results/research/executed_source/research_pipeline_initial.py`
- `results/research/final_complete_comparison.csv`
- `results/research/histories/101_none_fixed_adam_0.csv`
- `results/research/histories/101_none_fixed_adam_1.csv`
- `results/research/histories/101_none_mlp_0.csv`
- `results/research/histories/101_none_mlp_1.csv`
- `results/research/histories/101_none_original_0.csv`
- `results/research/histories/101_none_trainable_adam_0.csv`
- `results/research/histories/101_none_trainable_adam_1.csv`
- `results/research/histories/101_oversampled_fixed_adam_0.csv`
- `results/research/histories/101_oversampled_fixed_adam_1.csv`
- `results/research/histories/101_oversampled_mlp_0.csv`
- `results/research/histories/101_oversampled_mlp_1.csv`
- `results/research/histories/101_oversampled_original_0.csv`
- `results/research/histories/101_oversampled_trainable_adam_0.csv`
- `results/research/histories/101_oversampled_trainable_adam_1.csv`
- `results/research/histories/101_weighted_fixed_adam_0.csv`
- `results/research/histories/101_weighted_fixed_adam_1.csv`
- `results/research/histories/101_weighted_mlp_0.csv`
- `results/research/histories/101_weighted_mlp_1.csv`
- `results/research/histories/101_weighted_original_0.csv`
- `results/research/histories/101_weighted_trainable_adam_0.csv`
- `results/research/histories/101_weighted_trainable_adam_1.csv`
- `results/research/histories/19_none_fixed_adam_0.csv`
- `results/research/histories/19_none_fixed_adam_1.csv`
- `results/research/histories/19_none_mlp_0.csv`
- `results/research/histories/19_none_mlp_1.csv`
- `results/research/histories/19_none_original_0.csv`
- `results/research/histories/19_none_trainable_adam_0.csv`
- `results/research/histories/19_none_trainable_adam_1.csv`
- `results/research/histories/19_oversampled_fixed_adam_0.csv`
- `results/research/histories/19_oversampled_fixed_adam_1.csv`
- `results/research/histories/19_oversampled_mlp_0.csv`
- `results/research/histories/19_oversampled_mlp_1.csv`
- `results/research/histories/19_oversampled_original_0.csv`
- `results/research/histories/19_oversampled_trainable_adam_0.csv`
- `results/research/histories/19_oversampled_trainable_adam_1.csv`
- `results/research/histories/19_weighted_fixed_adam_0.csv`
- `results/research/histories/19_weighted_fixed_adam_1.csv`
- `results/research/histories/19_weighted_mlp_0.csv`
- `results/research/histories/19_weighted_mlp_1.csv`
- `results/research/histories/19_weighted_original_0.csv`
- `results/research/histories/19_weighted_trainable_adam_0.csv`
- `results/research/histories/19_weighted_trainable_adam_1.csv`
- `results/research/histories/42_none_fixed_adam_0.csv`
- `results/research/histories/42_none_fixed_adam_1.csv`
- `results/research/histories/42_none_mlp_0.csv`
- `results/research/histories/42_none_mlp_1.csv`
- `results/research/histories/42_none_original_0.csv`
- `results/research/histories/42_none_trainable_adam_0.csv`
- `results/research/histories/42_none_trainable_adam_1.csv`
- `results/research/histories/42_oversampled_fixed_adam_0.csv`
- `results/research/histories/42_oversampled_fixed_adam_1.csv`
- `results/research/histories/42_oversampled_mlp_0.csv`
- `results/research/histories/42_oversampled_mlp_1.csv`
- `results/research/histories/42_oversampled_original_0.csv`
- `results/research/histories/42_oversampled_trainable_adam_0.csv`
- `results/research/histories/42_oversampled_trainable_adam_1.csv`
- `results/research/histories/42_weighted_fixed_adam_0.csv`
- `results/research/histories/42_weighted_fixed_adam_1.csv`
- `results/research/histories/42_weighted_mlp_0.csv`
- `results/research/histories/42_weighted_mlp_1.csv`
- `results/research/histories/42_weighted_original_0.csv`
- `results/research/histories/42_weighted_trainable_adam_0.csv`
- `results/research/histories/42_weighted_trainable_adam_1.csv`
- `results/research/histories/73_none_fixed_adam_0.csv`
- `results/research/histories/73_none_fixed_adam_1.csv`
- `results/research/histories/73_none_mlp_0.csv`
- `results/research/histories/73_none_mlp_1.csv`
- `results/research/histories/73_none_original_0.csv`
- `results/research/histories/73_none_trainable_adam_0.csv`
- `results/research/histories/73_none_trainable_adam_1.csv`
- `results/research/histories/73_oversampled_fixed_adam_0.csv`
- `results/research/histories/73_oversampled_fixed_adam_1.csv`
- `results/research/histories/73_oversampled_mlp_0.csv`
- `results/research/histories/73_oversampled_mlp_1.csv`
- `results/research/histories/73_oversampled_original_0.csv`
- `results/research/histories/73_oversampled_trainable_adam_0.csv`
- `results/research/histories/73_oversampled_trainable_adam_1.csv`
- `results/research/histories/73_weighted_fixed_adam_0.csv`
- `results/research/histories/73_weighted_fixed_adam_1.csv`
- `results/research/histories/73_weighted_mlp_0.csv`
- `results/research/histories/73_weighted_mlp_1.csv`
- `results/research/histories/73_weighted_original_0.csv`
- `results/research/histories/73_weighted_trainable_adam_0.csv`
- `results/research/histories/73_weighted_trainable_adam_1.csv`
- `results/research/histories/7_none_fixed_adam_0.csv`
- `results/research/histories/7_none_fixed_adam_1.csv`
- `results/research/histories/7_none_mlp_0.csv`
- `results/research/histories/7_none_mlp_1.csv`
- `results/research/histories/7_none_original_0.csv`
- `results/research/histories/7_none_trainable_adam_0.csv`
- `results/research/histories/7_none_trainable_adam_1.csv`
- `results/research/histories/7_oversampled_fixed_adam_0.csv`
- `results/research/histories/7_oversampled_fixed_adam_1.csv`
- `results/research/histories/7_oversampled_mlp_0.csv`
- `results/research/histories/7_oversampled_mlp_1.csv`
- `results/research/histories/7_oversampled_original_0.csv`
- `results/research/histories/7_oversampled_trainable_adam_0.csv`
- `results/research/histories/7_oversampled_trainable_adam_1.csv`
- `results/research/histories/7_weighted_fixed_adam_0.csv`
- `results/research/histories/7_weighted_fixed_adam_1.csv`
- `results/research/histories/7_weighted_mlp_0.csv`
- `results/research/histories/7_weighted_mlp_1.csv`
- `results/research/histories/7_weighted_original_0.csv`
- `results/research/histories/7_weighted_trainable_adam_0.csv`
- `results/research/histories/7_weighted_trainable_adam_1.csv`
- `results/research/initialization_sensitivity.csv`
- `results/research/learning_rate_validation_ablation.csv`
- `results/research/models/101_none_dummy.pkl`
- `results/research/models/101_none_fixed_adam.pkl`
- `results/research/models/101_none_logistic.pkl`
- `results/research/models/101_none_mlp.pkl`
- `results/research/models/101_none_original.pkl`
- `results/research/models/101_none_trainable_adam.pkl`
- `results/research/models/101_oversampled_dummy.pkl`
- `results/research/models/101_oversampled_fixed_adam.pkl`
- `results/research/models/101_oversampled_logistic.pkl`
- `results/research/models/101_oversampled_mlp.pkl`
- `results/research/models/101_oversampled_original.pkl`
- `results/research/models/101_oversampled_trainable_adam.pkl`
- `results/research/models/101_weighted_dummy.pkl`
- `results/research/models/101_weighted_fixed_adam.pkl`
- `results/research/models/101_weighted_logistic.pkl`
- `results/research/models/101_weighted_mlp.pkl`
- `results/research/models/101_weighted_original.pkl`
- `results/research/models/101_weighted_trainable_adam.pkl`
- `results/research/models/19_none_dummy.pkl`
- `results/research/models/19_none_fixed_adam.pkl`
- `results/research/models/19_none_logistic.pkl`
- `results/research/models/19_none_mlp.pkl`
- `results/research/models/19_none_original.pkl`
- `results/research/models/19_none_trainable_adam.pkl`
- `results/research/models/19_oversampled_dummy.pkl`
- `results/research/models/19_oversampled_fixed_adam.pkl`
- `results/research/models/19_oversampled_logistic.pkl`
- `results/research/models/19_oversampled_mlp.pkl`
- `results/research/models/19_oversampled_original.pkl`
- `results/research/models/19_oversampled_trainable_adam.pkl`
- `results/research/models/19_weighted_dummy.pkl`
- `results/research/models/19_weighted_fixed_adam.pkl`
- `results/research/models/19_weighted_logistic.pkl`
- `results/research/models/19_weighted_mlp.pkl`
- `results/research/models/19_weighted_original.pkl`
- `results/research/models/19_weighted_trainable_adam.pkl`
- `results/research/models/42_none_dummy.pkl`
- `results/research/models/42_none_fixed_adam.pkl`
- `results/research/models/42_none_logistic.pkl`
- `results/research/models/42_none_mlp.pkl`
- `results/research/models/42_none_original.pkl`
- `results/research/models/42_none_trainable_adam.pkl`
- `results/research/models/42_oversampled_dummy.pkl`
- `results/research/models/42_oversampled_fixed_adam.pkl`
- `results/research/models/42_oversampled_logistic.pkl`
- `results/research/models/42_oversampled_mlp.pkl`
- `results/research/models/42_oversampled_original.pkl`
- `results/research/models/42_oversampled_trainable_adam.pkl`
- `results/research/models/42_weighted_dummy.pkl`
- `results/research/models/42_weighted_fixed_adam.pkl`
- `results/research/models/42_weighted_logistic.pkl`
- `results/research/models/42_weighted_mlp.pkl`
- `results/research/models/42_weighted_original.pkl`
- `results/research/models/42_weighted_trainable_adam.pkl`
- `results/research/models/73_none_dummy.pkl`
- `results/research/models/73_none_fixed_adam.pkl`
- `results/research/models/73_none_logistic.pkl`
- `results/research/models/73_none_mlp.pkl`
- `results/research/models/73_none_original.pkl`
- `results/research/models/73_none_trainable_adam.pkl`
- `results/research/models/73_oversampled_dummy.pkl`
- `results/research/models/73_oversampled_fixed_adam.pkl`
- `results/research/models/73_oversampled_logistic.pkl`
- `results/research/models/73_oversampled_mlp.pkl`
- `results/research/models/73_oversampled_original.pkl`
- `results/research/models/73_oversampled_trainable_adam.pkl`
- `results/research/models/73_weighted_dummy.pkl`
- `results/research/models/73_weighted_fixed_adam.pkl`
- `results/research/models/73_weighted_logistic.pkl`
- `results/research/models/73_weighted_mlp.pkl`
- `results/research/models/73_weighted_original.pkl`
- `results/research/models/73_weighted_trainable_adam.pkl`
- `results/research/models/7_none_dummy.pkl`
- `results/research/models/7_none_fixed_adam.pkl`
- `results/research/models/7_none_logistic.pkl`
- `results/research/models/7_none_mlp.pkl`
- `results/research/models/7_none_original.pkl`
- `results/research/models/7_none_trainable_adam.pkl`
- `results/research/models/7_oversampled_dummy.pkl`
- `results/research/models/7_oversampled_fixed_adam.pkl`
- `results/research/models/7_oversampled_logistic.pkl`
- `results/research/models/7_oversampled_mlp.pkl`
- `results/research/models/7_oversampled_original.pkl`
- `results/research/models/7_oversampled_trainable_adam.pkl`
- `results/research/models/7_weighted_dummy.pkl`
- `results/research/models/7_weighted_fixed_adam.pkl`
- `results/research/models/7_weighted_logistic.pkl`
- `results/research/models/7_weighted_mlp.pkl`
- `results/research/models/7_weighted_original.pkl`
- `results/research/models/7_weighted_trainable_adam.pkl`
- `results/research/normalization_only_metrics.csv`
- `results/research/normalization_only_predictions.csv`
- `results/research/original_gradient_trajectory.csv`
- `results/research/original_reproduction/anfis_rules.txt`
- `results/research/original_reproduction/environment.json`
- `results/research/original_reproduction/execution.log`
- `results/research/original_reproduction/experiment_configuration.csv`
- `results/research/original_reproduction/figures/01_class_distribution.png`
- `results/research/original_reproduction/figures/02_age_distribution.png`
- `results/research/original_reproduction/figures/02_credit_amount_distribution.png`
- `results/research/original_reproduction/figures/02_month_duration_distribution.png`
- `results/research/original_reproduction/figures/02_payment_to_income_ratio_distribution.png`
- `results/research/original_reproduction/figures/07_correlation_matrix.png`
- `results/research/original_reproduction/figures/08_anfis_training_loss.png`
- `results/research/original_reproduction/figures/09_roc_comparison.png`
- `results/research/original_reproduction/figures/anfis_confusion_matrix.png`
- `results/research/original_reproduction/figures/neural_network_confusion_matrix.png`
- `results/research/original_reproduction/membership_parameters.csv`
- `results/research/original_reproduction/model_comparison.csv`
- `results/research/original_snapshot/README.md`
- `results/research/original_snapshot/Run.md`
- `results/research/original_snapshot/anfis_rules.txt`
- `results/research/original_snapshot/experiment_configuration.csv`
- `results/research/original_snapshot/main.py`
- `results/research/original_snapshot/membership_parameters.csv`
- `results/research/original_snapshot/model_comparison.csv`
- `results/research/original_snapshot/requirements.txt`
- `results/research/paired_differences.csv`
- `results/research/paired_summary.csv`
- `results/research/per_split_metrics.csv`
- `results/research/pipeline_execution.log`
- `results/research/pipeline_initial_incomplete.log`
- `results/research/plots/case_row_1.png`
- `results/research/plots/case_row_6.png`
- `results/research/plots/diagnostics_seed42_weighted.png`
- `results/research/plots/dimension_runtime_performance.png`
- `results/research/plots/model_comparison.png`
- `results/research/plots/original_optimization_and_memberships.png`
- `results/research/plots/rule_concentration.png`
- `results/research/reproduction_check.csv`
- `results/research/split_assignments.csv`
- `results/research/telemetry_execution.log`
- `results/research/telemetry_verification.json`
- `results/research/test_outcome.json`
- `results/research/test_predictions.csv`
- `results/research/tests.log`
- `results/research/train_only_feature_ranking.csv`
- `test_research.py`
- `verify_delivery.py`

</details>

## 8. Three remaining improvements ranked by expected assessment impact

1. **Integrate and verify the actual submission.** Update the report with the separated original/new results, both real RQ3 cases and explicit limitations; check the official brief, IEEE conference layout, stated word count, verified authors/OsloMet emails and code+README appendix. This addresses the largest remaining completeness/clarity gap and makes the existing technical evidence assessable. The official report/brief are still required.
2. **Strengthen the scientific comparison without chasing scores.** Prioritize an explicit encoding sensitivity for coded inputs and a locked/nested validation protocol for thresholds and comparable stopping budgets; if feasible, use an independent final evaluation population. This targets technical quality and the main remaining uncertainty about generalization, feature choices and optimization fairness. Do not select a final method on the already inspected test means.
3. **Evaluate explanation usefulness and stability.** Check semantic labels/provenance, compare rule contributions across fitted seeds for the same records, and obtain a small expert comprehension assessment using predefined questions. This turns verified structural transparency into stronger evidence for RQ3 and can add significance/originality. A compact-rule study is supplementary, not a prerequisite to reporting the completed honest findings.

The ranking uses the assessment priorities supplied in the request (technical quality and completeness/clarity dominate), not a verified official marking rubric. No specific grade is predicted.

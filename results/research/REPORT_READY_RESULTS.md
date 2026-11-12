# Verified report-ready evidence and suggested text

This document is drafting support, not the missing final IEEE report. All numbers derive from executed CSV/JSON artifacts. Original and new experiments must remain separate in the paper. Exact original RQ wording must be reconciled with the report when available.

## Tables and figures

Table 1: original executable single split (`original_reproduction/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| ANFIS | 0.730000 | 0.571429 | 0.400000 | 0.470588 | 0.690357 |
| Neural Network | 0.690000 | 0.485294 | 0.550000 | 0.515625 | 0.692738 |


Table 2: new weighted condition, validation-selected thresholds, five overlapping holdouts; mean ± sample SD:

| model | balanced_accuracy | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| original | 0.600000 ± 0.036383 | 0.380000 ± 0.150647 | 0.411869 ± 0.083567 | 0.656857 ± 0.053678 | 0.462107 ± 0.053358 |
| fixed_adam | 0.594524 ± 0.038379 | 0.583333 ± 0.346210 | 0.433369 ± 0.113459 | 0.687405 ± 0.025717 | 0.489580 ± 0.057202 |
| trainable_adam | 0.590476 ± 0.027303 | 0.666667 ± 0.302536 | 0.452288 ± 0.094368 | 0.672238 ± 0.057764 | 0.482022 ± 0.069747 |
| mlp | 0.601905 ± 0.021372 | 0.606667 ± 0.196709 | 0.469669 ± 0.042563 | 0.680214 ± 0.006127 | 0.480514 ± 0.025601 |
| logistic | 0.597381 ± 0.028663 | 0.623333 ± 0.269928 | 0.460607 ± 0.070658 | 0.678762 ± 0.019305 | 0.479657 ± 0.043382 |
| dummy | 0.500000 ± 0.000000 | 1.000000 ± 0.000000 | 0.461538 ± 0.000000 | 0.500000 ± 0.000000 | 0.300000 ± 0.000000 |


Default thresholds and the other two imbalance conditions are in MODEL_COMPARISON.md; complete accuracy/precision/specificity/Brier and confusion data are in per_split_metrics.csv and aggregate_metrics.csv. Table 3 is dimension_summary.csv (means from dimension_metrics.csv); Table 4 is the two case summaries with full rule CSVs. All trial configurations and scores are retained in all_trials.csv.

Figure 1: plots/model_comparison.png. Figure 2: plots/original_optimization_and_memberships.png. Figure 3: plots/diagnostics_seed42_weighted.png. Figure 4: plots/rule_concentration.png. Figure 5: plots/dimension_runtime_performance.png. Figures 6–7: plots/case_row_1.png and plots/case_row_6.png. Error bars are split SDs, not independent-data confidence intervals. Threshold-sensitivity curves use test labels for diagnosis and are not model-selection plots.

## Exact numerical answers to the working RQs

RQ1, new weighted condition with validation thresholds: dummy: BA 0.500000, recall 1.000000, F1 0.461538, AUC 0.500000, AP 0.300000; fixed_adam: BA 0.594524, recall 0.583333, F1 0.433369, AUC 0.687405, AP 0.489580; logistic: BA 0.597381, recall 0.623333, F1 0.460607, AUC 0.678762, AP 0.479657; mlp: BA 0.601905, recall 0.606667, F1 0.469669, AUC 0.680214, AP 0.480514; original: BA 0.600000, recall 0.380000, F1 0.411869, AUC 0.656857, AP 0.462107; trainable_adam: BA 0.590476, recall 0.666667, F1 0.452288, AUC 0.672238, AP 0.482022. Fixed Adam minus MLP mean paired differences were BA -0.007381, recall -0.023333, F1 -0.036300, AUC 0.007190, AP 0.009066. Trainable-premise mean BA was 0.590476, compared with 0.594524 for fixed Adam. No universal superiority or statistical significance follows.

RQ2: full-grid rules were exactly 81/243/729 and consequent coefficients 405/1458/5103 at d=4/5/6. Observed runtime and performance depend on model and feature set:

| feature_set | d | model | fit_seconds | balanced_accuracy | roc_auc |
| --- | --- | --- | --- | --- | --- |
| base | 4 | fixed_adam | 0.169139 | 0.607540 | 0.703056 |
| base | 4 | original | 2.429680 | 0.606349 | 0.691508 |
| original_additions_5 | 5 | fixed_adam | 0.712249 | 0.601984 | 0.677976 |
| original_additions_5 | 5 | original | 9.562574 | 0.551984 | 0.652302 |
| original_additions_6 | 6 | fixed_adam | 2.162076 | 0.604762 | 0.663413 |
| original_additions_6 | 6 | original | 32.969713 | 0.538095 | 0.557619 |


These are three-seed means from new runs, not reproduced historical 5/6-input outcomes. Parameter growth is certain; predictive benefit depends on the introduced inputs and fitting protocol. Train-only ranking sensitivity remains in RULE_COMPLEXITY_ANALYSIS.md and train_only_feature_ranking.csv.

RQ3: correctly classified bad row 1 had score 0.575667125642 at threshold 0.470000000000, reconstructed logit 0.305011358523764, and stored reconstruction error 0.000e+00. Incorrectly classified good row 6 had score 0.489683527201, logit -0.041271748577546, and error 0.000e+00. All 81 signed rule contributions reconstruct both logits; automated tests additionally reload rounded CSVs and compare within 1e-12. Faithfulness is verified for these cases; human comprehensibility is not.

## Suggested Methods replacement text

We retained the original four-input fixed-premise Sugeno classifier as a reference and conducted a separate exploratory matched-split evaluation. Five stratified partitions used 640 fitting, 160 validation and 200 test observations, with identical held-out observations for each model. Standardization and Gaussian initialization used fitting data only. Unadjusted, weighted and random-oversampled conditions were applied within fitting data; validation and test observations were never oversampled. Alongside original gradient descent, two bounded Adam configurations were evaluated for fixed and trainable Gaussian premises. An MLP with the original (16,8) architecture, regularized logistic regression and a majority reference completed the comparison. Model configuration, stopping epoch and decision threshold were selected using validation data only. Results at threshold 0.5 and at the validation-selected threshold are reported separately.

## Suggested Results replacement text

The original executable single-split experiment was reproduced exactly in the documented pinned environment, yielding ROC-AUC 0.690357 for ANFIS and 0.692738 for the MLP. These values do not reproduce the previously supplied multi-split class-weighted summary, for which supporting code and per-split artifacts were unavailable. In the new weighted matched-split evaluation, fixed-premise Adam ANFIS achieved mean balanced accuracy 0.594524 (SD 0.038379), compared with 0.601905 (SD 0.021372) for the MLP. Their mean paired balanced-accuracy difference was -0.007381. These differences are descriptive because test partitions overlap. Full metric distributions and all tuning trials are retained in the accompanying artifacts.

The original weighted gradient-descent trajectory decreased training BCE from 0.693303 to 0.684904 over 500 updates and ended with gradient norm 0.026891. Validation-only learning-rate ablations probed whether this finite optimization budget constrained fitting. Numerical normalization effects were separately measured by holding coefficients and premises constant. These diagnostics distinguish optimization and numerical mechanisms from claims about generalization.

## Suggested Discussion replacement text

The results support treating the original ANFIS outcome as dependent on the chosen optimization and threshold protocol rather than as an intrinsic limit of fuzzy inference. However, the optimized model changes multiple aspects of fitting, including regularization and stopping, so predictive differences cannot be assigned exclusively to Adam or premise adaptation. More input variables deterministically expand the full rule grid, but their predictive effect remains confounded with the information content and encoding of the added variables. Repeated holdouts and the small number of bad-credit test observations limit inferential strength. The two real held-out cases demonstrate exact additive decomposition of logits, while their overlapping Gaussian antecedents and numerous linear consequents limit the claim that the entire system is readily understandable.

## Suggested Conclusion replacement text

The project now provides an executable baseline reproduction, a bounded comparison on matched partitions, and verified individual rule-level decompositions. The empirical findings are limited to the supplied historical four-input credit dataset and the documented exploratory protocols. Neither trainable fuzzy premises nor a larger full-grid rule base should be assumed to improve generalization. Reproducibility and local numerical transparency are established by saved partitions, scores, model states and tests; broader predictive validity and human interpretability require further evidence.

## Remaining report integration

Retain the original findings as a clearly labelled historical experiment, reconcile unsupported repeated summaries with their missing artifacts, and insert the new tables as a separate extension. Verify exact RQs, word count, IEEE template, author OsloMet addresses, official instructions and source-code/README appendix against the actual report and brief. No final word count, formatting compliance or grade is claimed here. Literature connections and verified bibliographic details are in LIMITATIONS_AND_DISCUSSION.md.

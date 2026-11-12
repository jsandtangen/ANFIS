# Rule growth and feature sensitivity (RQ2)

Implemented full-grid first-order counts are exact:

| d | rules | consequent_coefficients | fixed_premise_scalars | learned_premise_total_parameters | test_three_array_bytes |
| --- | --- | --- | --- | --- | --- |
| 4 | 81 | 405 | 24 | 429 | 388800 |
| 5 | 243 | 1458 | 30 | 1488 | 1166400 |
| 6 | 729 | 5103 | 36 | 5139 | 3499200 |


Each rule has d slopes plus one bias. Frozen Gaussian centres/widths are model state but not trained in original/fixed models. Trainable-premise variants add 6d scalars (three centres and three widths per input), giving 429/1488/5139 trainable parameters if used at d=4/5/6. Only four-input trainable-premise variants were run. Widths use log coordinates but each still counts as one degree of freedom.

Rules increase threefold per added input. Consequent coefficients increase from 405 to 1458 to 5103, a factor of 12.6 from four to six. This is a mathematical property of the partition, independently of empirical predictive benefit. Workspace estimates count only three 200-by-R float64 test matrices; they are neither measured process memory nor peak training RAM. Training (640 rows) and temporary arrays require more memory. OS RAM queries were denied.

## Executed dimension experiment

Three paired seeds, weighted fitting, same 640/160/200 protocol. Original GD uses .02/500; fixed Adam uses a pre-fixed .01/.001 configuration with validation stopping. No dimension-specific tuning. Table entries below are means; raw per-seed metrics, counts, time and normalized sums are in dimension_metrics.csv.

| feature_set | d | model | rules | consequent_parameters | fit_seconds | epochs_run | mean_effective_rules | balanced_accuracy | recall | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 4 | fixed_adam | 81.000000 | 405.000000 | 0.169139 | 63.000000 | 28.534699 | 0.607540 | 0.605556 | 0.703056 | 0.517130 |
| base | 4 | original | 81.000000 | 405.000000 | 2.429680 | 500.000000 | 155.851673 | 0.606349 | 0.338889 | 0.691508 | 0.494198 |
| original_additions_5 | 5 | fixed_adam | 243.000000 | 1458.000000 | 0.712249 | 56.333333 | 56.260732 | 0.601984 | 0.344444 | 0.677976 | 0.498074 |
| original_additions_5 | 5 | original | 243.000000 | 1458.000000 | 9.562574 | 500.000000 | 160.067468 | 0.551984 | 0.327778 | 0.652302 | 0.454486 |
| original_additions_6 | 6 | fixed_adam | 729.000000 | 5103.000000 | 2.162076 | 54.666667 | 101.787492 | 0.604762 | 0.600000 | 0.663413 | 0.484535 |
| original_additions_6 | 6 | original | 729.000000 | 5103.000000 | 32.969713 | 500.000000 | 155.325803 | 0.538095 | 0.583333 | 0.557619 | 0.384591 |
| train_MI_5 | 5 | fixed_adam | 243.000000 | 1458.000000 | 0.726957 | 57.666667 | 51.972932 | 0.621429 | 0.716667 | 0.702698 | 0.511916 |
| train_MI_5 | 5 | original | 243.000000 | 1458.000000 | 9.323722 | 500.000000 | 116.889950 | 0.604365 | 0.494444 | 0.660198 | 0.459663 |
| train_MI_6 | 6 | fixed_adam | 729.000000 | 5103.000000 | 2.366910 | 54.666667 | 119.194638 | 0.601984 | 0.577778 | 0.668889 | 0.492339 |
| train_MI_6 | 6 | original | 729.000000 | 5103.000000 | 33.643987 | 500.000000 | 149.041420 | 0.537698 | 0.594444 | 0.605278 | 0.422164 |


Original additions are residence_since at d=5 and residence_since plus n_credits at d=6. The base is exactly the four original inputs. Supplementary train_MI sets add the top one/two of residence_since, n_credits and n_guarantors according to fitting-only discrete mutual information. Selection rankings:

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


A ranking chosen on fitting data avoids test leakage but is itself uncertain. Coded numeric inputs make this a restricted sensitivity analysis. `n_guarantors` is used as an uninterpreted code until an authoritative mapping is supplied. The train_MI variants are reported alongside every requested variant, including coincidences; no best test subset is selected. Three seeds are too few to isolate subtle information effects from initialization, optimization or selection noise.

## Separating the claims

The rule/parameter count conclusion follows from Cartesian enumeration and can be verified without labels. Runtime combines more rules, implementation overhead and actual stopping epochs; original loops and vectorized Adam have very different implementation costs. Increasing input dimension also introduces different information, changes scaling and quantiles, and changes the initialized coefficient array. Consequently any change in AUC or recall cannot be attributed to rule count alone.

Quantile centres can coincide for discrete inputs, creating redundant premise sets and reducing distinct semantic rules below their nominal count. Mean effective rules is 1/sum(a_r²), averaged over test rows; it describes concentration rather than the number of linguistically distinct rules. The original epsilon-normalized values can sum below one, in which case this effective-count interpretation is distorted; consult normalized_sum_min before interpreting it. The stable new model has unit-sum normalization.

The supplied installment-burden, residence and credit-count codes are treated as ordered with equally spaced numeric values. This is an explicit approximation. A controlled one-hot/ordinal encoding comparison is still needed to justify distance semantics. No compact-rule variant or Cao et al. method is implemented. Plot: plots/dimension_runtime_performance.png.

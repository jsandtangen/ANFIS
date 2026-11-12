# Held-out interpretability case studies (RQ3)

The explained model is new fixed-premise Adam ANFIS, weighted condition, seed 42, selected using validation data only. Selection criteria were specified in config.json before new training: lowest source-row ID among correctly predicted bad cases and lowest ID among incorrectly predicted good cases, falling back to the requested correctness group if needed. These cases cover both true classes. Their test membership is saved in the checkpoint and tested.

For each rule, mu_jk=exp(-0.5*((x_j-c_jk)/sigma_jk)^2), w_r=product_j(mu_j,r_j), a_r=w_r/sum_s(w_s), f_r=beta_r^T[x,1], c_r=a_r*f_r. Log-domain evaluation implements exactly this relative normalization. z=sum_r c_r and score=sigmoid(z). The selected score threshold, rather than zero logit, determines the final class. Coefficients act on standardized inputs. Contributions have logit units; membership and firing values are not credit probabilities.

## Row 1: correct classification

Stable source reference: zero-based row 1, CSV line 3 including the header, dataset SHA256 `b0304c276a99d0e7d87ae4b23dcc6f48ef8456a7d040a9923640ef2986740130`. True class 1 (bad); predicted class 1. Score 0.575667125642, threshold 0.470000000000. Original feature order is month_duration, credit_amount, payment_to_income_ratio, age.

Original values: [48.0, 5951.0, 2.0, 22.0]. Standardized values: [2.240084374334162, 0.9855785906114148, -0.8573354355049446, -1.2226403295668138].

All input memberships (centres and widths in standardized units):

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


Five largest absolute signed contributions (rule IDs are zero-based):

| rule_id | antecedent | firing | normalized_firing | consequent | contribution |
| --- | --- | --- | --- | --- | --- |
| 72 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=LOW AND age=LOW | 0.015634 | 0.244391 | 0.386425 | 0.094439 |
| 73 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=LOW AND age=MEDIUM | 0.007821 | 0.122252 | 0.523445 | 0.063992 |
| 75 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=MEDIUM AND age=LOW | 0.007705 | 0.120448 | 0.487659 | 0.058737 |
| 76 | month_duration=HIGH AND credit_amount=HIGH AND payment_to_income_ratio=MEDIUM AND age=MEDIUM | 0.003854 | 0.060252 | 0.412144 | 0.024832 |
| 66 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=MEDIUM AND age=LOW | 0.002554 | 0.039927 | 0.561765 | 0.022429 |


The largest positive contribution is 0.094438736953 (rule 72); the most negative is -0.002804391154 (rule 46). All positive contributions sum to 0.311580480615; all negative contributions sum to -0.006569122091. Their sum is z=0.305011358523764. Direct forward output is 0.305011358523764; stored absolute reconstruction error is 0.000e+00. sigmoid(z)=0.575667125642.

For a small explicit numerical example, rule 0 has a=0.000394116717107 and f=-0.184720500077, giving c=-7.28014370727e-05. All 81 terms, rather than just the displayed terms, are required to reconstruct the output. Full precision values are in `cases/case_row_1_rules.csv`. Plot: `plots/case_row_1.png`.

## Row 6: incorrect classification

Stable source reference: zero-based row 6, CSV line 8 including the header, dataset SHA256 `b0304c276a99d0e7d87ae4b23dcc6f48ef8456a7d040a9923640ef2986740130`. True class 0 (good); predicted class 1. Score 0.489683527201, threshold 0.470000000000. Original feature order is month_duration, credit_amount, payment_to_income_ratio, age.

Original values: [24.0, 2835.0, 3.0, 53.0]. Standardized values: [0.2509702576864149, -0.1497168598093262, 0.034851033963615635, 1.5703572684303753].

All input memberships (centres and widths in standardized units):

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


Five largest absolute signed contributions (rule IDs are zero-based):

| rule_id | antecedent | firing | normalized_firing | consequent | contribution |
| --- | --- | --- | --- | --- | --- |
| 68 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.404251 | 0.073460 | 0.241958 | 0.017774 |
| 59 | month_duration=HIGH AND credit_amount=LOW AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.322313 | 0.058571 | -0.258010 | -0.015112 |
| 41 | month_duration=MEDIUM AND credit_amount=MEDIUM AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.324481 | 0.058965 | -0.226816 | -0.013374 |
| 65 | month_duration=HIGH AND credit_amount=MEDIUM AND payment_to_income_ratio=LOW AND age=HIGH | 0.199235 | 0.036205 | -0.237755 | -0.008608 |
| 32 | month_duration=MEDIUM AND credit_amount=LOW AND payment_to_income_ratio=MEDIUM AND age=HIGH | 0.258711 | 0.047013 | -0.174464 | -0.008202 |


The largest positive contribution is 0.017774356037 (rule 68); the most negative is -0.015111808016 (rule 59). All positive contributions sum to 0.053225152483; all negative contributions sum to -0.094496901061. Their sum is z=-0.041271748577546. Direct forward output is -0.041271748577546; stored absolute reconstruction error is 0.000e+00. sigmoid(z)=0.489683527201.

For a small explicit numerical example, rule 0 has a=0.000218461005982 and f=0.0173537093899, giving c=3.79110881082e-06. All 81 terms, rather than just the displayed terms, are required to reconstruct the output. Full precision values are in `cases/case_row_6_rules.csv`. Plot: `plots/case_row_6.png`.

## Interpretation and limits

The incorrectly predicted good observation can have a negative logit and still receive a bad label because the validation-selected threshold is below .5. This is a decision-policy effect, not a contradiction in reconstruction. Top rule signs reflect additive logit contributions relative to zero, not causal feature effects. Several overlapping rules can carry small contributions with opposite signs. The full tables make that cancellation observable.

`test_research.ArtifactTests.test_case_reconstruction_against_checkpoint` reloads the stored model and data, checks that each row belongs to the untouched test partition, recomputes strengths and consequents, reconstructs the logit from the exported CSV, and verifies sigmoid outputs. This is faithful local numerical decomposition. It does not establish that a human can understand all 81 rules, that explanations remain stable across fits, or that they justify a credit decision. RQ3 is supported at the level of verified model transparency with substantial semantic/comprehension limitations.

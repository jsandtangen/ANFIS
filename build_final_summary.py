"""Create the requested final summary exclusively from completed research artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_research_reports import table

OUT=Path('results/research')


def ms(x):return f'{x.mean():.6f} ± {x.std(ddof=1):.6f}'


def grouped(df,groups,metrics):
    return pd.DataFrame([dict(zip(groups,key if isinstance(key,tuple) else (key,)),
        **{metric:ms(g[metric]) for metric in metrics}) for key,g in df.groupby(groups,sort=False)])


def run():
    r=pd.read_csv(OUT/'per_split_metrics.csv');dims=pd.read_csv(OUT/'dimension_metrics.csv')
    conv=pd.read_csv(OUT/'dimension_convergence.csv');trials=pd.read_csv(OUT/'all_trials.csv')
    assert len(r)==180 and len(dims)==30 and len(conv)==30 and len(trials)==150
    cases=json.loads((OUT/'cases/case_summaries.json').read_text())
    telemetry=json.loads((OUT/'telemetry_verification.json').read_text())
    env=json.loads((OUT/'environment.json').read_text());diags=json.loads((OUT/'diagnostics.json').read_text())
    original=pd.read_csv(OUT/'original_reproduction/model_comparison.csv')
    uncertainty=pd.read_csv(OUT/'conditional_bootstrap.csv')
    outcomes=json.loads((OUT/'test_outcome.json').read_text())
    verification=json.loads((OUT/'delivery_verification.json').read_text())
    texts=['''# Final ACIT4620 research summary

Prepared 12 November 2026 from the completed local experiments. This is the evidence package for updating **Credit Risk Classification with a Simplified ANFIS and an MLP**, not an edited final paper or a grade prediction. The existing report and official examination brief are absent from this checkout. Section names below therefore identify the sections that should be changed; their existing numbers, exact wording and layout cannot be verified. Working RQs follow the supplied request.

The original source and saved top-level results remain intact. The baseline reproduction, new comparative study, diagnostic interventions and dimension sensitivity are distinguished throughout. Every numeric result below comes from a saved CSV/JSON or a calculation on those artifacts; no missing historical results have been estimated.

## Evidence boundary and original reproduction

The inspected original repository contained main.py, README.md, Run.md, unpinned requirements, one CSV, four result files and ten figures. No tests, report, notebooks, saved models, repeated-run logs, 5/6-input code, exam instructions or paper PDFs were present. Original Git HEAD was df2b4e97ecc5b111ff8cd231227f302746068de4. The old virtual environment pointed to a missing interpreter. A new Python 3.11.13 environment was installed without altering the old environment.

Executing unchanged main.py through reproduce_original.py reproduced every saved metric exactly:

''',table(original),'''
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

''']
    for condition in ['none','weighted','oversampled']:
        for mode in ['default','validation']:
            sub=r[(r.condition==condition)&(r.threshold_mode==mode)]
            texts+=[f'#### {condition}; {"fixed 0.5" if mode=="default" else "validation-selected"} threshold\n\n',
                table(grouped(sub,['model'],['accuracy','balanced_accuracy','precision','recall','f1','specificity'])), '\n',
                table(grouped(sub,['model'],['roc_auc','average_precision','brier','positive_rate','threshold'])), '\n',
                table(grouped(sub,['model'],['tn','fp','fn','tp','selected_fit_seconds','search_seconds'])), '\n']
    r.groupby(['condition','threshold_mode','model'])[['accuracy','balanced_accuracy','precision','recall','f1','specificity','roc_auc','average_precision','brier','positive_rate','tn','fp','fn','tp','threshold','selected_fit_seconds','search_seconds']].agg(['mean','std']).to_csv(OUT/'final_complete_comparison.csv')
    winners=[]
    candidates=r[r.model!='dummy'].groupby(['condition','threshold_mode','model']).mean(numeric_only=True).reset_index()
    for objective in ['accuracy','balanced_accuracy','precision','recall','f1','specificity','roc_auc','average_precision','brier']:
        for mode in ['default','validation']:
            sub=candidates[candidates.threshold_mode==mode]
            best=sub.loc[sub[objective].idxmin() if objective=='brier' else sub[objective].idxmax()]
            winners.append(dict(objective=objective,threshold_mode=mode,model=best.model,condition=best.condition,mean=best[objective]))
    texts+=['### Which model performs best for which objective?\n\n',table(pd.DataFrame(winners)),'''
This is a retrospective description of observed means, not a selection of a final model on test data. It is not a statistical winner table. Dummy is excluded because all-positive prediction trivially attains recall 1 and all-negative prediction specificity 1. The selected-threshold dummy predicts every observation bad: the validation BA ties at .5 and the first threshold .20 wins. Its F1 .461538 is a useful warning that F1 alone can reward a degenerate policy on this dataset.

For overall accuracy and uncalibrated Brier score, unadjusted logistic regression is the strongest observed reference; its low bad-credit recall at .5 shows why accuracy alone is inadequate. Under weighted default-threshold evaluation, MLP and fixed Adam are close in BA/F1, while fixed Adam has the highest observed weighted AUC/AP. Allowing trainable premises does not improve the weighted means relative to fixed Adam. The unadjusted MLP with a validation-selected threshold has the highest observed selected-threshold BA. None of these facts establishes a universally best classifier.

Training costs include fitting plus candidate-specific validation computations; search_seconds includes all trials in that model family, while selected_fit_seconds measures the selected candidate's computation. Timing is local, one-thread elapsed time. Some initial development diagnostics overlapped the first incomplete dimension run; final dimension benchmark timings come from its separate completion run. No CPU-independent efficiency claim is made.

### Differences and uncertainty

''']
    diffs=pd.read_csv(OUT/'paired_differences.csv')
    for condition in ['none','weighted','oversampled']:
        for mode in ['default','validation']:
            sub=diffs[(diffs.condition==condition)&(diffs.threshold_mode==mode)]
            texts+=[f'Paired differences: {condition}, {mode}; first model minus second, mean ± SD.\n\n',
                table(grouped(sub,['comparison'],['balanced_accuracy','recall','f1','roc_auc','average_precision'])), '\n']
    texts+=['''The saved paired_summary.csv additionally includes median and range; per-split effects are in paired_differences.csv. Repeated test sets overlap and models share many fitting observations. The SDs describe split sensitivity, not the standard error of five independent studies. No ordinary paired t-test, independent-split CI, or claim of statistical significance is justified here.

A supplementary seed-42 weighted, validation-threshold comparison used 2,000 stratified paired percentile bootstrap resamples of that test set; difference is fixed Adam minus MLP:

''',table(uncertainty[['metric','lower','upper']]),'''
These 95% intervals condition on those fitted models, chosen configurations, thresholds, test partition and class counts. They omit fitting/tuning uncertainty and do not generalize to all splits. In particular, seed-42 recall advantage can coexist with a different mean recall ordering across the five partitions. Many comparisons were inspected; all are exploratory. Each test partition contains only 60 bad cases, so recall moves in steps of 1/60.

Validation-selected thresholds sometimes reduce held-out BA or F1 relative to .5. This does not imply test leakage; it demonstrates limited transfer from a small, noisy validation set. The two threshold policies must remain separately labelled in the report. No operational costs or calibrated lending probabilities were estimated.

## 2. RQ2: rule complexity and dimensionality

### Exact structural counts

''']
    texts+=[table(pd.DataFrame([dict(inputs=d,rules=3**d,trained_consequent_parameters=3**d*(d+1),fixed_premise_scalars=6*d,
        total_stored_fuzzy_scalars=3**d*(d+1)+6*d,test_three_matrix_workspace_bytes=3*200*3**d*8) for d in [4,5,6]])),'''
Original/fixed models train only consequent coefficients, including biases. Their 6d premise scalars are frozen model state. If all premises were learned, the trainable count would be 429/1488/5139; only the four-input trainable variant was executed. The three-matrix workspace counts three N-by-R float64 test arrays, not peak RAM, interpreter overhead or every temporary. OS memory queries were denied, so no measured peak-memory result is claimed.

### Completed dimension results

All **30** planned dimension fits completed: 3 seeds × 5 feature-set specifications × 2 ANFIS protocols. They use the weighted condition, matched partitions, original GD .02/500 and fixed Adam .01/L2=.001 with validation stopping. No dimension-specific tuning was performed. Base inputs are month_duration, credit_amount, payment_to_income_ratio, age. Requested additions are residence_since (5) and then n_credits (6). Separate train_MI sensitivity adds the fitting-only top one/two discrete mutual-information-ranked codes among residence_since, n_credits, n_guarantors. Coinciding feature choices are still retained as separately labelled planned runs.

The following tables report mean ± sample SD over three overlapping holdouts, including every saved predictive metric, elapsed fitting time and relevant convergence/coverage measures. Raw seed-level outcomes are in dimension_metrics.csv.

''']
    groups=['feature_set','d','model']
    for cols in [['rules','consequent_parameters','fit_seconds','epochs_run','normalized_sum_min'],
        ['accuracy','balanced_accuracy','precision','recall','f1','specificity'],
        ['roc_auc','average_precision','brier','positive_rate'],['tn','fp','fn','tp']]:
        texts+=[table(grouped(dims,groups,cols)),'\n']
    texts+=['''Convergence was independently replayed for all 30 fits with the same initialization and data, using cached/vectorized mathematically equivalent original GD updates and the same fixed Adam implementation. Every saved dimension metric and validation-selected threshold agreed within 1e-10. Replay timings are **not** used as original-implementation benchmark timings. Replays add full trajectories and model states under dimension_replay/.

''',table(grouped(conv,groups,['best_epoch','selected_training_bce','selected_validation_bce','selected_consequent_gradient_l2','coefficient_l2'])),'''
Original has a fixed 500th-epoch endpoint, not a validation-selected optimum. Adam restores a validation-selected epoch and may stop later after patience. BCE columns above are unpenalized weighted BCE at the restored state; Adam stopping used BCE plus its coefficient penalty. Gradient values are consequent-only gradients at that state (including the penalty for Adam), so they are not directly identical convergence criteria across models. No claim of reaching a global optimum is made.

Fitting-only feature ranks and exact code values:

''',table(pd.read_csv(OUT/'train_only_feature_ranking.csv')),
        table(dims[['seed','feature_set','features']].drop_duplicates()),'''
### What the experiments actually demonstrate

Adding inputs deterministically multiplies the Cartesian rule count: 81→243→729. Consequent coefficients grow 405→1458→5103, a factor of 12.6 from four to six. This mathematical statement is separate from observed runtime and predictive quality. The original nested Python loops and larger dense arrays drive substantial runtime growth; vectorization/caching and stopping also contribute to the new implementation's time advantage. It would be incorrect to attribute the speed difference exclusively to Adam.

Predictive changes are the joint result of new information, coded feature semantics, rule count, initialization and fitting. A 4/5/6-input comparison alone does not isolate a causal effect of dimension on generalization. Train-only ranking provides a controlled sensitivity to the particular added codes, but uses only three seeds and a restricted candidate pool, not exhaustive feature selection. It does not justify choosing the best test subset.

The original epsilon normalization can make a_r sum below one. Therefore 1/sum(a_r²) can exceed the nominal number of rules: it is not a valid effective-rule count in affected original observations. Do not interpret that original diagnostic as genuine rule diversity. New exact-normalized activation concentration is meaningful as a relative concentration measure. Coincident Gaussian centres on coded inputs can create semantically duplicate sets even when the nominal Cartesian count is large.

Equal code spacing for installment burden, residence and number of credits is assumed by Gaussian/linear modeling, not validated. n_guarantors has an uncertain supplied semantic label; supplementary ranking treats it as a discrete code only. No categorical-encoding ablation, compact-rule implementation or reproduction of Cao et al. was completed. These remain limitations rather than invented results.

## 3. RQ3: real held-out explanations

Explained model: seed 42 weighted **fixed_adam**, using its validation-selected threshold. Before fitting, case criteria specified the lowest source-row ID among correctly predicted bad cases and the lowest ID among incorrectly predicted good cases. No case was chosen for visually appealing rules or large contributions. Both selected observations belong to the untouched test partition.

For standardized x, Gaussian grade mu_jk=exp(-0.5((x_j-c_jk)/sigma_jk)²), raw firing w_r=product_j(mu_j,r_j), normalized a_r=w_r/sum(w), first-order output f_r=beta_r·[x,1], contribution c_r=a_r f_r, and logit z=sum(c_r). Score=sigmoid(z), with a separate selected score threshold. Log-domain calculation implements exact relative normalization. Rule contributions are in logit units, and membership grades express set compatibility; neither is an outcome probability. All slopes/biases are in cases/explained_model_consequents.csv.

''']
    for case in cases:
        rules=pd.read_csv(OUT/case['rules_csv']);mf=pd.read_csv(OUT/case['membership_csv'])
        top=rules.iloc[np.argsort(-np.abs(rules.contribution.to_numpy()))[:8]]
        texts+=[f'''### Source row {case['row_zero_based']}: {'correct bad-credit classification' if case['true_label']==case['predicted_label'] else 'incorrect good-credit classification'}

Zero-based source row **{case['row_zero_based']}**, CSV line **{case['csv_line_including_header']}** including header. Source SHA256 is recorded in data_audit.json. True class={case['true_label']}, predicted class={case['predicted_label']}; good=0,bad=1. Original inputs in order (duration,amount,burden-code,age) are {case['original_inputs']}. Standardized inputs are {case['scaled_inputs']}.

Score **{case['score']:.12f}**; selected threshold **{case['threshold']:.12f}**; direct logit **{case['direct_logit']:.15f}**. All 12 input memberships:

''',table(mf), '\nEight most influential rules by absolute signed contribution (zero-based IDs):\n\n',table(top),f'''
The most positive term is {rules.contribution.max():.12f} (rule {int(rules.loc[rules.contribution.idxmax(),'rule_id'])}); the most negative is {rules.contribution.min():.12f} (rule {int(rules.loc[rules.contribution.idxmin(),'rule_id'])}). Positive terms sum to **{case['positive_contribution_sum']:.15f}**; negative terms sum to **{case['negative_contribution_sum']:.15f}**. All 81 terms sum to **{case['reconstructed_logit']:.15f}**, with stored in-memory reconstruction error **{case['error']:.3e}**. sigmoid(z) gives the score above. Exported CSV reconstruction and independent checkpoint recomputation are tested within 1e-12.

Full rule terms: [{case['rules_csv']}](results/research/{case['rules_csv']}). Membership file: [{case['membership_csv']}](results/research/{case['membership_csv']}). Figure: [case_row_{case['row_zero_based']}.png](results/research/plots/case_row_{case['row_zero_based']}.png).

''']
    texts+=['''### Meaning of the explanations

Row 1 is associated with a positive net logit and correctly classified bad credit. Its strongest displayed terms are local linear outputs gated by overlapping duration/amount/burden/age Gaussians. That states how the fitted model computes the score; it is not evidence that these attributes caused an actual default. Row 6 has a negative net logit but is still predicted bad because its score .489684 exceeds the validation threshold .47. That case exposes an understandable threshold-policy error and contribution cancellation that would be missed by displaying rule antecedents alone.

**Structural transparency** means the rule inventory, learned coefficients and arithmetic are accessible. **Faithful local explanation** is supported here because actual held-out logits and scores can be reconstructed exactly from all rules, with raw/scaled inputs, memberships, firing strengths, linear outputs and signed contributions exposed. **Meaningful human explanation** requires further evidence: whether a reader understands these terms, whether linguistic labels are appropriate, whether a small number of rules adequately summarizes the decision and whether explanations are stable. We have no user study, expert evaluation or cross-fit explanation-stability test. LOW/MEDIUM/HIGH are quantile labels rather than externally validated credit concepts; learned premises can also cross in the trainable variant. Top terms do not alone reconstruct the full decision, and contribution signs are relative to zero logit rather than a causal or counterfactual baseline.

RQ3 is therefore answered positively for verifiable local numerical decomposition, with clear limits for semantic and human interpretability. Fuzzy structure does not guarantee a meaningful explanation merely because rules can be printed.

## 4. Diagnosis of original ANFIS performance

### Verified observations

''',f'''1. **Limited optimization progress at the original budget.** In the seed-42 weighted replay, training BCE moved from {telemetry['initial']['training_bce']:.8f} to {telemetry['final']['training_bce']:.8f} over 500 updates; validation BCE moved from {telemetry['initial']['validation_bce']:.8f} to {telemetry['final']['validation_bce']:.8f}. The terminal gradient norm remained {telemetry['final']['gradient_l2']:.8f}. Cached replay coefficients matched the original within {telemetry['max_coefficient_difference']:.3e}. Nonstationarity and slow loss change are observed; no global optimum was certified.

2. **Normalization attenuation exists.** With original coefficients and premises fixed, exact log-domain normalization changed a seed-42 test logit by up to {telemetry['normalization_only_max_test_logit_difference']:.9f}. Original a_r values can have total below one when raw strengths are small, shrinking logits toward zero/score .5. This demonstrates a numerical mechanism, but not that it explains every classification error.

''',table(pd.read_csv(OUT/'normalization_only_metrics.csv')[['normalization','threshold_mode','balanced_accuracy','recall','f1','roc_auc','average_precision']]),'''
This intervention changes inference only and locks .5 or the original validation-selected threshold; it is a diagnostic comparison, not a newly tuned model. Per-row firing sums/logits are in normalization_only_predictions.csv.

3. **Threshold and imbalance policy materially change recall.** Weighted original mean recall is .700000 at .5 and .380000 under validation-selected thresholds, although score-ranking metrics are unchanged. Its selected-threshold accuracy .688000 is accompanied by lower bad recall. Near-single-class prediction counts and per-split matrices are retained in ANFIS_DIAGNOSTICS.md and per_split_metrics.csv. Unadjusted models at .5 typically miss most bad observations; all-negative majority accuracy is already .7.

4. **Grid size dilutes and complicates the parameterization.** Fixed normalized weights multiply each rule's consequent gradient. Many overlapping rules share credit for each observation. Larger grids increase the number of parameters and original implementation runtime. Absolute coverage and relative concentration are measured, not assumed; diagnostics.json contains every selected ANFIS fit's centres, widths, low-coverage fractions, coefficient norms, normalized-sum minima and rule-weight distribution.

5. **Original comparison was procedurally asymmetric.** ANFIS had weighting and fixed epochs; MLP had oversampling and internal early stopping. Duplicates could straddle MLP's internal stopping split. New external stopping and matched conditions resolve that specific comparison flaw, but they do not retroactively validate the original comparison as fair.

### Controlled learning-rate ablation and modifications

Same split, initialization, fixed premises, weights and 500 GD steps; learning rate only changes, evaluated on validation data:

''',table(pd.read_csv(OUT/'learning_rate_validation_ablation.csv')[['lr','train_loss_last_recorded','validation_bce','validation_balanced_accuracy','validation_recall','validation_f1']]),'''
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

''',f'''Final recorded unittest outcome: **{outcomes['tests_run']} tests, {outcomes['failures']} failures, {outcomes['errors']} errors, {outcomes['skipped']} skips**; successful={outcomes['successful']}. See tests.log and test_outcome.json.

1. `test_premise_and_consequent_gradients`: central finite differences for every slope/bias, Gaussian centre and log width on a small weighted problem.
2. `test_stable_inference_and_original_agreement`: ordinary-region original/stable agreement, finite extreme inputs and unit-sum stable weights.
3. `test_paired_split_and_resampling_integrity`: partition sizes/disjointness, stratified class counts and fitting-only resampling provenance.
4. `test_metrics_and_threshold`: known confusion matrix/metric values and deterministic threshold tie-breaking.
5. `test_case_reconstruction_against_checkpoint`: both real held-out rows, original data/scaler, exported rule weights/outputs/contributions, reconstructed logit and sigmoid score.
6. `test_saved_metrics_from_predictions`: all 180 main-study metric rows recomputed from saved test scores.

Delivery verification also reloaded **{verification['checkpoints_recomputed']}** selected checkpoint bundles, regenerated every test score, checked source-row partitions and fitting-only scaler means, verified original source/results byte preservation, and checked all 30 dimension rule/parameter counts. Maximum score difference was **{verification['maximum_score_difference']:.3e}**; maximum scaler-mean difference was **{verification['maximum_scaler_mean_difference']:.3e}**. Convergence replay adds independent dimension metric agreement. Structural/hash checks are evidence of consistency, not proof against all scientific design errors.

''','''### Reproduction commands

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

''',f'''The measured completion-run wall time was **{env['wall_seconds']:.3f} seconds**, covering the repeated dimension fits and initialization sensitivity only. The original full-program reproduction took **{json.loads((OUT/'original_reproduction/environment.json').read_text())['elapsed_seconds']:.3f} seconds**, including EDA/figures. Initial full-pipeline total elapsed time is unavailable; do not sum overlapping processes and call that wall time. Per-trial/per-dimension timings are retained. Hardware string: {env['processor']}, {env['logical_cpus']} logical CPUs; numerical compute limited to one BLAS thread.

''','''### Generated files and source inventory

The ten requested research documents are RESEARCH_AUDIT.md, REPRODUCTION_REPORT.md, EXPERIMENTAL_DESIGN.md, ANFIS_DIAGNOSTICS.md, MODEL_COMPARISON.md, INTERPRETABILITY_CASE_STUDIES.md, RULE_COMPLEXITY_ANALYSIS.md, REPORT_READY_RESULTS.md, LIMITATIONS_AND_DISCUSSION.md and README.md under results/research/. The root README and Run.md were updated; original copies remain in original_snapshot/. main.py, original result tables and original figures remain unchanged.

The complete path/size/SHA256 inventory is [FILE_MANIFEST.csv](results/research/FILE_MANIFEST.csv), refreshed by final verification. The explicit path inventory below covers generated research files and reproducibility sources; virtual environments, cache and Python bytecode are excluded. Full rule/trajectory/model files are included, not only aggregate tables.

<details><summary>Complete generated-artifact and reproduction-source path inventory</summary>

''']
    source_names=['main.py','reproduce_original.py','research_models.py','research_pipeline.py','research_telemetry.py',
        'finish_research.py','dimension_convergence.py','build_research_reports.py','build_final_summary.py',
        'verify_delivery.py','test_research.py','README.md','Run.md','requirements.txt','requirements-research.txt',
        'requirements-research.lock.txt','data/german_credit_data.csv','FINAL_RESEARCH_SUMMARY.md']
    files=sorted(set(source_names+[p.as_posix() for p in OUT.rglob('*') if p.is_file()]+['results/research/FILE_MANIFEST.csv']))
    texts+=['\n'.join(f'- `{path}`' for path in files),'\n\n</details>\n\n', '''## 8. Three remaining improvements ranked by expected assessment impact

1. **Integrate and verify the actual submission.** Update the report with the separated original/new results, both real RQ3 cases and explicit limitations; check the official brief, IEEE conference layout, stated word count, verified authors/OsloMet emails and code+README appendix. This addresses the largest remaining completeness/clarity gap and makes the existing technical evidence assessable. The official report/brief are still required.
2. **Strengthen the scientific comparison without chasing scores.** Prioritize an explicit encoding sensitivity for coded inputs and a locked/nested validation protocol for thresholds and comparable stopping budgets; if feasible, use an independent final evaluation population. This targets technical quality and the main remaining uncertainty about generalization, feature choices and optimization fairness. Do not select a final method on the already inspected test means.
3. **Evaluate explanation usefulness and stability.** Check semantic labels/provenance, compare rule contributions across fitted seeds for the same records, and obtain a small expert comprehension assessment using predefined questions. This turns verified structural transparency into stronger evidence for RQ3 and can add significance/originality. A compact-rule study is supplementary, not a prerequisite to reporting the completed honest findings.

The ranking uses the assessment priorities supplied in the request (technical quality and completeness/clarity dominate), not a verified official marking rubric. No specific grade is predicted.
''']
    Path('FINAL_RESEARCH_SUMMARY.md').write_text(''.join(texts),encoding='utf-8')
    print('Created FINAL_RESEARCH_SUMMARY.md from completed artifacts')


if __name__=='__main__':run()

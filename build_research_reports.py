"""Build report-ready Markdown from executed artifacts; never fills missing results."""
import hashlib
import importlib.metadata as md
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/research')


def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+ '\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'


def write(name,text):
    (OUT/name).write_text(text,encoding='utf-8')


def run():
    r=pd.read_csv(OUT/'per_split_metrics.csv');trials=pd.read_csv(OUT/'all_trials.csv')
    pred=pd.read_csv(OUT/'test_predictions.csv');dims=pd.read_csv(OUT/'dimension_metrics.csv')
    diags=json.loads((OUT/'diagnostics.json').read_text());cases=json.loads((OUT/'cases/case_summaries.json').read_text())
    telemetry=json.loads((OUT/'telemetry_verification.json').read_text())
    env=json.loads((OUT/'environment.json').read_text())
    conditional=pd.read_csv(OUT/'conditional_bootstrap.csv')
    metrics=['balanced_accuracy','recall','f1','roc_auc','average_precision']
    val=r[r.threshold_mode=='validation']
    weighted=val[val.condition=='weighted']
    summary=weighted.groupby('model')[metrics].agg(['mean','std'])
    def meanstd(condition,mode):
        sub=r[(r.condition==condition)&(r.threshold_mode==mode)]
        rows=[]
        for name,g in sub.groupby('model',sort=False):
            rows.append(dict(model=name,**{k:f'{g[k].mean():.6f} ± {g[k].std(ddof=1):.6f}' for k in metrics}))
        return pd.DataFrame(rows)
    original=pd.read_csv(OUT/'original_snapshot/model_comparison.csv')
    rerun=pd.read_csv(OUT/'original_reproduction/model_comparison.csv')
    checks=[]
    for _,row in original.iterrows():
        new=rerun[rerun.Model==row.Model].iloc[0]
        for key in original.columns[1:]:
            checks.append(dict(model=row.Model,metric=key,saved=float(row[key]),reproduced=float(new[key]),difference=float(new[key]-row[key])))
    pd.DataFrame(checks).to_csv(OUT/'reproduction_check.csv',index=False)
    data=json.loads((OUT/'data_audit.json').read_text())
    write('REPRODUCTION_REPORT.md',f'''# Reproduction of the original executable experiment

`reproduce_original.py` imported the unchanged `main.py`, redirected its output paths and executed `main()` in a one-thread context. It did not edit the original model, data preparation, weighting, oversampling, optimizer or threshold selection. Snapshot: Git HEAD df2b4e97ecc5b111ff8cd231227f302746068de4. Source SHA256: `{json.loads((OUT/'original_reproduction/environment.json').read_text())['main_sha256']}`.

## Verified result

{table(rerun)}

Every saved numeric result matched the rerun with maximum absolute difference {max(abs(c['difference']) for c in checks):.17g}. See `reproduction_check.csv`. The selected thresholds were ANFIS .51 and MLP .50. Confusion matrices (true good/bad rows, predicted good/bad columns) were ANFIS [[122,18],[36,24]] and MLP [[105,35],[27,33]], as recorded in `original_reproduction/execution.log`. The corresponding balanced accuracies are {(122/140+24/60)/2:.6f} and {(105/140+33/60)/2:.6f}; these are derived from the matrices, not originally present in the comparison table.

The split was 640 fitting / 160 validation / 200 test, stratified 70/30. Fit-only standardization used the original feature order. ANFIS trained with bad-class loss weight 2.33; MLP used oversampling followed by internal early stopping. This is not a matched class-weighted comparison.

## Environment and provenance

The old `.venv` could not start because its base interpreter was missing. A separate Python 3.11.13 environment with pinned scientific packages was installed in this project. `original_reproduction/environment.json` records runtime versions, elapsed full-program time ({json.loads((OUT/'original_reproduction/environment.json').read_text())['elapsed_seconds']:.3f} s), CPU identifier, platform and hashes. Full-program time includes EDA and plotting, so it is not an isolated fitting benchmark. `requirements-research.lock.txt` records all installed distributions for future installation. Exact agreement under this environment does not establish portability across arbitrary future versions.

Dataset SHA256: `{data['sha256']}`. Actual shape: {data['shape']}; missing cells: {data['missing']}; duplicate full rows: {data['duplicate_rows']}. Target counts: good=700, bad=300. The file is supplied locally, without an accompanying export script or data dictionary; its historical provenance beyond resemblance to Statlog is not independently established here.

## What could not be reproduced

The task's reported repeated class-weighted means (ANFIS/MLP balanced accuracy .536±.044/.605±.028, recall .417±.417/.707±.125, F1 .304±.167/.492±.024, AUC .545±.082/.671±.032, AP .361±.065/.466±.043) are not stored in this checkout. Their seeds, per-split scores, model settings and code are absent. The available program performs only one split and uses MLP oversampling. Those reported values remain unverified; new repeated evaluations below are new experiments and do not validate or overwrite them.

Original 5/6-feature code, checkpoints, timing outputs and report text are also absent. The new dimension experiments use the feature names described in the task but cannot be called exact reproductions of unavailable runs. The report's exact RQ wording and unofficial D assessment cannot be audited from repository evidence.
''')
    comparison='# New matched-split model comparison\n\nFive stratified holdouts; mean ± sample SD. Positive class is bad credit. The grids and partitions were fixed before these new runs.\n\n'
    for condition in ['none','weighted','oversampled']:
        for mode in ['default','validation']:
            comparison+=f'## {condition}: {mode} thresholds\n\n'+table(meanstd(condition,mode))+'\n'
    diff=pd.read_csv(OUT/'paired_differences.csv')
    comp=diff[(diff.condition=='weighted')&(diff.threshold_mode=='validation')]
    dtable=comp.groupby('comparison')[metrics].mean().reset_index()
    comparison+='## Paired differences, weighted validation thresholds\n\n'+table(dtable)
    comparison+='\nPositive differences favour the first model. `paired_summary.csv` includes SD, median and range; `paired_differences.csv` retains each seed. Overlapping holdouts are dependent, so these are descriptive differences. No independent-split standard error or t-test is asserted.\n\n'
    comparison+='Conditional seed-42 stratified paired bootstrap (fixed Adam minus MLP; percentile 95% intervals, 2,000 resamples):\n\n'+table(conditional[['metric','lower','upper']])
    comparison+='\nThese intervals condition on trained models, selected thresholds, the chosen partition and its class counts. They omit tuning/training uncertainty and are not a global test of superiority. See the full assumptions in EXPERIMENTAL_DESIGN.md.\n\n'
    timing=weighted.groupby('model')[['selected_fit_seconds','search_seconds']].agg(['mean','min','max'])
    timing.columns=['_'.join(c) for c in timing.columns]
    comparison+='## Local timing\n\n'+table(timing.reset_index())
    comparison+=f'\nThe pipeline completed {len(trials)} candidate fits/evaluations, selected 90 model-condition-split checkpoints, and produced {len(r)} metric rows and {len(pred)} held-out scores. The measured completion-run wall time was {env["wall_seconds"]:.3f} s; scope: {env.get("wall_scope","full main pipeline")}. Initial full-pipeline wall time is unavailable because its process ended during dimension work. Complete per-trial main-study elapsed times survived. Trials include the no-fit dummy reference. All candidate outcomes remain in all_trials.csv. No failed candidate is silently replaced.\n\n'
    comparison+='`per_split_metrics.csv` additionally contains accuracy, specificity, precision, Brier score, predicted-positive rates and every confusion matrix. `aggregate_metrics.csv` gives mean, SD, median and range for these metrics. Scores are uncalibrated, especially with modified fitting priors. Dummy at .5 predicts the majority class; validation threshold tie-breaking may instead select all-positive predictions, both with BA .5.\n'
    write('MODEL_COMPARISON.md',comparison)
    ds=pd.DataFrame([{k:d[k] for k in ['seed','condition','tag','normalized_sum_min','rows_raw_sum_below_1e10','mean_effective_rules','coefficient_l2','positive_rate_default','good_score_mean','bad_score_mean']} for d in diags])
    ds.to_csv(OUT/'diagnostic_summary.csv',index=False)
    diag42=ds[(ds.seed==42)&(ds.condition=='weighted')]
    init=pd.read_csv(OUT/'initialization_sensitivity.csv')
    rate=pd.read_csv(OUT/'learning_rate_validation_ablation.csv')
    original_rows=r[(r.model=='original')&(r.threshold_mode=='default')]
    near=original_rows[(original_rows.positive_rate<=.05)|(original_rows.positive_rate>=.95)]
    write('ANFIS_DIAGNOSTICS.md',f'''# ANFIS failure and stability diagnostics

These are measured diagnostics of the executable model. They do not establish the mechanism of unavailable historical runs. Test analyses describe already selected models; none change the main-study selection.

## Optimization evidence

The cached/vectorized replay of original seed-42 weighted GD differed from the unchanged implementation's fitted coefficients by at most {telemetry['max_coefficient_difference']:.3e}. It recorded epoch 0 through 500, including weighted validation BCE and gradient norms (`original_gradient_trajectory.csv`). Training BCE changed from {telemetry['initial']['training_bce']:.8f} to {telemetry['final']['training_bce']:.8f}; validation BCE changed from {telemetry['initial']['validation_bce']:.8f} to {telemetry['final']['validation_bce']:.8f}. Gradient L2 changed from {telemetry['initial']['gradient_l2']:.8f} to {telemetry['final']['gradient_l2']:.8f}. A nonzero terminal gradient and continuing loss change indicate the fixed 500 steps did not prove stationarity. Small per-rule normalized weights attenuate coefficient gradients; this is an algebraic mechanism, not proof of a unique cause of held-out errors.

Same initialization, same 500 steps, same fixed premises and weights, changing learning rate only (validation outputs):

{table(rate[['lr','train_loss_last_recorded','validation_bce','validation_balanced_accuracy','validation_recall','validation_f1']])}

This ablation evaluates no test thresholds and does not retrospectively replace selected configurations. Changes in fit and validation losses support an optimizer-budget diagnosis when they coincide; they do not show that higher learning rates always generalize better. Adam comparisons also change regularization and stopping, so attribute their combined effects cautiously.

## Coverage, normalization and score separation

Seed 42 weighted, held-out test diagnostics:

{table(diag42.drop(columns=['seed','condition']))}

Changing only original normalization to a log-domain exact normalization, retaining the fitted original premises and coefficients, changed a seed-42 weighted test logit by at most {telemetry['normalization_only_max_test_logit_difference']:.8g}. This explicitly tests epsilon-related attenuation; compare its scale with scores and margins before calling it a dominant performance cause. `diagnostics.json` records all split/condition minima and undercoverage counts, Gaussian centres/widths, coincident centre pairs, coefficient magnitudes and mean rule weights. Low feature coverage means max Gaussian grade below .01; saturated membership means grade below 1e-8. These operational cutoffs are diagnostics, not calibrated danger thresholds.

The new fixed and trainable models normalize in log space. This avoids product underflow and ensures unit-sum normalized weights even in tails, but relative normalization can still yield confident extrapolation where absolute membership is tiny. Thus numerical stability does not ensure adequate data support.

## Threshold and initialization sensitivity

Of {len(original_rows)} original-architecture default-threshold split-condition results, {len(near)} have a predicted-positive fraction <=.05 or >=.95. Their complete matrices and recalls are in per_split_metrics.csv. Selected validation thresholds are reported separately; a threshold can change recall and F1 without changing ROC-AUC or average precision. AUC and AP assess ranking and are unchanged between the two threshold rows of any one fitted model.

With split seed fixed to 42, original weighted GD was rerun using five consequent initialization seeds:

{table(init[['initialization_seed','threshold_mode','threshold','balanced_accuracy','recall','f1','roc_auc','positive_rate']])}

This separates initialization variability from changes in train/test composition. Both matter, but neither set of five runs establishes a universal instability rate. The original architecture's frozen premises cannot adapt regions to label structure; improvement from trainable premises must be assessed empirically, and may be negative. New comparisons preserve every result.

## Figures and artifacts

- `plots/original_optimization_and_memberships.png`: original training/validation BCE, gradient norm, and scaled membership functions.
- `plots/diagnostics_seed42_weighted.png`: test ROC, PR, threshold sensitivity, original score histograms, tuned validation objectives and gradient trajectories. Penalized objectives and original BCE are labelled and are not directly identical losses.
- `plots/rule_concentration.png`: sorted mean held-out normalized rule weights.
- `diagnostic_summary.csv`, `diagnostics.json`: numerical diagnostics for all selected ANFIS fits.
- `histories/`: all tuning trajectories, including rejected candidates.

The original main suppresses all Python warnings on import, and that global filter also applies to the extension importing it. Numerical tests, finite checks and convergence trajectories were used, but the absence of printed warnings must not be interpreted as a convergence certificate. This should be removed in a future separately recorded run if warning-level provenance is needed.
''')
    # Normalization-only intervention holds all fitted coefficients and premises fixed.
    from scipy.special import expit,logsumexp
    from research_pipeline import metrics as evaluate_metrics
    bundle=pickle.loads((OUT/'models/42_weighted_original.pkl').read_bytes())
    model=bundle['model'];source=pd.read_csv('data/german_credit_data.csv');idx=bundle['test_ids']
    X=bundle['scaler'].transform(source[FEATURE_NAMES].to_numpy(float)[idx]);yy=source.target.map({'good':0,'bad':1}).to_numpy()[idx]
    raw=model.calculate_strengths(X);normalized=model.normalize_strengths(raw)
    params=np.array(model.membership_parameters);mu=-.5*((X[:,:,None]-params[None,:,:,0])/params[None,:,:,1])**2
    rules=np.array(model.rules);lw=sum(mu[:,j,rules[:,j]] for j in range(4))
    stable=np.exp(lw-logsumexp(lw,axis=1,keepdims=True));outputs=model.rule_outputs(X)
    zo=model.forward(X);zs=(stable*outputs).sum(axis=1)
    pd.DataFrame(dict(row_id=idx,label=yy,raw_firing_sum=raw.sum(axis=1),original_normalized_sum=normalized.sum(axis=1),
        original_logit=zo,stable_logit=zs,original_score=expit(zo),stable_score=expit(zs))).to_csv(OUT/'normalization_only_predictions.csv',index=False)
    normalization=pd.DataFrame([dict(normalization=label,threshold_mode=mode,threshold=t,
        **evaluate_metrics(yy,p,t)) for label,p in [('original_epsilon',expit(zo)),('exact_log_domain',expit(zs))]
        for mode,t in [('default',.5),('original_validation',bundle['threshold'])]])
    normalization.to_csv(OUT/'normalization_only_metrics.csv',index=False)
    with (OUT/'ANFIS_DIAGNOSTICS.md').open('a',encoding='utf-8') as f:
        f.write('\n## Isolated normalization intervention\n\nHolding the fitted original coefficients and premises constant, replace only the firing normalization at inference. Thresholds remain .5 or the original validation-selected threshold; no threshold is selected on these test outcomes. This is a diagnostic intervention, not a retuned performance claim.\n\n'+table(normalization[['normalization','threshold_mode','threshold','balanced_accuracy','recall','f1','roc_auc','average_precision']])+'\nPer-row original/stable logits, scores and absolute firing sums are in normalization_only_predictions.csv.\n')
    explained=pickle.loads((OUT/'models/42_weighted_fixed_adam.pkl').read_bytes())['model']
    coefficients=pd.DataFrame(explained.coef,columns=FEATURE_NAMES+['intercept'])
    coefficients.insert(0,'rule_id',range(len(coefficients)))
    coefficients.to_csv(OUT/'cases/explained_model_consequents.csv',index=False)
    text='''# Held-out interpretability case studies (RQ3)

The explained model is new fixed-premise Adam ANFIS, weighted condition, seed 42, selected using validation data only. Selection criteria were specified in config.json before new training: lowest source-row ID among correctly predicted bad cases and lowest ID among incorrectly predicted good cases, falling back to the requested correctness group if needed. These cases cover both true classes. Their test membership is saved in the checkpoint and tested.

For each rule, mu_jk=exp(-0.5*((x_j-c_jk)/sigma_jk)^2), w_r=product_j(mu_j,r_j), a_r=w_r/sum_s(w_s), f_r=beta_r^T[x,1], c_r=a_r*f_r. Log-domain evaluation implements exactly this relative normalization. z=sum_r c_r and score=sigmoid(z). The selected score threshold, rather than zero logit, determines the final class. Coefficients act on standardized inputs. Contributions have logit units; membership and firing values are not credit probabilities.

'''
    for case in cases:
        record=case['row_zero_based'];rules=pd.read_csv(OUT/case['rules_csv']);mf=pd.read_csv(OUT/case['membership_csv'])
        text+=f'''## Row {record}: {'correct' if case['true_label']==case['predicted_label'] else 'incorrect'} classification

Stable source reference: zero-based row {record}, CSV line {case['csv_line_including_header']} including the header, dataset SHA256 `{data['sha256']}`. True class {case['true_label']} ({'bad' if case['true_label'] else 'good'}); predicted class {case['predicted_label']}. Score {case['score']:.12f}, threshold {case['threshold']:.12f}. Original feature order is {', '.join(FEATURE_NAMES)}.

Original values: {case['original_inputs']}. Standardized values: {case['scaled_inputs']}.

All input memberships (centres and widths in standardized units):

{table(mf)}

Five largest absolute signed contributions (rule IDs are zero-based):

{table(rules.iloc[np.argsort(-np.abs(rules.contribution.to_numpy()))[:5]])}

The largest positive contribution is {rules.contribution.max():.12f} (rule {int(rules.loc[rules.contribution.idxmax(),'rule_id'])}); the most negative is {rules.contribution.min():.12f} (rule {int(rules.loc[rules.contribution.idxmin(),'rule_id'])}). All positive contributions sum to {case['positive_contribution_sum']:.12f}; all negative contributions sum to {case['negative_contribution_sum']:.12f}. Their sum is z={case['reconstructed_logit']:.15f}. Direct forward output is {case['direct_logit']:.15f}; stored absolute reconstruction error is {case['error']:.3e}. sigmoid(z)={case['score']:.12f}.

For a small explicit numerical example, rule {int(rules.iloc[0].rule_id)} has a={rules.iloc[0].normalized_firing:.12g} and f={rules.iloc[0].consequent:.12g}, giving c={rules.iloc[0].contribution:.12g}. All 81 terms, rather than just the displayed terms, are required to reconstruct the output. Full precision values are in `{case['rules_csv']}`. Plot: `plots/case_row_{record}.png`.

'''
    text+='''## Interpretation and limits

The incorrectly predicted good observation can have a negative logit and still receive a bad label because the validation-selected threshold is below .5. This is a decision-policy effect, not a contradiction in reconstruction. Top rule signs reflect additive logit contributions relative to zero, not causal feature effects. Several overlapping rules can carry small contributions with opposite signs. The full tables make that cancellation observable.

`test_research.ArtifactTests.test_case_reconstruction_against_checkpoint` reloads the stored model and data, checks that each row belongs to the untouched test partition, recomputes strengths and consequents, reconstructs the logit from the exported CSV, and verifies sigmoid outputs. This is faithful local numerical decomposition. It does not establish that a human can understand all 81 rules, that explanations remain stable across fits, or that they justify a credit decision. RQ3 is supported at the level of verified model transparency with substantial semantic/comprehension limitations.
'''
    write('INTERPRETABILITY_CASE_STUDIES.md',text)
    dimcols=['rules','consequent_parameters','fit_seconds','epochs_run','mean_effective_rules','balanced_accuracy','recall','roc_auc','average_precision']
    dsumm=dims.groupby(['feature_set','d','model'])[dimcols].mean().reset_index()
    dsumm.to_csv(OUT/'dimension_summary.csv',index=False)
    complexity=pd.DataFrame([dict(d=d,rules=3**d,consequent_coefficients=3**d*(d+1),fixed_premise_scalars=6*d,
        learned_premise_total_parameters=3**d*(d+1)+6*d,test_three_array_bytes=3*200*3**d*8) for d in [4,5,6]])
    write('RULE_COMPLEXITY_ANALYSIS.md',f'''# Rule growth and feature sensitivity (RQ2)

Implemented full-grid first-order counts are exact:

{table(complexity)}

Each rule has d slopes plus one bias. Frozen Gaussian centres/widths are model state but not trained in original/fixed models. Trainable-premise variants add 6d scalars (three centres and three widths per input), giving 429/1488/5139 trainable parameters if used at d=4/5/6. Only four-input trainable-premise variants were run. Widths use log coordinates but each still counts as one degree of freedom.

Rules increase threefold per added input. Consequent coefficients increase from 405 to 1458 to 5103, a factor of 12.6 from four to six. This is a mathematical property of the partition, independently of empirical predictive benefit. Workspace estimates count only three 200-by-R float64 test matrices; they are neither measured process memory nor peak training RAM. Training (640 rows) and temporary arrays require more memory. OS RAM queries were denied.

## Executed dimension experiment

Three paired seeds, weighted fitting, same 640/160/200 protocol. Original GD uses .02/500; fixed Adam uses a pre-fixed .01/.001 configuration with validation stopping. No dimension-specific tuning. Table entries below are means; raw per-seed metrics, counts, time and normalized sums are in dimension_metrics.csv.

{table(dsumm)}

Original additions are residence_since at d=5 and residence_since plus n_credits at d=6. The base is exactly the four original inputs. Supplementary train_MI sets add the top one/two of residence_since, n_credits and n_guarantors according to fitting-only discrete mutual information. Selection rankings:

{table(pd.read_csv(OUT/'train_only_feature_ranking.csv'))}

A ranking chosen on fitting data avoids test leakage but is itself uncertain. Coded numeric inputs make this a restricted sensitivity analysis. `n_guarantors` is used as an uninterpreted code until an authoritative mapping is supplied. The train_MI variants are reported alongside every requested variant, including coincidences; no best test subset is selected. Three seeds are too few to isolate subtle information effects from initialization, optimization or selection noise.

## Separating the claims

The rule/parameter count conclusion follows from Cartesian enumeration and can be verified without labels. Runtime combines more rules, implementation overhead and actual stopping epochs; original loops and vectorized Adam have very different implementation costs. Increasing input dimension also introduces different information, changes scaling and quantiles, and changes the initialized coefficient array. Consequently any change in AUC or recall cannot be attributed to rule count alone.

Quantile centres can coincide for discrete inputs, creating redundant premise sets and reducing distinct semantic rules below their nominal count. Mean effective rules is 1/sum(a_r²), averaged over test rows; it describes concentration rather than the number of linguistically distinct rules. The original epsilon-normalized values can sum below one, in which case this effective-count interpretation is distorted; consult normalized_sum_min before interpreting it. The stable new model has unit-sum normalization.

The supplied installment-burden, residence and credit-count codes are treated as ordered with equally spaced numeric values. This is an explicit approximation. A controlled one-hot/ordinal encoding comparison is still needed to justify distance semantics. No compact-rule variant or Cao et al. method is implemented. Plot: plots/dimension_runtime_performance.png.
''')
    # Paper-facing figures use standard plotting, never substitute generated numerical evidence.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ['original','fixed_adam']:
        sub=dims[(dims.model==name)&dims.feature_set.isin(['base','original_additions_5','original_additions_6'])]
        g=sub.groupby('d')
        for ax,key in zip(axes,['fit_seconds','roc_auc']):
            ax.errorbar(g[key].mean().index,g[key].mean(),yerr=g[key].std(),marker='o',label=name)
            ax.set(xlabel='Inputs',ylabel=key);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'plots/dimension_runtime_performance.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,(name,g) in enumerate(weighted.groupby('model')):
        ax.errorbar(i,g.balanced_accuracy.mean(),yerr=g.balanced_accuracy.std(),fmt='o',capsize=4)
    ax.set_xticks(range(6),[name for name,_ in weighted.groupby('model')],rotation=15)
    ax.set(ylabel='Balanced accuracy; mean ± sample SD',title='Weighted condition, validation thresholds; five overlapping holdouts')
    fig.tight_layout();fig.savefig(OUT/'plots/model_comparison.png',dpi=160);plt.close(fig)
    rq1='; '.join(f'{name}: BA {g.balanced_accuracy.mean():.6f}, recall {g.recall.mean():.6f}, F1 {g.f1.mean():.6f}, AUC {g.roc_auc.mean():.6f}, AP {g.average_precision.mean():.6f}' for name,g in weighted.groupby('model'))
    paired=comp[comp.comparison=='fixed_adam minus mlp']
    fixed=weighted[weighted.model=='fixed_adam'];learned=weighted[weighted.model=='trainable_adam'];mlp=weighted[weighted.model=='mlp']
    write('REPORT_READY_RESULTS.md',f'''# Verified report-ready evidence and suggested text

This document is drafting support, not the missing final IEEE report. All numbers derive from executed CSV/JSON artifacts. Original and new experiments must remain separate in the paper. Exact original RQ wording must be reconciled with the report when available.

## Tables and figures

Table 1: original executable single split (`original_reproduction/model_comparison.csv`):

{table(rerun)}

Table 2: new weighted condition, validation-selected thresholds, five overlapping holdouts; mean ± sample SD:

{table(meanstd('weighted','validation'))}

Default thresholds and the other two imbalance conditions are in MODEL_COMPARISON.md; complete accuracy/precision/specificity/Brier and confusion data are in per_split_metrics.csv and aggregate_metrics.csv. Table 3 is dimension_summary.csv (means from dimension_metrics.csv); Table 4 is the two case summaries with full rule CSVs. All trial configurations and scores are retained in all_trials.csv.

Figure 1: plots/model_comparison.png. Figure 2: plots/original_optimization_and_memberships.png. Figure 3: plots/diagnostics_seed42_weighted.png. Figure 4: plots/rule_concentration.png. Figure 5: plots/dimension_runtime_performance.png. Figures 6–7: plots/case_row_1.png and plots/case_row_6.png. Error bars are split SDs, not independent-data confidence intervals. Threshold-sensitivity curves use test labels for diagnosis and are not model-selection plots.

## Exact numerical answers to the working RQs

RQ1, new weighted condition with validation thresholds: {rq1}. Fixed Adam minus MLP mean paired differences were BA {paired.balanced_accuracy.mean():.6f}, recall {paired.recall.mean():.6f}, F1 {paired.f1.mean():.6f}, AUC {paired.roc_auc.mean():.6f}, AP {paired.average_precision.mean():.6f}. Trainable-premise mean BA was {learned.balanced_accuracy.mean():.6f}, compared with {fixed.balanced_accuracy.mean():.6f} for fixed Adam. No universal superiority or statistical significance follows.

RQ2: full-grid rules were exactly 81/243/729 and consequent coefficients 405/1458/5103 at d=4/5/6. Observed runtime and performance depend on model and feature set:

{table(dsumm[dsumm.feature_set.isin(['base','original_additions_5','original_additions_6'])][['feature_set','d','model','fit_seconds','balanced_accuracy','roc_auc']])}

These are three-seed means from new runs, not reproduced historical 5/6-input outcomes. Parameter growth is certain; predictive benefit depends on the introduced inputs and fitting protocol. Train-only ranking sensitivity remains in RULE_COMPLEXITY_ANALYSIS.md and train_only_feature_ranking.csv.

RQ3: correctly classified bad row {cases[0]['row_zero_based']} had score {cases[0]['score']:.12f} at threshold {cases[0]['threshold']:.12f}, reconstructed logit {cases[0]['reconstructed_logit']:.15f}, and stored reconstruction error {cases[0]['error']:.3e}. Incorrectly classified good row {cases[1]['row_zero_based']} had score {cases[1]['score']:.12f}, logit {cases[1]['reconstructed_logit']:.15f}, and error {cases[1]['error']:.3e}. All 81 signed rule contributions reconstruct both logits; automated tests additionally reload rounded CSVs and compare within 1e-12. Faithfulness is verified for these cases; human comprehensibility is not.

## Suggested Methods replacement text

We retained the original four-input fixed-premise Sugeno classifier as a reference and conducted a separate exploratory matched-split evaluation. Five stratified partitions used 640 fitting, 160 validation and 200 test observations, with identical held-out observations for each model. Standardization and Gaussian initialization used fitting data only. Unadjusted, weighted and random-oversampled conditions were applied within fitting data; validation and test observations were never oversampled. Alongside original gradient descent, two bounded Adam configurations were evaluated for fixed and trainable Gaussian premises. An MLP with the original (16,8) architecture, regularized logistic regression and a majority reference completed the comparison. Model configuration, stopping epoch and decision threshold were selected using validation data only. Results at threshold 0.5 and at the validation-selected threshold are reported separately.

## Suggested Results replacement text

The original executable single-split experiment was reproduced exactly in the documented pinned environment, yielding ROC-AUC 0.690357 for ANFIS and 0.692738 for the MLP. These values do not reproduce the previously supplied multi-split class-weighted summary, for which supporting code and per-split artifacts were unavailable. In the new weighted matched-split evaluation, fixed-premise Adam ANFIS achieved mean balanced accuracy {fixed.balanced_accuracy.mean():.6f} (SD {fixed.balanced_accuracy.std():.6f}), compared with {mlp.balanced_accuracy.mean():.6f} (SD {mlp.balanced_accuracy.std():.6f}) for the MLP. Their mean paired balanced-accuracy difference was {paired.balanced_accuracy.mean():.6f}. These differences are descriptive because test partitions overlap. Full metric distributions and all tuning trials are retained in the accompanying artifacts.

The original weighted gradient-descent trajectory decreased training BCE from {telemetry['initial']['training_bce']:.6f} to {telemetry['final']['training_bce']:.6f} over 500 updates and ended with gradient norm {telemetry['final']['gradient_l2']:.6f}. Validation-only learning-rate ablations probed whether this finite optimization budget constrained fitting. Numerical normalization effects were separately measured by holding coefficients and premises constant. These diagnostics distinguish optimization and numerical mechanisms from claims about generalization.

## Suggested Discussion replacement text

The results support treating the original ANFIS outcome as dependent on the chosen optimization and threshold protocol rather than as an intrinsic limit of fuzzy inference. However, the optimized model changes multiple aspects of fitting, including regularization and stopping, so predictive differences cannot be assigned exclusively to Adam or premise adaptation. More input variables deterministically expand the full rule grid, but their predictive effect remains confounded with the information content and encoding of the added variables. Repeated holdouts and the small number of bad-credit test observations limit inferential strength. The two real held-out cases demonstrate exact additive decomposition of logits, while their overlapping Gaussian antecedents and numerous linear consequents limit the claim that the entire system is readily understandable.

## Suggested Conclusion replacement text

The project now provides an executable baseline reproduction, a bounded comparison on matched partitions, and verified individual rule-level decompositions. The empirical findings are limited to the supplied historical four-input credit dataset and the documented exploratory protocols. Neither trainable fuzzy premises nor a larger full-grid rule base should be assumed to improve generalization. Reproducibility and local numerical transparency are established by saved partitions, scores, model states and tests; broader predictive validity and human interpretability require further evidence.

## Remaining report integration

Retain the original findings as a clearly labelled historical experiment, reconcile unsupported repeated summaries with their missing artifacts, and insert the new tables as a separate extension. Verify exact RQs, word count, IEEE template, author OsloMet addresses, official instructions and source-code/README appendix against the actual report and brief. No final word count, formatting compliance or grade is claimed here. Literature connections and verified bibliographic details are in LIMITATIONS_AND_DISCUSSION.md.
''')
    files=['RESEARCH_AUDIT.md','REPRODUCTION_REPORT.md','EXPERIMENTAL_DESIGN.md','ANFIS_DIAGNOSTICS.md','MODEL_COMPARISON.md','INTERPRETABILITY_CASE_STUDIES.md','RULE_COMPLEXITY_ANALYSIS.md','REPORT_READY_RESULTS.md','LIMITATIONS_AND_DISCUSSION.md','README.md']
    write('README.md','''# ACIT4620 research evidence package

Start with REPORT_READY_RESULTS.md for verified tables and suggested academic prose, RESEARCH_AUDIT.md for implementation evidence, and REPRODUCTION_REPORT.md for the exact original rerun. The root README contains clean-environment commands. The original code and results remain preserved; all experiments here are labelled as reproduction or new work.

## Documents

'''+ '\n'.join(f'- [{p}]({p})' for p in files if p!='README.md')+'''

## Machine-readable artifacts

- config.json: declared new protocol and explanation-selection criteria.
- original_snapshot/: original source, instructions, dependency list and existing result tables.
- original_reproduction/: original execution log, regenerated figures and results, environment and hashes.
- reproduction_check.csv: saved-versus-reproduced metric differences.
- data_audit.json, environment.json, requirements-research.lock.txt at project root: data/runtime provenance.
- per_split_metrics.csv, aggregate_metrics.csv: each seed/condition/model/threshold and descriptive aggregates.
- all_trials.csv, histories/: every candidate and trajectory, including rejected configurations.
- test_predictions.csv, split_assignments.csv: held-out scores and source-row partition membership.
- paired_differences.csv, paired_summary.csv: paired descriptive effects.
- conditional_bootstrap.csv, conditional_bootstrap_samples.csv: conditional seed-42 paired test-sample bootstrap.
- diagnostics.json, diagnostic_summary.csv: premise coverage, score separation, weights and coefficients.
- original_gradient_trajectory.csv, telemetry_verification.json, learning_rate_validation_ablation.csv: optimization diagnosis.
- initialization_sensitivity.csv: five initializations at one fixed split.
- dimension_metrics.csv, dimension_summary.csv, train_only_feature_ranking.csv: new dimension/feature sensitivity.
- cases/: two real observations, memberships, all rule outputs and signed contributions, JSON reconstruction summaries.
- models/: 90 selected checkpoint bundles with model/scaler/feature order, split IDs and chosen threshold. Pickle is version-dependent; load only trusted local artifacts.
- plots/: diagnostic, comparison, dimension and case-study PNGs.
- pipeline_execution.log, telemetry_execution.log, tests.log: actual execution evidence.
- FILE_MANIFEST.csv: relative file path, bytes and SHA256 for delivery artifacts and source.

## Submission checklist, in priority order

1. Obtain the actual report, official exam description and required IEEE template; reconcile exact RQs and assessment requirements.
2. Resolve the missing original repeated-run artifacts or explicitly label those previous claims unverified. Preserve original and new results separately.
3. Integrate the verified comparison, original failure diagnostics and both real RQ3 cases; explain logits, thresholds, normalization and interpretation limits.
4. Include the full source and clean-environment README as required appendix material, with the data provenance/hash and pinned dependencies. Re-run commands in a fresh environment before submission.
5. Insert verified author names and OsloMet email addresses, check the stated 7,000–9,000-word requirement against the official brief, and validate IEEE layout and references.
6. Read the cited full papers and check all final bibliographic entries and methodological comparisons.
7. State overlapping-split dependence, exploratory tuning, restricted features/ordinal encoding, uncalibrated scores and absence of external validation. Do not claim statistical superiority or an assured grade.
8. If resources allow, add a separate encoding sensitivity study, broader nested tuning/external validation and human explanation evaluation. A compact-rule experiment is optional and must be labelled supplementary.

This package supplies reproducible evidence and drafting material. It is not a finished IEEE paper or a certification of examination compliance.
''')
    with (OUT/'README.md').open('a',encoding='utf-8') as index:
        index.write('\n## Final summary and completion evidence\n\nStart with [FINAL_RESEARCH_SUMMARY.md](../../FINAL_RESEARCH_SUMMARY.md) for the complete requested summary, all metric means/SDs, exact report-update instructions and explicit artifact inventory.\n\nThe initial pipeline ended before committing final dimension results; `pipeline_initial_incomplete.log` and `executed_source/research_pipeline_initial.py` preserve that attempt. `completion_execution.log` records a successful separate rerun of all 30 dimension fits plus initialization sensitivity. `dimension_convergence.csv` and `dimension_replay/` add 30 checked deterministic convergence replays; their times are not the original benchmark times. `normalization_only_predictions.csv` and `normalization_only_metrics.csv` isolate the inference normalization effect. `final_complete_comparison.csv` supplies complete mean/SD evaluation tables. `test_outcome.json` and `delivery_verification.json` record final checks.\n')
    Path('requirements-research.lock.txt').write_text('\n'.join(sorted(f'{dist.metadata["Name"]}=={dist.version}' for dist in md.distributions()))+'\n')
    print('Built research reports from executed artifacts')


FEATURE_NAMES=['month_duration','credit_amount','payment_to_income_ratio','age']
if __name__=='__main__':run()

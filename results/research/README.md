# ACIT4620 research evidence package

Start with REPORT_READY_RESULTS.md for verified tables and suggested academic prose, RESEARCH_AUDIT.md for implementation evidence, and REPRODUCTION_REPORT.md for the exact original rerun. The root README contains clean-environment commands. The original code and results remain preserved; all experiments here are labelled as reproduction or new work.

## Documents

- [RESEARCH_AUDIT.md](RESEARCH_AUDIT.md)
- [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)
- [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md)
- [ANFIS_DIAGNOSTICS.md](ANFIS_DIAGNOSTICS.md)
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md)
- [INTERPRETABILITY_CASE_STUDIES.md](INTERPRETABILITY_CASE_STUDIES.md)
- [RULE_COMPLEXITY_ANALYSIS.md](RULE_COMPLEXITY_ANALYSIS.md)
- [REPORT_READY_RESULTS.md](REPORT_READY_RESULTS.md)
- [LIMITATIONS_AND_DISCUSSION.md](LIMITATIONS_AND_DISCUSSION.md)

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

## Final summary and completion evidence

Start with [FINAL_RESEARCH_SUMMARY.md](../../FINAL_RESEARCH_SUMMARY.md) for the complete requested summary, all metric means/SDs, exact report-update instructions and explicit artifact inventory.

The initial pipeline ended before committing final dimension results; `pipeline_initial_incomplete.log` and `executed_source/research_pipeline_initial.py` preserve that attempt. `completion_execution.log` records a successful separate rerun of all 30 dimension fits plus initialization sensitivity. `dimension_convergence.csv` and `dimension_replay/` add 30 checked deterministic convergence replays; their times are not the original benchmark times. `normalization_only_predictions.csv` and `normalization_only_metrics.csv` isolate the inference normalization effect. `final_complete_comparison.csv` supplies complete mean/SD evaluation tables. `test_outcome.json` and `delivery_verification.json` record final checks.

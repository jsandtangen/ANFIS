# Reproduction of the original executable experiment

`reproduce_original.py` imported the unchanged `main.py`, redirected its output paths and executed `main()` in a one-thread context. It did not edit the original model, data preparation, weighting, oversampling, optimizer or threshold selection. Snapshot: Git HEAD df2b4e97ecc5b111ff8cd231227f302746068de4. Source SHA256: `fbdb6cd9bd12280b3cd98f13255194225407187ff7191ed6a876ae4b1010e070`.

## Verified result

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| ANFIS | 0.730000 | 0.571429 | 0.400000 | 0.470588 | 0.690357 |
| Neural Network | 0.690000 | 0.485294 | 0.550000 | 0.515625 | 0.692738 |


Every saved numeric result matched the rerun with maximum absolute difference 0. See `reproduction_check.csv`. The selected thresholds were ANFIS .51 and MLP .50. Confusion matrices (true good/bad rows, predicted good/bad columns) were ANFIS [[122,18],[36,24]] and MLP [[105,35],[27,33]], as recorded in `original_reproduction/execution.log`. The corresponding balanced accuracies are 0.635714 and 0.650000; these are derived from the matrices, not originally present in the comparison table.

The split was 640 fitting / 160 validation / 200 test, stratified 70/30. Fit-only standardization used the original feature order. ANFIS trained with bad-class loss weight 2.33; MLP used oversampling followed by internal early stopping. This is not a matched class-weighted comparison.

## Environment and provenance

The old `.venv` could not start because its base interpreter was missing. A separate Python 3.11.13 environment with pinned scientific packages was installed in this project. `original_reproduction/environment.json` records runtime versions, elapsed full-program time (9.447 s), CPU identifier, platform and hashes. Full-program time includes EDA and plotting, so it is not an isolated fitting benchmark. `requirements-research.lock.txt` records all installed distributions for future installation. Exact agreement under this environment does not establish portability across arbitrary future versions.

Dataset SHA256: `b0304c276a99d0e7d87ae4b23dcc6f48ef8456a7d040a9923640ef2986740130`. Actual shape: [1000, 21]; missing cells: 0; duplicate full rows: 0. Target counts: good=700, bad=300. The file is supplied locally, without an accompanying export script or data dictionary; its historical provenance beyond resemblance to Statlog is not independently established here.

## What could not be reproduced

The task's reported repeated class-weighted means (ANFIS/MLP balanced accuracy .536±.044/.605±.028, recall .417±.417/.707±.125, F1 .304±.167/.492±.024, AUC .545±.082/.671±.032, AP .361±.065/.466±.043) are not stored in this checkout. Their seeds, per-split scores, model settings and code are absent. The available program performs only one split and uses MLP oversampling. Those reported values remain unverified; new repeated evaluations below are new experiments and do not validate or overwrite them.

Original 5/6-feature code, checkpoints, timing outputs and report text are also absent. The new dimension experiments use the feature names described in the task but cannot be called exact reproductions of unavailable runs. The report's exact RQ wording and unofficial D assessment cannot be audited from repository evidence.

# ACIT4620: credit risk with simplified ANFIS and MLP

The original `main.py` and historical `results/` and `figures/` remain intact. Start with [FINAL_RESEARCH_SUMMARY.md](FINAL_RESEARCH_SUMMARY.md) for the complete verified summary, tables and report-update instructions. New, separately labelled research evidence is indexed in [results/research/README.md](results/research/README.md). [RESEARCH_AUDIT.md](results/research/RESEARCH_AUDIT.md) records implementation and missing-material details.

## Clean environment

Use Python 3.11 (executed with 3.11.13). From this repository root, on Windows PowerShell:

```powershell
py -3.11 -m venv .research-venv
.\.research-venv\Scripts\python.exe -m pip install -r requirements-research.lock.txt
.\.research-venv\Scripts\python.exe reproduce_original.py
.\.research-venv\Scripts\python.exe research_pipeline.py
.\.research-venv\Scripts\python.exe research_telemetry.py
.\.research-venv\Scripts\python.exe dimension_convergence.py
.\.research-venv\Scripts\python.exe -m unittest test_research -v
.\.research-venv\Scripts\python.exe build_research_reports.py
.\.research-venv\Scripts\python.exe verify_delivery.py
.\.research-venv\Scripts\python.exe build_final_summary.py
.\.research-venv\Scripts\python.exe verify_delivery.py
```

On macOS/Linux, create the environment using `python3.11 -m venv .research-venv` and replace the executable prefix with `.research-venv/bin/python`. The lock records installed distributions; platform-specific numerical differences and timings remain possible. `requirements-research.txt` specifies the direct scientific dependencies. The old `.venv` supplied with this checkout points to a missing interpreter; virtual environments should be recreated, not submitted as portable source.

The local execution used an isolated runtime at `.python/cpython-3.11.13-windows-x86_64-none/python.exe` to create `.research-venv`; that runtime and downloaded dependency caches are ignored by Git and are not necessary submission artifacts. An existing working Python 3.11 installation is sufficient.

## Inputs and expected execution

The supplied file is `data/german_credit_data.csv`, 1,000 rows, 700 good and 300 bad. The recorded SHA256 is in `results/research/data_audit.json`. No dataset download occurs during experiments. Positive class is bad=1. Four original features: month_duration, credit_amount, payment_to_income_ratio, age. The integer burden codes are not literal unqualified income percentages.

The complete research pipeline uses five paired splits, three imbalance conditions, original ANFIS, fixed and trainable-premise Adam ANFIS, MLP, logistic and a constant majority reference. It saves all trials, selected models, split IDs, scores and metrics. The dimension sensitivity uses three seeds and 4/5/6 inputs with two ANFIS protocols. Six-input original Python loops dominate runtime. Budget for several minutes on a desktop, with runtime dependent on hardware and background load; actual elapsed time is recorded. Telemetry and conditional bootstrap add further time. Run commands sequentially for new timing measurements. Local development tests and telemetry overlapped part of dimension execution, so existing timings are descriptive shared-machine elapsed measurements.

`reproduce_original.py` redirects the unchanged main program into `results/research/original_reproduction/`. Running `python main.py` directly remains possible after installing dependencies, but writes the historical top-level result and figure filenames. Prefer the reproduction wrapper to retain saved evidence. Original code and result tables were also copied into `results/research/original_snapshot/` before new work.

Re-running research commands overwrites the corresponding research artifacts deterministically for the recorded seeds; copy the research output directory first if retaining another run. Models are trusted local pickle bundles, tied to Python/sklearn versions. Do not load untrusted pickles.

## Source-code and README appendix delivery

Include `main.py`, `reproduce_original.py`, `research_models.py`, `research_pipeline.py`, `research_telemetry.py`, `finish_research.py`, `dimension_convergence.py`, `build_research_reports.py`, `build_final_summary.py`, `verify_delivery.py`, `test_research.py`, this README, both research requirements files, and the CSV data subject to its provenance/license verification. Include the research evidence package and its hash manifest, or the artifact access method permitted by the official exam brief. Source plus instructions are present; whether the report embeds/attaches them correctly cannot be checked because neither report nor official brief is in this repository.

Tests verify analytic gradients, stable inference, matched partitions and training-only resampling, actual held-out case reconstructions, and every saved main-study metric against saved scores. Two artifact tests are skipped until the pipeline runs. The completed run's `tests.log` records the actual final outcome. Evidence hashes and structural checks are generated by `verify_delivery.py`.

The requested IEEE report, exact RQ wording, author names/OsloMet emails and official word-count compliance remain to be checked against the missing report and exam brief. No grade is guaranteed. Use the priority checklist in the research README before submission.

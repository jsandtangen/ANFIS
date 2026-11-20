# Explainable Credit Risk Classification with ANFIS

Source code accompanying the ACIT4620 report. The project compares simplified ANFIS models with neural-network and statistical baselines on German Credit data. Experimental methods and findings are described in the report.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Initial ANFIS–MLP comparison, exploratory plots, evaluation, and fuzzy-rule export. |
| `research_models.py` | ANFIS implementation with Adam optimization and optional trainable membership functions. |
| `research_pipeline.py` | Repeated model comparisons, class-imbalance treatments, dimensionality experiments, and case explanations. |
| `research_telemetry.py` | Optimization diagnostics, learning-rate analysis, and conditional bootstrap intervals. |
| `compute_cost_analysis.py` | Misclassification-cost analysis using saved test confusion counts. |

## Dataset

The dataset is **not included in the report attachments**. Place the separately supplied CSV at:

```text
data/german_credit_data.csv
```

The code expects 1,000 observations with target labels `good` (0) and `bad` (1), distributed as 700/300. The four core inputs are `month_duration`, `credit_amount`, `payment_to_income_ratio`, and `age`. The research pipeline also requires `residence_since`, `n_credits`, and `n_guarantors`. Bad credit is the positive class.

## Setup and Execution

Use Python 3.11. The recorded environment used these dependencies:

```bash
python -m pip install numpy==2.3.5 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.7.2 matplotlib==3.10.7 threadpoolctl==3.6.0
```

Run from the project root:

```bash
python main.py
python research_pipeline.py
python research_telemetry.py
python compute_cost_analysis.py
```

`research_models.py` is imported automatically. Telemetry and cost analysis require the pipeline outputs. The current pipeline also requires `reproduce_original.py` alongside the scripts to record its final source hashes; it does not need to be executed separately.

## Outputs

The initial experiment saves plots in `figures/` and evaluation tables, rules, and membership parameters in `results/`. Extended experiments save metrics, predictions, models, diagnostics, and plots in `results/research/`.

The cost analysis produces `ACIT4620_cost_per_split.csv` and `ACIT4620_cost_summary.csv` in that directory. It evaluates the weighted condition using `(FP + r × FN) / N` for cost ratios `r = 1, 2, 5`, without retraining or changing thresholds.

Re-running scripts overwrites their corresponding outputs. Reproduction requires the same dataset, row order, dependencies, and configured seeds; numerical results and timings may vary across platforms.

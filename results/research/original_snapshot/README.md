# ACIT4620 - Explainable Credit Risk Classification with ANFIS

## Requirements

Python 3.10+ is recommended.

## 1. Open a terminal in this folder

```bash
cd ACIT4620_ANFIS_Project
```

## 2. Create a virtual environment (recommended)

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the project

```bash
python main.py
```

The program automatically loads:

`data/german_credit_data.csv`

## 5. Output

After running, check:

### figures/

Contains:
- class distribution
- feature distributions
- correlation matrix
- ANFIS training-loss curve
- ANFIS confusion matrix
- neural-network confusion matrix
- ROC comparison

### results/

Contains:
- `model_comparison.csv`
- `anfis_rules.txt`
- `membership_parameters.csv`
- `experiment_configuration.csv`

## Important

The ANFIS model deliberately uses four numerical features:

- month_duration
- credit_amount
- payment_to_income_ratio
- age

This keeps the fuzzy rule base manageable.

With four inputs and three membership functions per input, the model has:

3^4 = 81 fuzzy rules.

Do not change the experimental settings after seeing the results unless you document the change. This keeps the experiment reproducible.
"# ANFIS" 

"""
ACIT4620 - Explainable Credit Risk Classification with ANFIS
==============================================================

This project:
1. Loads the supplied German Credit CSV.
2. Performs basic data inspection and EDA.
3. Uses four numerical features for a compact ANFIS model.
4. Trains a neural-network baseline on the same features.
5. Evaluates both models with Accuracy, Precision, Recall, F1 and ROC-AUC.
6. Produces confusion matrices and an ROC comparison plot.
7. Saves the learned ANFIS rules and a model-comparison CSV.

Run:
    python main.py
"""

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "german_credit_data.csv"

FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# The actual supplied dataset uses these column names.
# Four numerical inputs keep the ANFIS rule base manageable.
FEATURES = [
    "month_duration",
    "credit_amount",
    "payment_to_income_ratio",
    "age",
]

TARGET = "target"


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Put german_credit_data.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    # Some CSV exports contain an unnamed index column.
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"File: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "The supplied CSV is missing these required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(f"  - {c}" for c in df.columns)
        )


# ============================================================
# DATA INSPECTION
# ============================================================

def inspect_data(df):
    print("\n" + "=" * 70)
    print("DATA INSPECTION")
    print("=" * 70)

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nTarget distribution:")
    print(df[TARGET].value_counts(dropna=False).to_string())

    print("\nData types:")
    print(df.dtypes.to_string())


# ============================================================
# TARGET ENCODING
# ============================================================

def encode_target(df):
    df = df.copy()

    # The supplied CSV uses string values: good / bad.
    # Some pandas versions expose the column as dtype=str rather than object,
    # so we explicitly detect string dtypes here.
    if pd.api.types.is_string_dtype(df[TARGET]) or df[TARGET].dtype == object:
        target_clean = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            "good": 0,
            "bad": 1,
        }

        unknown = sorted(set(target_clean.unique()) - set(mapping))
        if unknown:
            raise ValueError(
                f"Unknown target values: {unknown}\n"
                "Expected 'good' and 'bad'."
            )

        df[TARGET] = target_clean.map(mapping).astype(int)

    else:
        unique = sorted(df[TARGET].dropna().unique())

        if set(unique).issubset({0, 1}):
            df[TARGET] = df[TARGET].astype(int)
        elif set(unique).issubset({1, 2}):
            df[TARGET] = df[TARGET].map({1: 0, 2: 1}).astype(int)
        else:
            raise ValueError(
                f"Unsupported target values: {unique}"
            )

    return df


# ============================================================
# EDA
# ============================================================

def perform_eda(df):
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Target distribution
    counts = df[TARGET].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Good", "Bad"],
        [counts.get(0, 0), counts.get(1, 0)]
    )
    plt.title("Credit Risk Class Distribution")
    plt.xlabel("Credit class")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=300)
    plt.close()

    # Numerical feature distributions by target
    for feature in FEATURES:
        plt.figure(figsize=(8, 5))

        plt.hist(
            df.loc[df[TARGET] == 0, feature],
            bins=20,
            alpha=0.65,
            label="Good",
        )

        plt.hist(
            df.loc[df[TARGET] == 1, feature],
            bins=20,
            alpha=0.65,
            label="Bad",
        )

        plt.title(f"{feature}: Good vs Bad Credit")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        filename = (
            f"02_{feature}_distribution.png"
        )
        plt.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close()

    # Correlation matrix for selected numerical variables + target
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()

    labels = FEATURES + [TARGET]
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    plt.title("Correlation Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_correlation_matrix.png", dpi=300)
    plt.close()

    # Print numerical summary
    print("\nSelected feature statistics:")
    print(
        df[FEATURES]
        .describe()
        .round(2)
        .to_string()
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Convert numeric columns safely.
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Remove rows with missing selected inputs/target.
    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_fit,
        X_val,
        X_test,
        y_fit,
        y_val,
        y_test,
        X_fit_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# SIMPLE ANFIS
# ============================================================

class SimpleANFIS:
    """
    Compact Takagi-Sugeno-style neuro-fuzzy model.

    Each input has three Gaussian membership functions:
        LOW, MEDIUM, HIGH

    With 4 inputs this creates:
        3^4 = 81 fuzzy rules.

    Consequents are learned using gradient descent.
    """

    MF_NAMES = ["LOW", "MEDIUM", "HIGH"]

    def __init__(
        self,
        n_inputs,
        learning_rate=0.01,
        epochs=250,
        seed=42,
    ):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.n_mfs = 3

        self.rules = list(
            itertools.product(
                range(self.n_mfs),
                repeat=n_inputs,
            )
        )

        self.n_rules = len(self.rules)

        self.membership_parameters = None
        self.consequents = None
        self.training_history = []

    @staticmethod
    def gaussian(x, center, sigma):
        sigma = max(float(sigma), 1e-6)
        return np.exp(
            -0.5 * ((x - center) / sigma) ** 2
        )

    def initialize(self, X):
        self.membership_parameters = []

        for feature_idx in range(self.n_inputs):
            x = X[:, feature_idx]

            q25, q50, q75 = np.percentile(
                x,
                [25, 50, 75],
            )

            std = np.std(x)
            sigma = max(std * 0.75, 0.15)

            self.membership_parameters.append(
                [
                    (q25, sigma),
                    (q50, sigma),
                    (q75, sigma),
                ]
            )

        rng = np.random.default_rng(self.seed)

        self.consequents = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(self.n_rules, self.n_inputs + 1),
        )

    def calculate_strengths(self, X):
        n_samples = X.shape[0]

        strengths = np.ones(
            (n_samples, self.n_rules)
        )

        for rule_idx, rule in enumerate(self.rules):
            for feature_idx in range(self.n_inputs):
                mf_idx = rule[feature_idx]

                center, sigma = (
                    self.membership_parameters[
                        feature_idx
                    ][mf_idx]
                )

                memberships = self.gaussian(
                    X[:, feature_idx],
                    center,
                    sigma,
                )

                # Product t-norm for AND.
                strengths[:, rule_idx] *= memberships

        return strengths

    @staticmethod
    def normalize_strengths(strengths):
        denominator = (
            strengths.sum(
                axis=1,
                keepdims=True,
            )
            + 1e-10
        )

        return strengths / denominator

    def rule_outputs(self, X):
        X_bias = np.column_stack(
            [X, np.ones(X.shape[0])]
        )

        return X_bias @ self.consequents.T

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X):
        strengths = self.calculate_strengths(X)
        normalized = self.normalize_strengths(strengths)

        rule_outputs = self.rule_outputs(X)

        output = (
            normalized * rule_outputs
        ).sum(axis=1)

        return output

    def predict_proba(self, X):
        return self.sigmoid(
            self.forward(X)
        )

    def predict(self, X, threshold=0.5):
        return (
            self.predict_proba(X) >= threshold
        ).astype(int)

    def fit(self, X, y, class_weight=None):
        self.initialize(X)

        n_samples = X.shape[0]

        if class_weight is None:
            sample_weights = np.ones(n_samples)
        else:
            sample_weights = np.where(
                y == 1,
                class_weight.get(1, 1.0),
                class_weight.get(0, 1.0),
            )
        weight_sum = sample_weights.sum()

        print(
            f"\nANFIS: "
            f"{self.n_rules} rules, "
            f"{self.n_inputs} inputs, "
            f"{self.epochs} epochs"
        )

        for epoch in range(self.epochs):
            strengths = self.calculate_strengths(X)
            normalized = self.normalize_strengths(
                strengths
            )

            X_bias = np.column_stack(
                [X, np.ones(n_samples)]
            )

            rule_outputs = (
                X_bias @ self.consequents.T
            )

            raw_output = (
                normalized * rule_outputs
            ).sum(axis=1)

            probabilities = self.sigmoid(
                raw_output
            )

            # Binary cross-entropy.
            per_sample_loss = -(
                y * np.log(probabilities + 1e-10)
                + (1 - y) * np.log(1 - probabilities + 1e-10)
            )
            loss = np.sum(sample_weights * per_sample_loss) / weight_sum

            self.training_history.append(loss)

            # Gradient for each rule's consequent parameters.
            error = (probabilities - y) * sample_weights

            for rule_idx in range(self.n_rules):
                weighted_error = (
                    error
                    * normalized[:, rule_idx]
                )

                gradient = (
                    X_bias.T @ weighted_error
                    / weight_sum
                )

                self.consequents[rule_idx] -= (
                    self.learning_rate
                    * gradient
                )

            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{self.epochs} "
                    f"- loss: {loss:.5f}"
                )

        return self

    def get_rules(self, feature_names):
        """
        Returns all fuzzy rules.

        The consequent is a first-order Sugeno function:
            f = p1*x1 + ... + pn*xn + b
        """

        rules = []

        for rule_idx, rule in enumerate(self.rules):
            conditions = []

            for feature_idx, mf_idx in enumerate(rule):
                conditions.append(
                    f"{feature_names[feature_idx]} is "
                    f"{self.MF_NAMES[mf_idx]}"
                )

            coeffs = self.consequents[rule_idx]

            consequent_terms = []

            for i, feature in enumerate(feature_names):
                consequent_terms.append(
                    f"{coeffs[i]:.4f}*{feature}"
                )

            consequent = (
                " + ".join(consequent_terms)
                + f" + {coeffs[-1]:.4f}"
            )

            rules.append(
                f"Rule {rule_idx + 1}: "
                f"IF {' AND '.join(conditions)} "
                f"THEN output = {consequent}"
            )

        return rules

    def get_membership_parameters(
        self,
        feature_names,
    ):
        rows = []

        for feature_idx, feature in enumerate(
            feature_names
        ):
            for mf_idx, mf_name in enumerate(
                self.MF_NAMES
            ):
                center, sigma = (
                    self.membership_parameters[
                        feature_idx
                    ][mf_idx]
                )

                rows.append(
                    {
                        "Feature": feature,
                        "Membership": mf_name,
                        "Center": center,
                        "Sigma": sigma,
                    }
                )

        return pd.DataFrame(rows)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    threshold=0.5,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(f"Threshold   : {threshold:.2f}")
    for key, value in results.items():
        if key != "Model":
            print(f"{key:12s}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Good", "Bad"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )

    return results


def choose_threshold(y_true, probabilities):
    """Choose a classification threshold using validation data only.

    The test set is never used for threshold selection. We maximize F1
    while checking a grid of thresholds from 0.20 to 0.80.
    The validation criterion is balanced accuracy so both classes contribute
    equally despite the dataset imbalance.
    """
    thresholds = np.arange(0.20, 0.81, 0.01)
    from sklearn.metrics import balanced_accuracy_score
    scores = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        scores.append(balanced_accuracy_score(y_true, predictions))

    best_index = int(np.argmax(scores))
    return float(thresholds[best_index])


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    model_name,
    y_true,
    probabilities,
    threshold=0.5,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["Good", "Bad"],
    )

    plt.yticks(
        [0, 1],
        ["Good", "Bad"],
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# TRAINING CURVE
# ============================================================

def save_anfis_training_curve(history):
    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("ANFIS Training Loss")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "08_anfis_training_loss.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# ROC COMPARISON
# ============================================================

def save_roc_comparison(
    y_test,
    anfis_prob,
    nn_prob,
):
    fpr_anfis, tpr_anfis, _ = roc_curve(
        y_test,
        anfis_prob,
    )

    fpr_nn, tpr_nn, _ = roc_curve(
        y_test,
        nn_prob,
    )

    auc_anfis = roc_auc_score(
        y_test,
        anfis_prob,
    )

    auc_nn = roc_auc_score(
        y_test,
        nn_prob,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_anfis,
        tpr_anfis,
        label=f"ANFIS (AUC = {auc_anfis:.3f})",
    )

    plt.plot(
        fpr_nn,
        tpr_nn,
        label=f"Neural Network (AUC = {auc_nn:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curve: ANFIS vs Neural Network"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "09_roc_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# SAVE RULES
# ============================================================

def save_anfis_explainability(
    anfis,
    feature_names,
):
    rules = anfis.get_rules(
        feature_names
    )

    rules_path = (
        RESULTS_DIR / "anfis_rules.txt"
    )

    with open(
        rules_path,
        "w",
        encoding="utf-8",
    ) as f:
        for rule in rules:
            f.write(rule + "\n")

    membership_df = (
        anfis.get_membership_parameters(
            feature_names
        )
    )

    membership_df.to_csv(
        RESULTS_DIR / "membership_parameters.csv",
        index=False,
    )

    return rules_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("ACIT4620 - EXPLAINABLE CREDIT RISK WITH ANFIS")
    print("=" * 70)

    # 1. Load
    df = load_dataset()

    # 2. Validate
    validate_dataset(df)

    # 3. Inspect
    inspect_data(df)

    # 4. Encode target
    df = encode_target(df)

    # 5. EDA
    perform_eda(df)

    # 6. Prepare train/test data
    (
        X_fit,
        X_val,
        X_test,
        y_fit,
        y_val,
        y_test,
        X_fit_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = prepare_data(df)

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)
    print(f"Training samples:   {len(X_fit)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples:    {len(X_test)}")
    print("\nClass counts in training set:")
    print(y_fit.value_counts().sort_index().rename(index={0: "Good", 1: "Bad"}).to_string())

    # --------------------------------------------------------
    # ANFIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING ANFIS")
    print("=" * 70)

    anfis = SimpleANFIS(
        n_inputs=len(FEATURES),
        learning_rate=0.02,
        epochs=500,
        seed=RANDOM_STATE,
    )

    anfis.fit(
        X_fit_scaled,
        y_fit.to_numpy(),
        class_weight={0: 1.0, 1: 2.33},
    )

    anfis_val_prob = anfis.predict_proba(X_val_scaled)
    anfis_threshold = choose_threshold(y_val, anfis_val_prob)
    print(f"Selected ANFIS threshold from validation set: {anfis_threshold:.2f}")

    anfis_prob = anfis.predict_proba(X_test_scaled)

    anfis_results = evaluate_model(
        "ANFIS",
        y_test,
        anfis_prob,
        threshold=anfis_threshold,
    )

    save_confusion_matrix(
        "ANFIS",
        y_test,
        anfis_prob,
        threshold=anfis_threshold,
    )

    save_anfis_training_curve(
        anfis.training_history
    )

    rules_path = save_anfis_explainability(
        anfis,
        FEATURES,
    )

    # --------------------------------------------------------
    # NEURAL NETWORK BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING NEURAL NETWORK BASELINE")
    print("=" * 70)

    nn = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )

    # Balance the training data by random oversampling the minority class.
    rng = np.random.default_rng(RANDOM_STATE)
    good_idx = np.where(y_fit.to_numpy() == 0)[0]
    bad_idx = np.where(y_fit.to_numpy() == 1)[0]
    extra_bad = rng.choice(bad_idx, size=len(good_idx) - len(bad_idx), replace=True)
    balanced_idx = np.concatenate([good_idx, bad_idx, extra_bad])
    rng.shuffle(balanced_idx)

    X_nn_fit = X_fit_scaled[balanced_idx]
    y_nn_fit = y_fit.to_numpy()[balanced_idx]

    nn.fit(
        X_nn_fit,
        y_nn_fit,
    )

    nn_val_prob = nn.predict_proba(X_val_scaled)[:, 1]
    nn_threshold = choose_threshold(y_val, nn_val_prob)
    print(f"Selected NN threshold from validation set: {nn_threshold:.2f}")

    nn_prob = nn.predict_proba(
        X_test_scaled
    )[:, 1]

    nn_results = evaluate_model(
        "Neural Network",
        y_test,
        nn_prob,
        threshold=nn_threshold,
    )

    save_confusion_matrix(
        "Neural Network",
        y_test,
        nn_prob,
        threshold=nn_threshold,
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            anfis_results,
            nn_results,
        ]
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    save_roc_comparison(
        y_test,
        anfis_prob,
        nn_prob,
    )

    # --------------------------------------------------------
    # Print selected fuzzy rules
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANFIS EXPLAINABILITY")
    print("=" * 70)

    print(
        f"\nAll fuzzy rules saved to:\n"
        f"{rules_path}"
    )

    print("\nFirst 10 rules:")

    rules = anfis.get_rules(
        FEATURES
    )

    for rule in rules[:10]:
        print(rule)

    # --------------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------------

    config = pd.DataFrame(
        {
            "Setting": [
                "Dataset",
                "Target",
                "Features",
                "Test size",
                "Validation size within training set",
                "Random state",
                "Class balancing",
                "Decision threshold",
                "ANFIS membership functions",
                "ANFIS number of rules",
                "ANFIS epochs",
                "ANFIS learning rate",
                "ANFIS bad-class weight",
                "NN architecture",
            ],
            "Value": [
                str(DATA_PATH.name),
                TARGET,
                ", ".join(FEATURES),
                TEST_SIZE,
                VALIDATION_SIZE,
                RANDOM_STATE,
                "ANFIS weighted loss; NN random oversampling",
                f"ANFIS={anfis_threshold:.2f}; NN={nn_threshold:.2f}",
                "LOW, MEDIUM, HIGH (Gaussian)",
                anfis.n_rules,
                anfis.epochs,
                anfis.learning_rate,
                2.33,
                "(16, 8) + sigmoid output",
            ],
        }
    )

    config.to_csv(
        RESULTS_DIR / "experiment_configuration.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nGenerated folders:")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Results: {RESULTS_DIR}")

    print("\nYou can now use the generated:")
    print("- EDA figures")
    print("- ANFIS loss curve")
    print("- Confusion matrices")
    print("- ROC comparison")
    print("- Model comparison CSV")
    print("- Fuzzy rules")
    print("- Membership-function parameters")


if __name__ == "__main__":
    main()

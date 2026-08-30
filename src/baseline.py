"""
Centralized baseline model. This is the accuracy ceiling every later
FL / DP-FL result gets compared against.

Run with:
    python src/baseline.py --config configs/base.yaml
"""

import argparse

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUMERIC_COLS = ["annual_inc", "loan_amnt", "int_rate", "dti", "emp_length"]
CATEGORICAL_COLS = ["addr_state", "purpose"]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            # class_weight="balanced" is essential here: credit default data
            # is always imbalanced (defaults are the minority class), and
            # without this the model just learns to predict the majority
            # class and looks falsely good on accuracy alone.
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def main(config_path: str):
    config = load_config(config_path)
    data_path = config["data"]["processed_path"]
    target_col = config["data"]["target_column"]
    seed = config["experiment"]["seed"]

    df = pd.read_csv(data_path)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== Class Balance (test set) ===")
    print(y_test.value_counts(normalize=True).rename("proportion"))
    print()

    print("=== Centralized Baseline Results ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}  (misleading alone on imbalanced data)")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_proba):.4f}  (more informative than ROC-AUC on imbalanced data)")
    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)

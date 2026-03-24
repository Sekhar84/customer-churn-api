"""
Customer Churn API — Model Training (with MLflow)
===================================================
Trains a Logistic Regression classifier to predict customer churn.
Every run is logged to MLflow — parameters, metrics, and artefacts.

Run:
    python src/train.py                 <- default C=1.0
    python src/train.py --C 0.1        <- more regularisation
    python src/train.py --C 10.0       <- less regularisation

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    open http://127.0.0.1:5000
"""

import json
import argparse
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────
CLEAN_PATH = Path("data/churn_clean.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── MLflow setup ───────────────────────────────────────────────────────────
# These two lines run once when the module loads — before any training

# WHERE to store run data
# sqlite:///mlflow.db → local SQLite file called mlflow.db in project root
# http://server:5000  → remote MLflow server (production use)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# WHICH experiment this run belongs to
# Think of it as a folder — all churn model runs go here
# Creates the experiment automatically if it doesn't exist yet
mlflow.set_experiment("customer-churn")


def train(C=1.0, max_iter=1000, run_name="logistic-regression"):
    """
    Train a Logistic Regression churn model and log everything to MLflow.

    Parameters
    ----------
    C        : float  — regularisation strength (lower = more regularised)
    max_iter : int    — max solver iterations
    run_name : str    — label shown in MLflow UI
    """
    print("=" * 55)
    print("Customer Churn — Model Training")
    print(f"  C={C}  max_iter={max_iter}  run={run_name}")
    print("=" * 55)

    # ── Load and split data ────────────────────────────────────────────
    df = pd.read_csv(CLEAN_PATH)
    print(f"\nLoaded: {df.shape[0]:,} customers, {df.shape[1]-1} features")

    feature_cols = [c for c in df.columns if c != "Churn"]
    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ── Scale features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ══════════════════════════════════════════════════════════════════
    # MLFLOW — open a run
    # Everything inside this 'with' block is ONE run in MLflow
    # If training crashes, the run is automatically marked FAILED
    # ══════════════════════════════════════════════════════════════════
    with mlflow.start_run(run_name=run_name) as run:

        print(f"\nMLflow run: {run.info.run_id[:8]}...")

        # ── MLFLOW: log parameters (inputs) ───────────────────────────
        # Parameters are what you PUT IN to the training
        # Log BEFORE training — recorded even if training crashes
        mlflow.log_params({
            "model_type":  "LogisticRegression",
            "C":           C,
            "max_iter":    max_iter,
            "test_size":   0.2,
            "random_seed": 42,
            "n_features":  len(feature_cols),
            "train_rows":  len(X_train),
            "test_rows":   len(X_test),
        })

        # ── MLFLOW: log tags (searchable labels) ──────────────────────
        # Tags are metadata — useful for filtering runs in the UI
        mlflow.set_tags({
            "dataset":   "IBM Telco Churn",
            "framework": "sklearn",
            "task":      "binary_classification",
        })

        # ── Train model ────────────────────────────────────────────────
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(
            C            = C,
            max_iter     = max_iter,
            random_state = 42,
        )
        model.fit(X_train_scaled, y_train)
        print("  Done.")

        # ── Evaluate ───────────────────────────────────────────────────
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_prob)
        auc_pr   = average_precision_score(y_test, y_prob)

        print(f"\n{'─'*55}")
        print(f"Results:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")
        print(f"  AUC-PR   : {auc_pr:.4f}  <- primary metric (imbalanced)")
        print(f"\nClassification report:")
        print(classification_report(y_test, y_pred,
                                    target_names=["Stayed", "Churned"]))

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion matrix:")
        print(f"               Predicted")
        print(f"               Stayed  Churned")
        print(f"  Actual Stayed  {cm[0][0]:>5}   {cm[0][1]:>5}")
        print(f"  Actual Churned {cm[1][0]:>5}   {cm[1][1]:>5}")

        # ── MLFLOW: log metrics (outputs) ──────────────────────────────
        # AUC-PR is the primary metric for imbalanced classification
        # It measures precision-recall trade-off across all thresholds
        # More informative than AUC-ROC when positive class is minority (26.5%)
        mlflow.log_metrics({
            "auc_pr":          round(auc_pr, 4),   # PRIMARY metric
            "auc_roc":         round(auc, 4),       # secondary
            "accuracy":        round(accuracy, 4),
            "precision_churn": round(cm[1][1] / (cm[0][1] + cm[1][1]), 4),
            "recall_churn":    round(cm[1][1] / (cm[1][0] + cm[1][1]), 4),
        })

        # ── Feature importance ─────────────────────────────────────────
        coef = pd.Series(
            model.coef_[0], index=feature_cols
        ).sort_values(key=abs, ascending=False)

        print(f"\nTop 10 features:")
        print(f"  {'Feature':<30} {'Coefficient':>12}  Direction")
        print(f"  {'─'*30} {'─'*12}  {'─'*15}")
        for feat, val in coef.head(10).items():
            direction = "↑ increases churn" if val > 0 else "↓ reduces churn"
            print(f"  {feat:<30} {val:>12.4f}  {direction}")

        # ── Save artefacts locally ─────────────────────────────────────
        joblib.dump(model,        MODELS_DIR / "model.joblib")
        joblib.dump(scaler,       MODELS_DIR / "scaler.joblib")
        joblib.dump(feature_cols, MODELS_DIR / "features.joblib")

        metrics = {
            "accuracy": round(accuracy, 4),
            "auc":      round(auc, 4),
            "C":        C,
        }
        with open(MODELS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── MLFLOW: log artefacts (files) ──────────────────────────────
        # Attaches the models/ folder to this run in MLflow
        # You can download these files from the UI later
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")

        print(f"\nArtefacts saved to models/ and logged to MLflow")
        print(f"\n{'='*55}")
        print(f"Training complete")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")
        print(f"  AUC-PR   : {auc_pr:.4f}  <- primary metric")
        print(f"  MLflow run: {run.info.run_id[:8]}...")
        print(f"  View UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--C",        type=float, default=1.0,
                        help="Regularisation strength (default: 1.0)")
    parser.add_argument("--max-iter", type=int,   default=1000,
                        help="Max solver iterations (default: 1000)")
    parser.add_argument("--run-name", type=str,   default="logistic-regression",
                        help="MLflow run name")
    args = parser.parse_args()

    train(C=args.C, max_iter=args.max_iter, run_name=args.run_name)

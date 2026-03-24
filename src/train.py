"""
Customer Churn API — Model Training
=====================================
Trains a Logistic Regression classifier to predict customer churn.

Steps:
  1. Load clean data
  2. Split train/test (80/20, stratified)
  3. Scale features (StandardScaler)
  4. Train LogisticRegression
  5. Evaluate — accuracy, AUC, classification report
  6. Save model + scaler to models/

Run:
    python src/train.py

Outputs:
    models/model.joblib    — trained LogisticRegression
    models/scaler.joblib   — fitted StandardScaler
    models/features.joblib — feature column names
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────
CLEAN_PATH  = Path("data/churn_clean.csv")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train():
    print("=" * 55)
    print("Customer Churn — Model Training")
    print("=" * 55)

    # ── 1. Load clean data ──────────────────────────────────────────────
    df = pd.read_csv(CLEAN_PATH)
    print(f"\nLoaded: {df.shape[0]:,} customers, {df.shape[1]-1} features")
    print(f"Churn rate: {df['Churn'].mean():.1%}")

    # ── 2. Split features and target ────────────────────────────────────
    # X = all columns except Churn
    # y = Churn column only
    feature_cols = [c for c in df.columns if c != "Churn"]
    X = df[feature_cols]
    y = df["Churn"]

    # ── 3. Train/test split ─────────────────────────────────────────────
    # test_size=0.2   → 20% test, 80% train
    # stratify=y      → keeps churn rate equal in both splits
    # random_state=42 → reproducible split every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"\nTrain: {len(X_train):,} rows  (churn rate: {y_train.mean():.1%})")
    print(f"Test:  {len(X_test):,} rows  (churn rate: {y_test.mean():.1%})")

    # ── 4. Scale features ───────────────────────────────────────────────
    # StandardScaler: transforms each feature to mean=0, std=1
    # IMPORTANT: fit ONLY on train, transform both train and test
    # Fitting on test would leak test information into training — wrong
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)         # transform only

    # ── 5. Train model ──────────────────────────────────────────────────
    # max_iter=1000: LogisticRegression needs enough iterations to converge
    # C=1.0: regularisation strength (default) — prevents overfitting
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        random_state = 42,
    )
    model.fit(X_train_scaled, y_train)
    print("  Done.")

    # ── 6. Evaluate ─────────────────────────────────────────────────────
    y_pred      = model.predict(X_test_scaled)
    y_prob      = model.predict_proba(X_test_scaled)[:, 1]

    accuracy    = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    print(f"\n{'─'*55}")
    print(f"Test set results:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Stayed", "Churned"]))

    print(f"Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"               Predicted")
    print(f"               Stayed  Churned")
    print(f"  Actual Stayed  {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"  Actual Churned {cm[1][0]:>5}   {cm[1][1]:>5}")

    # ── 7. Feature importance ───────────────────────────────────────────
    # In Logistic Regression, coefficients show feature impact
    # Positive coefficient → feature increases churn probability
    # Negative coefficient → feature decreases churn probability
    coef = pd.Series(
        model.coef_[0],
        index=feature_cols
    ).sort_values(key=abs, ascending=False)

    print(f"\nTop 10 features by importance:")
    print(f"  {'Feature':<30} {'Coefficient':>12}  {'Direction'}")
    print(f"  {'─'*30} {'─'*12}  {'─'*15}")
    for feat, val in coef.head(10).items():
        direction = "↑ increases churn" if val > 0 else "↓ reduces churn"
        print(f"  {feat:<30} {val:>12.4f}  {direction}")

    # ── 8. Save artefacts ───────────────────────────────────────────────
    joblib.dump(model,        MODELS_DIR / "model.joblib")
    joblib.dump(scaler,       MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "features.joblib")

    # Save metrics to JSON for CI quality gate and MLflow later
    metrics = {
        "accuracy":    round(accuracy, 4),
        "auc":         round(auc, 4),
        "train_rows":  len(X_train),
        "test_rows":   len(X_test),
        "churn_rate":  round(float(y.mean()), 4),
        "n_features":  len(feature_cols),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtefacts saved:")
    print(f"  models/model.joblib")
    print(f"  models/scaler.joblib")
    print(f"  models/features.joblib")
    print(f"  models/metrics.json")

    print(f"\n{'='*55}")
    print(f"Training complete")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()

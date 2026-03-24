"""
Customer Churn API — Data Preprocessing
========================================
Reads raw Telco churn CSV, cleans and encodes it,
saves a model-ready dataset to data/churn_clean.csv

Run:
    python src/preprocess.py

Input:  data/telco_churn.csv   (7,043 rows, 21 columns)
Output: data/churn_clean.csv   (7,043 rows, numeric only)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_PATH   = Path("data/telco_churn.csv")
CLEAN_PATH = Path("data/churn_clean.csv")


def preprocess():
    print("=" * 55)
    print("Customer Churn — Preprocessing")
    print("=" * 55)

    # ── 1. Load raw data ────────────────────────────────────────────────
    df = pd.read_csv(RAW_PATH)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Churn rate:     {(df['Churn'] == 'Yes').mean():.1%}")

    # ── 2. Fix TotalCharges ─────────────────────────────────────────────
    # TotalCharges is stored as string — has empty spaces for new customers
    # (tenure=0 means they signed up but haven't been billed yet)
    # Steps:
    #   a. Replace empty string with NaN
    #   b. Convert to float
    #   c. Fill NaN with 0 (new customers have no total charges yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    print(f"\nTotalCharges missing values: {n_missing} → filling with 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # ── 3. Target variable ──────────────────────────────────────────────
    # Churn: "Yes" → 1 (churned), "No" → 0 (stayed)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"Churned customers: {df['Churn'].sum():,} ({df['Churn'].mean():.1%})")

    # ── 4. Binary columns (Yes/No → 1/0) ───────────────────────────────
    # These columns have exactly two values — straightforward mapping
    binary_cols = ["Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "MultipleLines"]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # Gender: Female → 0, Male → 1
    df["gender"] = (df["gender"] == "Male").astype(int)

    # ── 5. Service columns ──────────────────────────────────────────────
    # These have 3 values: "Yes", "No", "No internet service" / "No phone service"
    # We simplify: 1 if "Yes", 0 otherwise
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in service_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # Count how many services each customer has
    # This is a useful derived feature — more services = more invested
    df["num_services"] = df[service_cols].sum(axis=1)
    print(f"\nServices per customer: mean={df['num_services'].mean():.1f}  "
          f"max={df['num_services'].max()}")

    # ── 6. Contract type (one-hot encoding) ─────────────────────────────
    # Contract has 3 values — we create 2 binary columns
    # (we drop one to avoid multicollinearity — "Month-to-month" is the baseline)
    #
    #   contract_one_year  = 1 if One year,  0 otherwise
    #   contract_two_year  = 1 if Two year,  0 otherwise
    #   both 0             = Month-to-month (the default / baseline)
    df["contract_one_year"] = (df["Contract"] == "One year").astype(int)
    df["contract_two_year"] = (df["Contract"] == "Two year").astype(int)

    print(f"\nContract breakdown:")
    print(df["Contract"].value_counts().to_string())

    # ── 7. Internet service (one-hot encoding) ───────────────────────────
    # DSL is the baseline — we create 2 columns
    df["internet_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["internet_none"]  = (df["InternetService"] == "No").astype(int)

    # ── 8. Payment method (one-hot encoding) ─────────────────────────────
    # Electronic check is the baseline
    df["payment_mailed_check"]   = (df["PaymentMethod"] == "Mailed check").astype(int)
    df["payment_bank_transfer"]  = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(int)
    df["payment_credit_card"]    = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)

    # ── 9. Select final feature columns ─────────────────────────────────
    # Drop original categorical columns (replaced by encoded versions)
    # Drop customerID (identifier, not a feature)
    feature_cols = [
        # Numeric
        "tenure", "MonthlyCharges", "TotalCharges", "num_services",
        # Demographics
        "gender", "SeniorCitizen", "Partner", "Dependents",
        # Services
        "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        # Contract and billing
        "PaperlessBilling",
        "contract_one_year", "contract_two_year",
        "internet_fiber", "internet_none",
        "payment_mailed_check", "payment_bank_transfer", "payment_credit_card",
        # Target
        "Churn",
    ]

    df_clean = df[feature_cols].copy()

    # ── 10. Verify no missing values ─────────────────────────────────────
    missing = df_clean.isna().sum().sum()
    print(f"\nMissing values in clean data: {missing}")
    print(f"Final shape: {df_clean.shape}")
    print(f"\nFeature columns ({len(feature_cols)-1} features + 1 target):")
    for col in feature_cols:
        marker = " ← TARGET" if col == "Churn" else ""
        print(f"  {col}{marker}")

    # ── 11. Save ──────────────────────────────────────────────────────────
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved: {CLEAN_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    preprocess()

"""
Customer Churn API — FastAPI Application
=========================================
Serves the churn prediction model via HTTP endpoints.

Endpoints:
  GET  /health     — is the server alive and model loaded?
  GET  /model-info — model version, AUC-PR, features
  POST /predict    — send customer details, get churn probability

The API accepts human-readable customer data (contract_type="Two year")
and converts it internally to the encoded features the model expects.
Callers never need to know about one-hot encoding.

Run:
  python -m uvicorn main:app --reload --port 8000

Test:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"tenure": 2, "monthly_charges": 95.0,
         "contract_type": "Month-to-month",
         "internet_service": "Fiber optic", "num_services": 0}'
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR   = Path("models")
METRICS_PATH = MODELS_DIR / "metrics.json"

# ── Global model state ─────────────────────────────────────────────────────
# Module-level variables — loaded once at startup, reused every request
_model    = None
_scaler   = None
_features = None
_metrics  = {}


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global _model, _scaler, _features, _metrics

    print("Loading models...")
    _model    = joblib.load(MODELS_DIR / "model.joblib")
    _scaler   = joblib.load(MODELS_DIR / "scaler.joblib")
    _features = joblib.load(MODELS_DIR / "features.joblib")

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)

    print(f"Models loaded — AUC-PR: {_metrics.get('auc_pr', 'unknown')}")
    yield
    _model = _scaler = _features = None


app = FastAPI(
    title       = "Customer Churn Predictor",
    description = "Predicts churn probability for telecom customers.",
    version     = "1.0",
    lifespan    = lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

class CustomerFeatures(BaseModel):
    """
    Human-readable customer features.
    API converts these to encoded model features internally.
    """
    tenure:            int   = Field(1,    ge=0, le=72,
                                     description="Months as customer (0-72)")
    monthly_charges:   float = Field(65.0, ge=0,
                                     description="Monthly bill amount")
    contract_type:     str   = Field("Month-to-month",
                                     description="Month-to-month / One year / Two year")
    internet_service:  str   = Field("DSL",
                                     description="DSL / Fiber optic / No")
    num_services:      int   = Field(2,    ge=0, le=6,
                                     description="Number of add-on services (0-6)")
    senior_citizen:    int   = Field(0,    ge=0, le=1)
    partner:           int   = Field(0,    ge=0, le=1)
    dependents:        int   = Field(0,    ge=0, le=1)
    paperless_billing: int   = Field(1,    ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 2,
                "monthly_charges": 95.0,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "num_services": 0,
                "senior_citizen": 0,
                "partner": 0,
                "dependents": 0,
                "paperless_billing": 1,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_label:        str   = Field(..., description="HIGH / MEDIUM / LOW")
    recommendation:    str   = Field(..., description="What the retention team should do")


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_row(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert human-readable input into the 24-column feature vector.

    This is where encoding happens — the caller sends "Two year",
    we convert it to contract_one_year=0, contract_two_year=1.
    """
    row = {f: 0 for f in _features}

    # Numeric — copy directly
    row["tenure"]           = customer.tenure
    row["MonthlyCharges"]   = customer.monthly_charges
    row["SeniorCitizen"]    = customer.senior_citizen
    row["Partner"]          = customer.partner
    row["Dependents"]       = customer.dependents
    row["PaperlessBilling"] = customer.paperless_billing
    row["num_services"]     = customer.num_services
    row["TotalCharges"]     = customer.tenure * customer.monthly_charges

    # Contract type
    contract = customer.contract_type.strip().lower()
    if "one" in contract:
        row["contract_one_year"] = 1
    elif "two" in contract:
        row["contract_two_year"] = 1

    # Internet service
    internet = customer.internet_service.strip().lower()
    if "fiber" in internet:
        row["internet_fiber"] = 1
    elif internet == "no":
        row["internet_none"] = 1

    return pd.DataFrame([row])


def get_risk_label(prob: float) -> tuple[str, str]:
    """
    Convert probability to risk label and recommendation.
    Thresholds calibrated to 26.5% base churn rate.
    """
    if prob > 0.40:
        return "HIGH",   "Call immediately with retention offer"
    elif prob > 0.15:
        return "MEDIUM", "Send targeted email campaign"
    else:
        return "LOW",    "No action needed — customer is loyal"


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    """Liveness check — called by load balancers and CI/CD health checks."""
    return {
        "status":       "healthy",
        "model_loaded": _model is not None,
    }


@app.get("/model-info", tags=["System"])
def model_info():
    """Model metadata — confirm the right model is running after deployment."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(_model).__name__,
        "n_features": len(_features) if _features else 0,
        "auc_pr":     _metrics.get("auc_pr",   _metrics.get("auc", "unknown")),
        "auc_roc":    _metrics.get("auc",      "unknown"),
        "accuracy":   _metrics.get("accuracy", "unknown"),
        "C":          _metrics.get("C",        "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Accepts human-readable values — API handles encoding internally.
    Returns probability, risk label, and retention recommendation.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Convert human input → feature vector
    X = build_feature_row(customer)

    # 2. Scale (same scaler fitted on training data)
    X_scaled = _scaler.transform(X)

    # 3. Predict — predict_proba returns [[prob_stay, prob_churn]]
    prob = float(_model.predict_proba(X_scaled)[0][1])

    # 4. Convert to risk label
    risk_label, recommendation = get_risk_label(prob)

    return PredictionResponse(
        churn_probability = round(prob, 4),
        risk_label        = risk_label,
        recommendation    = recommendation,
    )

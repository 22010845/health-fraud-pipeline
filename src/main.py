import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Health Insurance Fraud Detection API")

# Load model at startup
model = joblib.load('data/xgboost_fraud_model.joblib')

class ClaimPayload(BaseModel):
    claim_amount: float
    historical_avg_amount: float
    claims_last_30_days: int
    provider_risk_score: float

@app.get("/")
def health_check():
    return {"status": "online", "system": "Health Fraud Detection Engine"}

@app.post("/predict")
def predict_fraud(claim: ClaimPayload):
    # Calculate engineered features matching src/features.py logic
    amount_to_historical_ratio = claim.claim_amount / (claim.historical_avg_amount + 1e-5)
    high_velocity_claim = 1 if claim.claims_last_30_days >= 4 else 0
    weighted_provider_risk = claim.provider_risk_score * np.log1p(claim.claim_amount)

    # Format features into DataFrame matching trained model input
    input_data = pd.DataFrame([{
        'claim_amount': claim.claim_amount,
        'historical_avg_amount': claim.historical_avg_amount,
        'claims_last_30_days': claim.claims_last_30_days,
        'provider_risk_score': claim.provider_risk_score,
        'amount_to_historical_ratio': amount_to_historical_ratio,
        'high_velocity_claim': high_velocity_claim,
        'weighted_provider_risk': weighted_provider_risk
    }])

    # Generate prediction and risk probability
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "is_fraud": prediction,
        "fraud_probability": round(probability, 4),
        "risk_level": "HIGH" if probability > 0.5 else "LOW"
    }
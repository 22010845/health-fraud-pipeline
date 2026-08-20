import streamlit as st
import requests

st.set_page_config(page_title="Health Insurance Fraud Detector", page_icon="🛡️")

st.title("🛡️ Health Insurance Fraud Detection Dashboard")
st.write("Enter claim details below to assess real-time fraud risk using our XGBoost model.")

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=1200.0, step=50.0)
        historical_avg_amount = st.number_input("Historical Avg Amount ($)", min_value=1.0, value=400.0, step=50.0)
        
    with col2:
        claims_last_30_days = st.number_input("Claims in Last 30 Days", min_value=0, value=2, step=1)
        provider_risk_score = st.slider("Provider Risk Score", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

    submit_button = st.form_submit_button("Analyze Claim")

if submit_button:
    # Payload matching FastAPI expectations
    payload = {
        "claim_amount": claim_amount,
        "historical_avg_amount": historical_avg_amount,
        "claims_last_30_days": claims_last_30_days,
        "provider_risk_score": provider_risk_score
    }
    
    try:
        # Request prediction from running FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.divider()
            
            prob = result["fraud_probability"] * 100
            
            if result["risk_level"] == "HIGH":
                st.error(f"⚠️ HIGH RISK DETECTED — Fraud Probability: {prob:.2f}%")
            else:
                st.success(f"✅ LOW RISK — Fraud Probability: {prob:.2f}%")
                
            st.json(result)
        else:
            st.error("API error. Ensure the FastAPI backend is running.")
    except Exception as e:
        st.error(f"Could not connect to FastAPI server. Make sure `python -m uvicorn src.main:app` is running! Error: {e}")
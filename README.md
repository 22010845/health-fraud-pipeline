# 🛡️ Health Insurance Fraud Detection Pipeline

An end-to-end production-grade Machine Learning pipeline and API service designed to detect fraudulent health insurance claims in imbalanced transaction datasets.

## 📌 Architecture & Tech Stack
* **Language:** Python 3.11
* **Database / Storage:** DuckDB (SQL), Pandas
* **Modeling & Imbalance Handling:** XGBoost, Scikit-Learn, SMOTE (imbalanced-learn)
* **Serving & REST API:** FastAPI, Uvicorn, Pydantic
* **Frontend Dashboard:** Streamlit
* **Containerization:** Docker

## 🚀 Quick Start Instructions

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/22010845/health-fraud-pipeline.git](https://github.com/22010845/health-fraud-pipeline.git)
cd health-fraud-pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
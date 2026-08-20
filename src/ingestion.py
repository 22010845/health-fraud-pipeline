import pandas as pd
import numpy as np
import duckdb

def generate_synthetic_claims(n_samples=5000, fraud_ratio=0.03):
    """
    Generates synthetic health insurance claim data with imbalanced fraud targets.
    """
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_clean = n_samples - n_fraud
    
    # Legitimate claims
    clean_data = {
        'claim_id': [f"CLM_{i:05d}" for i in range(n_clean)],
        'claim_amount': np.random.gamma(shape=2, scale=500, size=n_clean),
        'historical_avg_amount': np.random.gamma(shape=2, scale=450, size=n_clean),
        'claims_last_30_days': np.random.poisson(lam=1.5, size=n_clean),
        'provider_risk_score': np.random.beta(a=1, b=10, size=n_clean),
        'is_fraud': 0
    }
    
    # Fraudulent claims (higher amounts, unusual claim frequency)
    fraud_data = {
        'claim_id': [f"CLM_{i + n_clean:05d}" for i in range(n_fraud)],
        'claim_amount': np.random.gamma(shape=5, scale=1200, size=n_fraud),
        'historical_avg_amount': np.random.gamma(shape=2, scale=400, size=n_fraud),
        'claims_last_30_days': np.random.poisson(lam=5.0, size=n_fraud),
        'provider_risk_score': np.random.beta(a=5, b=2, size=n_fraud),
        'is_fraud': 1
    }
    
    df = pd.concat([pd.DataFrame(clean_data), pd.DataFrame(fraud_data)]).sample(frac=1).reset_index(drop=True)
    return df

def save_to_duckdb(df, db_path='data/claims.duckdb'):
    """
    Saves the pandas DataFrame to a local DuckDB SQL database.
    """
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS raw_claims AS SELECT * FROM df")
    conn.close()
    print(f"Data successfully written to DuckDB at: {db_path}")

if __name__ == "__main__":
    df = generate_synthetic_claims()
    df.to_csv('data/raw_claims.csv', index=False)
    save_to_duckdb(df)
    print(f"Generated {len(df)} records. Fraud distribution:\n{df['is_fraud'].value_counts()}")
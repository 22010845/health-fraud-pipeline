import pandas as pd
import numpy as np
import duckdb

def load_data_from_duckdb(db_path='data/claims.duckdb'):
    """
    Reads raw claims data directly from DuckDB using SQL.
    """
    conn = duckdb.connect(db_path)
    df = conn.execute("SELECT * FROM raw_claims").df()
    conn.close()
    return df

def build_features(df):
    """
    Creates risk ratios and interaction features for fraud detection.
    """
    # Feature 1: Ratio of current claim amount vs historical average
    df['amount_to_historical_ratio'] = df['claim_amount'] / (df['historical_avg_amount'] + 1e-5)
    
    # Feature 2: High velocity claim indicator (frequent claims in 30 days)
    df['high_velocity_claim'] = (df['claims_last_30_days'] >= 4).astype(int)
    
    # Feature 3: Provider risk weighted by claim size
    df['weighted_provider_risk'] = df['provider_risk_score'] * np.log1p(df['claim_amount'])
    
    return df

def save_processed_data(df, output_path='data/processed_claims.csv'):
    """
    Saves transformed feature set to CSV for model training.
    """
    df.to_csv(output_path, index=False)
    print(f"Features created successfully! Processed dataset saved to: {output_path}")

if __name__ == "__main__":
    raw_df = load_data_from_duckdb()
    featured_df = build_features(raw_df)
    save_processed_data(featured_df)
    print(f"Dataset shape with features: {featured_df.shape}")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_fraud_model(data_path='data/processed_claims.csv'):
    """
    Loads processed feature data, handles class imbalance with SMOTE,
    trains an XGBoost model, and saves the trained model artifact.
    """
    # 1. Load data
    df = pd.read_csv(data_path)
    
    # 2. Separate features (X) and target variable (y)
    drop_cols = ['claim_id', 'is_fraud']
    X = df.drop(columns=drop_cols)
    y = df['is_fraud']
    
    # 3. Train/Test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Handle imbalance on training set using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 5. Train XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_res, y_train_res)
    
    # 6. Save trained model
    joblib.dump(model, 'data/xgboost_fraud_model.joblib')
    print("Model training complete! Model saved to data/xgboost_fraud_model.joblib")
    
    # Save test sets for evaluation script
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print("Saved evaluation test datasets to data/ directory.")

if __name__ == "__main__":
    train_fraud_model()
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_model():
    """
    Loads the trained model and test datasets to calculate performance metrics.
    """
    # 1. Load model and test data
    model = joblib.load('data/xgboost_fraud_model.joblib')
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()

    # 2. Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Calculate and print performance metrics
    print("=== MODEL EVALUATION REPORT ===")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Legitimate flagged as Legitimate): {cm[0][0]}")
    print(f"False Positives (Legitimate flagged as Fraud): {cm[0][1]}")
    print(f"False Negatives (Fraud missed): {cm[1][0]}")
    print(f"True Positives (Fraud correctly flagged): {cm[1][1]}")

if __name__ == "__main__":
    evaluate_model()
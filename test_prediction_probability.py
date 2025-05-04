import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("models/final_xgboost_model.pkl")

# Define input as DataFrame
X_input = pd.DataFrame(
    [
        {
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 75.5,
            "is_new_customer": 1,
            "is_loyal_customer": 0,
            "tenure_monthly_ratio": 0.159,
            "high_charge_short_tenure": 0,
        }
    ]
)

# Predict probability of churn
proba = model.predict_proba(X_input)[0][1]

# Final threshold
threshold = 0.56
churn = int(proba >= threshold)

# Output
print(f"Churn probability: {proba:.3f}")
print(f"Prediction (threshold={threshold}): {'CHURN' if churn else 'NO CHURN'}")

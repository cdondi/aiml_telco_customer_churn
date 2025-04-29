"""
•	Load cleaned_7k.csv (or cleaned_1M.csv)
•	Train a baseline ML model (e.g., Logistic Regression or Random Forest)
•	Save the trained model to models/ folder as churn_model.pkl
•	Optionally output a metrics.json file with evaluation metrics
To run this script from the command line:
    python utils/train_model.py --input data/telco_cleaned_7k.csv --model_output models/telco_churn_model_7k.pkl --metrics_output metrics/telco_metrics_7k.json
"""

import pandas as pd
import argparse
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(input_path, model_output_path, metrics_output_path):
    # Load the preprocessed dataset
    df = pd.read_csv(input_path)

    # Basic feature selection (drop non-numeric or irrelevant columns)
    X = df.select_dtypes(include=["number"])

    # Drop customerID if it is numeric and included
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a simple Logistic Regression model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Save model
    joblib.dump(model, model_output_path)

    # Save metrics
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {model_output_path}")
    print(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a churn prediction model")
    parser.add_argument("--input", required=True, help="Path to input cleaned CSV")
    parser.add_argument(
        "--model_output", required=True, help="Path to save trained model"
    )
    parser.add_argument(
        "--metrics_output", required=True, help="Path to save evaluation metrics JSON"
    )

    args = parser.parse_args()

    train_model(args.input, args.model_output, args.metrics_output)

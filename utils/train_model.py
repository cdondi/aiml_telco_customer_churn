"""
Train a churn prediction model using Logistic Regression.

- Load cleaned_7k.csv (or cleaned_1M.csv)
- Train a Logistic Regression model
- Save model to models/ as telco_churn_model_7k.pkl
- Save metrics (accuracy, precision, recall, f1) to metrics/telco_metrics_7k.json
- Log all parameters, metrics, and model using MLflow

Usage:
    python utils/train_model.py \
        --input data/telco_cleaned_7k.csv \
        --model_output models/telco_churn_model_7k.pkl \
        --metrics_output metrics/telco_metrics_7k.json
"""

import argparse
import json
import joblib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(input_path, model_output_path, metrics_output_path, hyper_params, model_type="logistic"):
    # Load the preprocessed dataset
    df = pd.read_csv(input_path)

    # Basic feature selection (drop non-numeric or irrelevant columns)
    X = df.select_dtypes(include=["number"])

    # Drop customerID if it is numeric and included
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X = X.astype({col: "float64" for col in X.select_dtypes(include=["int"]).columns})

    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a simple Logistic Regression model
    if model_type == "logistic":
        model = LogisticRegression(**hyper_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**hyper_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    joblib.dump(model, model_output_path)

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Save metrics
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Model saved to %s", model_output_path)
    logger.info("Metrics saved to %s", metrics_output_path)

    return model, metrics, X_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a churn prediction model")
    parser.add_argument("--input", required=True, help="Path to input cleaned CSV")
    parser.add_argument("--model_output", required=True, help="Path to save trained model")
    # fmt: off
    parser.add_argument("--metrics_output", required=True, help="Path to save evaluation metrics JSON")
    # fmt: on
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--penalty", type=str, default="l2")

    args = parser.parse_args()

    # Convert string "None" to actual None
    penalty = None if args.penalty.lower() == "none" else args.penalty

    hyper_params = {
        "max_iter": args.max_iter,
        "penalty": penalty,
        "solver": "lbfgs",  # You can expand later
        "class_weight": "balanced",
    }

    # Group all experiments under one project
    mlflow.set_experiment("telco-customer-churn")

    # # Start MLflow run
    # mlflow.start_run()

    try:
        # Log parameters
        for k, v in hyper_params.items():
            mlflow.log_param(k, v)

        # Train the model
        model, metrics, X_test = train_model(args.input, args.model_output, args.metrics_output, hyper_params)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model with input_example and inferred signature
        input_example = X_test.iloc[:1]
        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

        mlflow.set_tag("status", "success")
    except Exception as e:
        mlflow.set_tag("status", "failed")
        mlflow.log_param("error_message", str(e))
        raise
    finally:
        mlflow.end_run()

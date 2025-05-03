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
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(input_path, model_output_path, metrics_output_path, hyper_params, model_type="xgboost", use_resampling=False, threshold=0.5):
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

    if use_resampling:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train a simple Logistic Regression model
    if model_type == "logistic":
        model = LogisticRegression(**hyper_params)
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**hyper_params)
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
    elif model_type == "xgboost":
        # Compute class imbalance for XGBoost
        counter = Counter(y)
        neg, pos = counter[0], counter[1]
        hyper_params["scale_pos_weight"] = neg / pos

        # Split into train and validation sets for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Fit with early stopping
        model = XGBClassifier(**hyper_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Save validation data for threshold tuning
        X_val.to_csv("data/xgb_best_x_val.csv", index=False)
        pd.DataFrame(y_val).to_csv("data/xgb_best_y_val.csv", index=False)

        # Predict probabilities instead of labels
        y_probs = model.predict_proba(X_val)[:, 1]  # Probability of class "1" (churn)

        # Apply custom threshold
        y_pred = (y_probs >= threshold).astype(int)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    joblib.dump(model, model_output_path)

    # Save metrics. hyper_params is required for threshold sweep later
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    # Save metrics
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Model saved to %s", model_output_path)
    logger.info("Metrics saved to %s", metrics_output_path)

    return model, metrics, X_test, X_val, y_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a churn prediction model")
    parser.add_argument("--final_model", action="store_true", help="Train and save the final selected model")
    # parser.add_argument("--input", required=True, help="Path to input cleaned CSV")
    # parser.add_argument("--model_output", required=True, help="Path to save trained model")
    # # fmt: off
    # parser.add_argument("--metrics_output", required=True, help="Path to save evaluation metrics JSON")
    # # fmt: on
    # parser.add_argument("--max_iter", type=int, default=1000)
    # parser.add_argument("--penalty", type=str, default="l2")

    args = parser.parse_args()

    if args.final_model:
        from xgboost import XGBClassifier
        import joblib
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import mlflow

        # Load full training dataset
        df = pd.read_csv("data/telco_cleaned_7k.csv")

        # Split and prepare data
        X = df.select_dtypes(include=["number"])

        # Drop customerID if it is numeric and included
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        # Python cannot represent missing integer values
        X = X.astype({col: "float64" for col in X.select_dtypes(include=["int"]).columns})

        y = df["Churn"].map({"Yes": 1, "No": 0})

        # Best hyperparameters (from Optuna + threshold sweep)
        best_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.035,
            "subsample": 0.794,
            "colsample_bytree": 0.998,
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "early_stopping_rounds": 10,
        }

        # Compute class imbalance for XGBoost
        counter = Counter(y)
        neg, pos = counter[0], counter[1]
        best_params["scale_pos_weight"] = neg / pos

        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Save test set for model registration and deployment
        X_test.to_csv("data/xgb_best_x_test.csv", index=False)
        y_test.to_csv("data/xgb_best_y_test.csv", index=False)

        threshold = 0.56  # Best threshold from sweep

        # Set up MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("xgboost-final-model")

        with mlflow.start_run():
            # Log parameters
            for k, v in best_params.items():
                if k != "early_stopping_rounds":  # Not accepted by log_param
                    mlflow.log_param(k, v)
            mlflow.log_param("threshold", threshold)

            # Train model
            model = XGBClassifier(**best_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            # Evaluate
            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs >= threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metrics({"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1})

            # Log model with input_example and inferred signature
            input_example = X_test.iloc[:1]
            signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

            # Save model
            joblib.dump(model, "models/final_xgboost_model.pkl")
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

            print("✅ Final model saved and logged to MLflow.")

    else:
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

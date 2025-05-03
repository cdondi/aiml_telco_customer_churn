# threshold_sweep_runner.py

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import os

# Load best model and its validation data
model = joblib.load("models/resampled/best_xgboost_model.pkl")
X_val = pd.read_csv("data/xgb_best_x_val.csv")
y_val = pd.read_csv("data/xgb_best_y_val.csv").squeeze()

# Sweep thresholds
thresholds = np.round(np.arange(0.45, 0.561, 0.01), 2)

# Set MLflow experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("xgboost-resampling-threshold-sweep")

# Sweep through thresholds and log metrics
for thresh in thresholds:
    with mlflow.start_run():
        mlflow.set_tag("sweep_type", "threshold_only")
        y_probs = model.predict_proba(X_val)[:, 1]
        y_pred = (y_probs >= thresh).astype(int)

        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        mlflow.log_param("threshold", round(thresh, 2))
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

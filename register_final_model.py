# register_final_model.py

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from mlflow.models.signature import infer_signature

# Load final model
model = joblib.load("models/final_xgboost_model.pkl")

# Load test data
X_test = pd.read_csv("data/xgb_best_x_test.csv")
y_test = pd.read_csv("data/xgb_best_y_test.csv").squeeze()

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Apply final threshold (adjust if your best is different)
threshold = 0.56
y_pred = (y_probs >= threshold).astype(int)

# Evaluate
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Log everything with MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-final-xgboost")

with mlflow.start_run(run_name="final_xgboost_registration"):
    mlflow.log_param("threshold", threshold)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Infer model signature and log model
    input_example = X_test.iloc[:1]
    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature, registered_model_name="xgboost-final-churn-model")

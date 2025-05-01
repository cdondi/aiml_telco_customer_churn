import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
from utils.train_model import train_model
import os

mlflow.set_tracking_uri("http://localhost:5000")
model_type = "random_forest"  # or "logistic"


def get_output_paths(model_type: str, trial_number: int):
    model_dir = os.path.join("models", model_type)
    metrics_dir = os.path.join("metrics", model_type)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model_output_path = os.path.join(model_dir, f"model_optuna_{trial_number}.pkl")
    metrics_output_path = os.path.join(metrics_dir, f"metrics_optuna_{trial_number}.json")
    return model_output_path, metrics_output_path


def objective(trial):
    # Define hyperparameters to search
    max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", None])
    solver = "liblinear" if penalty == "l1" else "lbfgs"

    # Group all experiments under one project
    mlflow.set_experiment("telco-customer-churn-optuna")

    # Prepare model parameters
    if model_type == "logistic":
        mlflow.set_experiment("churn-optuna-logreg")
        max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", None])
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        hyper_params = {
            "max_iter": max_iter,
            "penalty": penalty,
            "solver": solver,
            "class_weight": "balanced",
        }

    elif model_type == "random_forest":
        mlflow.set_experiment("churn-optuna-random-forest")
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=50)
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        hyper_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "class_weight": "balanced",
            "random_state": 42,
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    mlflow.log_param("model_type", model_type)
    model_output_path, metrics_output_path = get_output_paths(model_type, trial.number)

    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run
    mlflow.start_run()
    try:
        # Log hyperparameters
        for k, v in hyper_params.items():
            mlflow.log_param(k, v)

        # Train model and evaluate
        model, metrics, X_test = train_model(
            input_path="data/telco_cleaned_7k.csv",
            model_output_path=f"models/model_optuna_{max_iter}_{penalty}.pkl",
            metrics_output_path=f"metrics/metrics_optuna_{max_iter}_{penalty}.json",
            hyper_params=hyper_params,
            model_type=model_type,
        )

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model with input example and signature
        input_example = X_test.iloc[:1]
        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

        return metrics["f1_score"]

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=6)

    print("Best trial:")
    print(study.best_trial)

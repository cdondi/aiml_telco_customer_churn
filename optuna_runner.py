import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from utils.train_model import train_model
from xgboost import XGBClassifier


mlflow.set_tracking_uri("http://localhost:5000")
model_type = "random_forest"  # or "logistic" or "xgboost"
model_type = "xgboost"


def get_output_paths(model_type: str, trial_number: int, use_resampling=False):
    base_folder = "resampled" if use_resampling else ""
    model_dir = os.path.join("models", base_folder)
    metrics_dir = os.path.join("metrics", base_folder)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"model_optuna_{model_type}_{trial_number}.pkl")
    metrics_path = os.path.join(metrics_dir, f"metrics_optuna_{model_type}_{trial_number}.json")
    return model_path, metrics_path


def objective(trial):
    use_resampling = True  # ← toggle this ON for SMOTE, OFF for standard runs

    # Define hyperparameters to search
    max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", None])
    solver = "liblinear" if penalty == "l1" else "lbfgs"

    if use_resampling == True:
        append_exp_group = "resampling"
    else:
        append_exp_group = "regular"

    # Prepare model parameters
    if model_type == "logistic":
        # Group all experiments under one project
        mlflow.set_experiment(f"churn-logistic-optuna-{append_exp_group}")

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
        # Group all experiments under one project
        mlflow.set_experiment(f"churn-random-forest-optuna-{append_exp_group}")

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

    elif model_type == "xgboost":
        # Group all experiments under one project
        mlflow.set_experiment(f"churn-xgboost-optuna-{append_exp_group}")

        n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300])
        max_depth = trial.suggest_categorical("max_depth", [3, 5, 10, 15])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        hyper_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "binary:logistic",
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        for k, v in hyper_params.items():
            mlflow.log_param(k, v)

        model = XGBClassifier(**hyper_params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    mlflow.log_param("model_type", model_type)
    model_output_path, metrics_output_path = get_output_paths(model_type, trial.number, use_resampling)

    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run
    mlflow.start_run()

    # Tag the run for clarity
    mlflow.set_tag("resampled", str(use_resampling))

    try:
        # Log hyperparameters
        for k, v in hyper_params.items():
            mlflow.log_param(k, v)

        # Train model and evaluate
        model, metrics, X_test = train_model(
            input_path="data/telco_cleaned_7k.csv",
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
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

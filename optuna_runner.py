import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
from utils.train_model import train_model

mlflow.set_tracking_uri("http://localhost:5000")


def objective(trial):
    # Define hyperparameters to search
    max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", None])
    solver = "liblinear" if penalty == "l1" else "lbfgs"

    # Prepare model parameters
    logreg_params = {
        "max_iter": max_iter,
        "penalty": penalty,
        "solver": solver,
        "class_weight": "balanced",
    }

    # Group all experiments under one project
    mlflow.set_experiment("telco-customer-churn-optuna")

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        for k, v in logreg_params.items():
            mlflow.log_param(k, v)

        # Train model and evaluate
        model, metrics, X_test = train_model(
            input_path="data/telco_cleaned_7k.csv",
            model_output_path=f"models/model_optuna_{max_iter}_{penalty}.pkl",
            metrics_output_path=f"metrics/metrics_optuna_{max_iter}_{penalty}.json",
            logreg_params=logreg_params,
        )

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model with input example and signature
        input_example = X_test.iloc[:1]
        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

        return metrics["f1_score"]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=6)

    print("Best trial:")
    print(study.best_trial)

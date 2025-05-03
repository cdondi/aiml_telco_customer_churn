# create_final_metadata.py
import json

final_metrics = {"accuracy": 0.767, "precision": 0.547, "recall": 0.703, "f1_score": 0.615}

final_params = {"model_type": "xgboost", "n_estimators": 300, "max_depth": 5, "learning_rate": 0.035, "subsample": 0.794, "colsample_bytree": 0.998, "objective": "binary:logistic", "eval_metric": "logloss", "use_label_encoder": False, "random_state": 42, "threshold": 0.56}
# "early_stopping_rounds": 10,

with open("metrics/final_xgboost_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=4)

with open("params/final_xgboost_params.json", "w") as f:
    json.dump(final_params, f, indent=4)

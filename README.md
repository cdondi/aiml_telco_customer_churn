# Customer Churn Prediction System

This project is an end-to-end machine learning system to predict customer churn for a subscription-based business. The goal is to identify users who are likely to cancel their service soon so that retention teams can take proactive action.

## Getting Started

1. Clone this repository.
2. Place the raw dataset in the `data/` folder.
3. Run the preprocessing and dataset simulation scripts:
    - `python utils/preprocess_data.py --input data/telco_7k.csv --output data/telco_cleaned_7k.csv`
    - `python utils/simulate_large_dataset.py --input data/telco_cleaned_7k.csv --output data/telco_cleaned_1M.csv --multiplier 150`
4. Use `train_model.py` for standalone training and `optuna_runner.py` for automated hyperparameter tuning.

---

## Problem Statement

Churn is a critical metric for any recurring revenue business. This project uses historical customer data to build a predictive model that flags accounts with high churn risk. Accurately identifying these accounts enables targeted interventions, improving customer lifetime value and reducing revenue loss.

---

## Dataset

- **Source:** IBM Telco Churn Dataset (Wide Format from Hugging Face)
- **Features include:**
  - Demographics (gender, seniority)
  - Contract and payment type
  - Service features (Streaming, Internet, Phone)
  - Tenure, Monthly Charges, Total Charges
  - Churn Label

---

## Project Structure

```
churn-prediction/
├── data/                # Raw, cleaned, and simulated data
├── notebooks/           # Jupyter notebooks for EDA and visualization
├── models/              # Trained model files (.pkl)
├── metrics/             # Model evaluation outputs (JSON)
├── utils/               # Scripts (training, preprocessing, simulation)
├── mlruns/              # MLflow tracking logs
├── optuna_runner.py     # Hyperparameter tuning entry point
├── train_model.py       # Train a model manually with arguments
├── dvc.yaml             # DVC pipeline definition
├── params.yaml          # Default parameter file
├── README.md
```

---

## Features Used

- Tenure buckets and binary flags
- One-hot encoding for categorical features
- Feature interactions (tenure/charges)
- Resampling for class imbalance
- Threshold tuning (in progress)

---

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

Primary focus is on **recall** and **F1-score** due to class imbalance and business priority to reduce false negatives.

---

## Tools Used

- Python (Pandas, Scikit-learn, XGBoost)
- MLflow (Experiment tracking)
- Optuna (Hyperparameter tuning)
- DVC + S3 (Data and model versioning)
- Matplotlib/Seaborn (Visualization)
- Joblib (Model serialization)
- Git (Code version control)

---

## Class Imbalance Handling

- `class_weight="balanced"` for scikit-learn models
- Resampling (SMOTE + RandomUnderSampler)
- Will explore `scale_pos_weight` in XGBoost
- Threshold tuning planned to improve precision

---

## Model Training Summary

- `train_model.py`: Standard model training (Logistic, Random Forest, XGBoost)
- `optuna_runner.py`: Modular hyperparameter tuning with Optuna + MLflow
- MLflow logs all hyperparams, metrics, and artifacts
- Models are stored in `models/`, metrics in `metrics/`

---

## Current Pipeline Design Summary

- **train_model.py**
  - Core logic: load data, train, evaluate, and return model + metrics
- **optuna_runner.py**
  - Automates hyperparameter sweeps using Optuna
  - Logs experiments with MLflow
- **threshold_sweep_runner.py**
  - Runs post-hoc threshold optimization to improve precision/recall balance
- **MLflow + DVC**
  - Full reproducibility and traceability of experiments

---

## Model Saving & Loading

```python
import joblib

# Save
joblib.dump(model, 'models/model_optuna_xgboost_<trial>.pkl')

# Load
model = joblib.load('models/model_optuna_xgboost_<trial>.pkl')
```

---

## Model Selection Rationale

After evaluating three algorithms — Logistic Regression, Random Forest, and XGBoost — across precision, recall, and F1-score (with class imbalance addressed via resampling), **XGBoost emerged as the preferred model**. It demonstrated the best balance of precision (~0.50), recall (~0.74), and F1-score (~0.60), making it more suitable for real-world deployment where both accurate churn detection and cost-efficiency matter. While Logistic Regression achieved slightly higher recall, it did so at the cost of precision. Random Forest underperformed in both metrics.

Next steps will focus on optimizing XGBoost via **threshold tuning**, followed by optional **feature engineering** or alternate imbalance strategies like `scale_pos_weight`.

---

## Model Performance Comparison (With Resampling)

| Model               | Accuracy | Precision  | Recall     | F1-Score   | Notes                             |
|---------------------|----------|------------|------------|------------|-----------------------------------|
| Logistic Regression | ~0.725   | ~0.489     | **~0.789** | **~0.604** | Highest recall, lower precision   |
| Random Forest       | ~0.754   | ~0.534     | ~0.572     | ~0.552     | Lower recall and F1               |
| XGBoost             | ~0.737   | **~0.503** | ~0.743     | **~0.602** | Best balance of all metrics       |



| Model               | Accuracy | Precision  | Recall     | F1-Score   | Notes                              |
|---------------------|----------|------------|------------|------------|------------------------------------|
| Logistic Regression | ~0.725   | ~0.489     | **~0.789** | ~0.604     | High recall, lower precision       |
| Random Forest       | ~0.754   | ~0.534     | ~0.572     | ~0.552     | Weaker on recall and F1            |
| XGBoost (tuned)     | ~0.743   | **0.534** |  ~0.714     | **0.611**  | Best balance after threshold tuning|


## Threshold Tuning and Selection
To improve the precision-recall tradeoff of this XGBoost model, I implemented threshold tuning using the best hyperparameter configuration from Optuna. Instead of relying on the default threshold of 0.50 for classification, I swept thresholds between 0.45 and 0.56 (inclusive) and evaluated model performance on validation data.
## Sweep Configuration
	-	Algorithm: XGBoost (with SMOTE resampling)
	-	Validation data: Held-out from training set
	-	Threshold range: 0.45 to 0.56
	-	Step size: 0.01
	-	Evaluation metrics: Precision, Recall, F1-score (via MLflow)

### Threshold Tuning Results
| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|----------|----------|
| 0.45      | 0.44      | 0.83     | 0.57     |
| 0.46      | 0.45      | 0.81     | 0.58     |
| 0.47      | 0.46      | 0.79     | 0.58     |
| 0.48      | 0.47      | 0.77     | 0.58     |
| 0.49      | 0.48      | 0.75     | 0.58     |
| 0.50      | 0.47      | 0.76     | 0.58     |
| 0.51      | 0.48      | 0.75     | 0.59     |
| 0.52      | 0.49      | 0.74     | 0.59     |
| 0.53      | 0.50      | 0.74     | 0.59     |
| 0.54      | 0.49      | 0.75     | 0.59     |
| 0.55      | 0.50      | 0.74     | 0.60     |
| **0.56**  | **0.51**  | **0.73** | **0.60** |

I ended up selecting 0.56 as the best threshold. Why 0.56 ?
	-	At threshold = 0.56, the model achieved the best F1-score (~0.60) with improved precision (~0.51) and strong recall (~0.73).
	-	This threshold better aligns with real-world scenarios where false positives (wrongly flagging non-churners) can be costly.
## Outcome
  - The selected threshold of 0.56 was hard-coded into the final model serving logic.
	-	This change ensures that predictions from the deployed model are aligned with the desired business tradeoffs between precision and recall.

---
## Model Saving & Loading
import joblib
# Save
joblib.dump(model, 'models/final_xgboost_model.pkl')
# Load
model = joblib.load('models/final_xgboost_model.pkl')


## Model Serving (Local API)
To serve the final model locally using MLflow:
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/telco-xgboost-final/1" -p 1234


## To send a sample prediction request (ensure your input matches the trained feature set) depending on the use case:
{
    "dataframe_records": [
      {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 75.5,
        "TotalCharges": 905.5,
        "InternetService_Fiber optic": 1,
        "InternetService_No": 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_Yes": 1,
        "DeviceProtection_Yes": 1,
        "TechSupport_Yes": 0,
        "StreamingTV_Yes": 1,
        "StreamingMovies_Yes": 1,
        "Contract_One year": 0,
        "Contract_Two year": 1,
        "PaperlessBilling_Yes": 1,
        "PaymentMethod_Electronic check": 0,
        "PaymentMethod_Mailed check": 1,
        "is_new_customer": 1,
        "is_loyal_customer": 0,
        "tenure_monthly_ratio": 0.159,
        "high_charge_short_tenure": 0
      }
    ]
  }

  OR

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

OR

run python test_api.py 
OR
python test_prediction_probability.py


## Deployment Readiness

XGBoost with resampling and tuned threshold offers the best compromise between detecting churn and minimizing false positives. 
Threshold tuning improved precision from 0.503 to 0.534 with a small drop in recall.

**Files to deploy:**
- `models/resampled/best_xgboost_model.pkl`
- `data/xgb_best_x_test.csv`
- `data/xgb_best_y_test.csv`
---

## Future Enhancements

- Add real-time inference API (FastAPI or Flask)
- Streamlined deployment with Docker and CI/CD
- Integrate dashboard (Grafana, Streamlit, etc.)
- SHAP values for model explainability
- Extend to time-series churn prediction
- Add a dashboard to adjust and visualize thresholds dynamically.
  	•	Lower thresholds increase recall but may reduce precision.
	  •	Higher thresholds increase precision but may reduce recall.
	  •	In production, this threshold can be tuned to business priorities and costs.

---

## Author

Clive Dondi  
AI/ML Engineer & Software Developer  
Contact: clivedondi@hotmail.com

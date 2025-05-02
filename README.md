# Customer Churn Prediction System

This project is an end-to-end machine learning system to predict customer churn for a subscription-based business. The goal is to identify users who are likely to cancel their service soon so that retention teams can take proactive action.

## To Get Started

1. Clone this repo or create it locally.
2. Place your raw dataset in the `data/` folder.
3. Start exploring in `notebooks/`.
4. Build your model in `churn_model.py`.

Happy modeling!

---

## Problem Statement

Churn is a critical metric for any recurring revenue business. This project uses historical customer data to build a predictive model that flags accounts with high churn risk. Accurately identifying these accounts enables targeted interventions, improving customer lifetime value and reducing revenue loss.

---

## Dataset

- **Source:** [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Fields include:**
  - Demographic info (gender, age)
  - Account tenure
  - Payment method
  - Contract type
  - Monthly charges, total charges
  - Service usage (internet, phone, streaming)
  - Churn flag

---

## Project Structure

churn-prediction/
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── models/ # Trained model artifacts (.pkl)
├── sql/ # SQL queries for data extraction and transformation
├── app/ # Optional web app interface (Streamlit/FastAPI)
├── utils/ # Helper scripts and generators
├── churn_model.py # Main training script
└── project_roadmap.md # Project roadmap
└── README.md # Project overview

---

## Features Used

- Days since last login
- Monthly average spend
- % drop in usage
- # of support tickets
- Contract type, tenure
- Categorical encodings (one-hot, label)
- Scaled numerical features

---

## Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## Evaluation Metrics

- Accuracy
- Precision / Recall
- F1 Score
- ROC-AUC

Special focus is given to **recall** and **F1-score**, due to the imbalanced nature of churn prediction.

---

## Tools & Tech

- **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
- **SQL** (for preprocessing and joining multi-table data)
- **Matplotlib / Seaborn** (for data visualization)
- **Joblib** (for saving models)
- **Streamlit** _(optional)_ for deployment
- **DVC** _(optional)_ for versioning data & models

---

## Handling Class Imbalance

- Applied `class_weight='balanced'`
- Experimented with `SMOTE` and `undersampling`
- Evaluated metrics beyond accuracy

---

## Model Saving & Loading

```python
import joblib

# Save
joblib.dump(model, 'models/churn_model.pkl')

# Load
model = joblib.load('models/churn_model.pkl')
```

## Current Pipeline Design Summary
- train_model.py
  - Core logic (data -> model + metrics)
  - Tools used - Python, scikit-learn
- optuna_runner.py
  - Experiment orchestration + hyperopt
  - Tools used - Optuna, MLflow
- DVC
  - Dataset and model versioning
  - Tools used - DVC, S3
- MLflow
  - Experiment tracking, metrics, models
  - - Tools used - MLflow UI + logs
- Git
  - Code versionong


📡 Future Enhancements
• Add a real-time inference API using FastAPI
• Automate model training with MLflow or DVC pipelines
• Integrate with business dashboards
• Incorporate time-series forecasting for churn timing

### Model Training Updates
After evaluating three algorithms — Logistic Regression, Random Forest, and XGBoost — across precision, recall, and F1-score (with class imbalance addressed via resampling), XGBoost emerged as the preferred model. It demonstrated the best balance of precision (~0.50), recall (~0.74), and F1-score (~0.60), making it more suitable for real-world deployment where both accurate churn detection and cost-efficiency matter. While Logistic Regression achieved slightly higher recall, it did so at the cost of precision. Random Forest underperformed in both metrics.

Next steps will focus on optimizing XGBoost via threshold tuning to improve precision without retraining, followed by optional feature engineering or alternate imbalance strategies like scale_pos_weight. All models were tracked using MLflow and versioned with DVC for full reproducibility.

### Model Performance Comparison (With Resampling)

| Model               | Accuracy | Precision | Recall | F1-Score | Notes                                 |
|---------------------|----------|-----------|--------|----------|---------------------------------------|
| Logistic Regression | ~0.725   | ~0.489    | **~0.789** | **~0.604** | Highest recall, lower precision |
| Random Forest       | ~0.754   | ~0.534    | ~0.572 | ~0.552   | Lower recall and F1                   |
| XGBoost             | ~0.737   | **~0.503** | ~0.743 | **~0.602** | Best balance of all metrics        |


# More Information
For more details on EDA and model training, Please take a look at project_roadmap.md

Built by Clive Dondi, AI/ML Engineer & Software Developer
Contact: clivedondi@hotmail.com

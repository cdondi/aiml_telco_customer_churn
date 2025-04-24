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

📡 Future Enhancements
• Add a real-time inference API using FastAPI
• Automate model training with MLflow or DVC pipelines
• Integrate with business dashboards
• Incorporate time-series forecasting for churn timing

Built by Clive Dondi, AI/ML Engineer & Software Developer
Contact: clivedondi@hotmail.com

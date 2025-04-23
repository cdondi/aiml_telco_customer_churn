import os

# Use the current directory
root_dir = os.getcwd()

# Folders to create
folders = ["data", "notebooks", "models", "sql", "app", "utils"]

# Files and their contents
files = {
    "requirements.txt": """
pandas>=1.5.0
numpy>=1.21
scikit-learn>=1.1
xgboost>=1.6
matplotlib>=3.5
seaborn>=0.11
joblib>=1.1
jupyterlab>=3.0
notebook>=6.4
sqlalchemy>=1.4
psycopg2-binary>=2.9
streamlit>=1.15
imbalanced-learn>=0.10
""",
    "churn_model.py": """# Entry point for training the churn prediction model

if __name__ == '__main__':
    print('Train churn model here.')
""",
}

# Create subfolders
for folder in folders:
    os.makedirs(os.path.join(root_dir, folder), exist_ok=True)

# Create root-level files
for filename, content in files.items():
    with open(os.path.join(root_dir, filename), "w") as f:
        f.write(content.strip())

print("Project structure scaffolded successfully (without README).")

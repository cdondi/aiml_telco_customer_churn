"""
•	Reads ../data/telco_cleaned_7k.csv
        •	Creates:
            •	tenure buckets
            •	binary flags
            •	interaction features
        •	Outputs:
                •	../data/telco_cleaned_7k.csv
                •	../data/telco_cleaned_1M.csv (once the 1M dataset is generated)
"""

import pandas as pd
import argparse


def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Basic cleaning: drop rows with missing target or critical columns if any
    df.dropna(subset=["Churn"], inplace=True)

    # Tenure Buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 24, 48, 72],
        labels=["0–6m (High Risk)", "6–24m", "24–48m", "48–72m"],
    )

    # Binary Flags
    df["is_new_customer"] = (df["tenure"] < 6).astype(int)
    df["is_loyal_customer"] = (df["tenure"] > 60).astype(int)

    # Interaction Features
    df["tenure_monthly_ratio"] = df["tenure"] / df["MonthlyCharges"]
    df["high_charge_short_tenure"] = (
        (df["MonthlyCharges"] > 80) & (df["tenure"] < 6)
    ).astype(int)

    # Save preprocessed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telco churn data")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")

    args = parser.parse_args()
    preprocess_data(args.input, args.output)

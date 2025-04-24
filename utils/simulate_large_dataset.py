"""
•	Reads ../data/telco_cleaned_7k.csv
        •	This code :
                •	Adds slight noise to MonthlyCharges
                •	Shuffles the dataset
                •	Scales up to ~1 million rows
        •	Outputs:
                •	../data/telco_cleaned_1M.csv (once the 1M dataset is generated)
"""

import pandas as pd
import argparse
import numpy as np


def simulate_large_dataset(input_path, output_path, multiplier):
    df = pd.read_csv(input_path)

    # Repeat the dataset
    df_large = pd.concat([df] * multiplier, ignore_index=True)

    # Optionally add slight noise to numeric fields (like MonthlyCharges)
    if "MonthlyCharges" in df_large.columns:
        noise = np.random.normal(loc=0.0, scale=2.0, size=len(df_large))
        df_large["MonthlyCharges"] = df_large["MonthlyCharges"] + noise
        df_large["MonthlyCharges"] = df_large["MonthlyCharges"].clip(lower=0)

    # Shuffle the rows
    df_large = df_large.sample(frac=1).reset_index(drop=True)

    # Save the simulated dataset
    df_large.to_csv(output_path, index=False)
    print(f"Simulated dataset with {len(df_large):,} rows saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a larger version of a dataset for streaming practice"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=150,
        help="Factor to expand the dataset (default: 150)",
    )

    args = parser.parse_args()
    simulate_large_dataset(args.input, args.output, args.multiplier)

#!/bin/bash

echo "🔁 Running MLflow experiment sweep..."

# Define parameter combinations
declare -a max_iters=(500 1000 2000)
declare -a penalties=("l2" "none")

for max_iter in "${max_iters[@]}"
do
  for penalty in "${penalties[@]}"
  do
    echo "🚀 Running with max_iter=$max_iter, penalty=$penalty"

    python utils/train_model.py \
      --input data/telco_cleaned_7k.csv \
      --model_output models/model_${max_iter}_${penalty}.pkl \
      --metrics_output metrics/metrics_${max_iter}_${penalty}.json \
      --max_iter $max_iter \
      --penalty $penalty
  done
done

echo "✅ Sweep complete. View results at http://localhost:5000"
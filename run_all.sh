#!/bin/bash

echo "Running all experiments..."

python experiments/exp1_distribution_shift.py
python experiments/exp2_output_only_failure.py
python experiments/exp3_logging_ablation.py
python experiments/exp4_exact_fiber.py

echo "Done. Results saved in /results and /plots"
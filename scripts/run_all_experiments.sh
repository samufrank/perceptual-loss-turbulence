#!/bin/bash
# Run all Part 4 experiments sequentially

set -e

export PYTHONUNBUFFERED=1

echo "=== Starting experiments at $(date) ==="
 
echo "============================================"
echo "Part 4: Turbulence-Robust Feature Learning"
echo "Running all experiments"
echo "============================================"
echo ""

# Experiment 3: Spatial Only
echo "============================================"
echo "Experiment 3: Spatial Loss Only"
echo "Output: results/part4/exp3_spatial_only/"
echo "============================================"
python train_encoder.py --config configs/exp3_spatial_only.yaml
echo ""
echo "Experiment 3 complete at {date}"
echo ""

# Experiment 4: Spatial + Frequency
echo "============================================"
echo "Experiment 4: Spatial + Frequency"
echo "Output: results/part4/exp4_spatial_frequency/"
echo "============================================"
python train_encoder.py --config configs/exp4_spatial_frequency.yaml
echo ""
echo "Experiment 4 complete at ${date}"
echo ""

# Experiment 5: Spatial + Contrastive
echo "============================================"
echo "Experiment 5: Spatial + Contrastive"
echo "Output: results/part4/exp5_spatial_contrastive/"
echo "============================================"
python train_encoder.py --config configs/exp5_spatial_contrastive.yaml
echo ""
echo "Experiment 5 complete at ${date}"
echo ""

# Experiment 6: Full Model
echo "============================================"
echo "Experiment 6: Full Model (All Components)"
echo "Output: results/part4/exp6_full_model/"
echo "============================================"
python train_encoder.py --config configs/exp6_full_model.yaml
echo ""
echo "Experiment 6 complete at ${date}"
echo ""

echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================"

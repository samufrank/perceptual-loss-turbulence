#!/bin/bash
# Unified overnight training script for all experiments
# Runs 8 experiments total: 4 on small dataset (improved losses) + 4 on large dataset (original losses)

set -e  # Exit on error

echo "========================================================"
echo "Unified Overnight Training - All Experiments"
echo "========================================================"
echo ""
echo "  Phase 1: Improved losses on small dataset (4 exp × 1.7hr = 6.8hr)"
echo "  Phase 2: Original losses on large dataset (4 exp × 1.7hr = 6.8hr)"
echo ""
echo "========================================================"
echo ""

START_TIME=$(date +%s)

# =============================================================================
# PHASE 1: Improved Losses on Small Dataset
# =============================================================================
echo ""
echo "========================================================"
echo "PHASE 1: IMPROVED LOSSES (Small Dataset - 134 scenes)"
echo "========================================================"
echo ""

# Exp 3 V2: Spatial only (baseline rerun)
echo "[1/8] Starting Exp 3 V2 (Spatial Only - Small)..."
python train_encoder.py --config configs/exp3_spatial_only_v2.yaml
echo "✓ Exp 3 V2 complete."
echo ""

# Exp 4 V2: Spatial + Improved Frequency
echo "[2/8] Starting Exp 4 V2 (Spatial + Frequency Divergence - Small)..."
python train_encoder.py --config configs/exp4_spatial_frequency_v2.yaml
echo "✓ Exp 4 V2 complete."
echo ""

# Exp 5 V2: Spatial + Improved Contrastive
echo "[3/8] Starting Exp 5 V2 (Spatial + Hard Negatives - Small)..."
python train_encoder.py --config configs/exp5_spatial_contrastive_v2.yaml
echo "✓ Exp 5 V2 complete."
echo ""

# Exp 6 V2: Full Model with Improvements
echo "[4/8] Starting Exp 6 V2 (Full Model Improved - Small)..."
python train_encoder.py --config configs/exp6_full_model_v2.yaml
echo "✓ Exp 6 V2 complete."
echo ""

PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - START_TIME))
echo "Phase 1 complete in $((PHASE1_DURATION / 3600))h $((PHASE1_DURATION % 3600 / 60))m"
echo ""

# =============================================================================
# PHASE 2: Original Losses on Large Dataset
# =============================================================================
echo ""
echo "========================================================"
echo "PHASE 2: ORIGINAL LOSSES (Large Dataset - 334 scenes)"
echo "========================================================"
echo ""

# Exp 3 Large: Spatial only
echo "[5/8] Starting Exp 3 Large (Spatial Only - Large)..."
python train_encoder.py --config configs/exp3_spatial_only_large.yaml
echo "✓ Exp 3 Large complete."
echo ""

# Exp 4 Large: Spatial + Original Frequency
echo "[6/8] Starting Exp 4 Large (Spatial + Frequency Original - Large)..."
python train_encoder.py --config configs/exp4_spatial_frequency_large.yaml
echo "✓ Exp 4 Large complete."
echo ""

# Exp 5 Large: Spatial + Original Contrastive
echo "[7/8] Starting Exp 5 Large (Spatial + Contrastive Original - Large)..."
python train_encoder.py --config configs/exp5_spatial_contrastive_large.yaml
echo "✓ Exp 5 Large complete."
echo ""

# Exp 6 Large: Full Model Original
echo "[8/8] Starting Exp 6 Large (Full Model Original - Large)..."
python train_encoder.py --config configs/exp6_full_model_large.yaml
echo "✓ Exp 6 Large complete."
echo ""

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================"
echo ""
echo "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""
echo "Results locations:"
echo "  Small + Improved: results/part4/exp{3,4,5,6}_*_v2/"
echo "  Large + Original: results/part4/exp{3,4,5,6}_*_large/"
echo ""
echo "Next steps:"
echo "  1. Export and compare results"
echo "  2. Generate comparison tables"
echo "  3. Analyze loss improvements vs data scale effects"
echo ""

# =============================================================================
# Auto-Export Results
# =============================================================================
echo "Exporting results to CSV..."

# Export small dataset improved results
python scripts/export_tensorboard_data.py \
    --experiments exp3_spatial_only_v2 exp4_spatial_frequency_v2 exp5_spatial_contrastive_v2 exp6_full_model_v2 \
    --output-dir results/part4/analysis_v2

# Export large dataset results
python scripts/export_tensorboard_data.py \
    --experiments exp3_spatial_only_large exp4_spatial_frequency_large exp5_spatial_contrastive_large exp6_full_model_large \
    --output-dir results/part4/analysis_large

echo "Results exported. Ready for analysis."

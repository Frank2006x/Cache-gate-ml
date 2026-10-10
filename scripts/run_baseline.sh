#!/usr/bin/env bash
# ==============================================================================
# Script to run standard continuous TTA baselines (TENT & EATA without Cache-Gating)
# ==============================================================================

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo " Running Baseline TTA Experiments (Full Adaptation / No Gate)"
echo "=========================================================="

BACKBONE="resnet18"
DATASET="cifar10"
STREAM_LEN=1000
REPEAT_PROB=0.5
LOG_DIR="./runs/baselines"

# 1. Source Model (No adaptation)
echo "--> Running Source Model Baseline..."
python main.py \
    --model source \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating none \
    --log_dir "$LOG_DIR" \
    --exp_name "source_baseline_gaussian_noise"

# 2. TENT Baseline (Full adaptation every frame)
echo "--> Running TENT Baseline on Gaussian Noise..."
python main.py \
    --model tent \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating none \
    --log_dir "$LOG_DIR" \
    --exp_name "tent_baseline_gaussian_noise"

# 3. EATA Baseline (Full adaptation every frame)
echo "--> Running EATA Baseline on Gaussian Noise..."
python main.py \
    --model eata \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating none \
    --log_dir "$LOG_DIR" \
    --exp_name "eata_baseline_gaussian_noise"

# 4. SAR Baseline
echo "--> Running SAR Baseline on Gaussian Noise..."
python main.py \
    --model sar \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating none \
    --log_dir "$LOG_DIR" \
    --exp_name "sar_baseline_gaussian_noise"

# 5. CoTTA Baseline
echo "--> Running CoTTA Baseline on Gaussian Noise..."
python main.py \
    --model cotta \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating none \
    --log_dir "$LOG_DIR" \
    --exp_name "cotta_baseline_gaussian_noise"

# 6. DUA Baseline (Dynamic Uncertainty-Aware Normalization)
echo "--> Running DUA Baseline on Gaussian Noise..."
python main.py \
    --model dua \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating none \
    --log_dir "$LOG_DIR" \
    --exp_name "dua_baseline_gaussian_noise"

echo "=========================================================="
echo " All 6 Baseline runs complete! Check results in $LOG_DIR"
echo "=========================================================="

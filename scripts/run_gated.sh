#!/usr/bin/env bash
# ==============================================================================
# Script to run Algorithmic Cache-Gated TTA (SSIM & pHash Threshold Sweeps)
# ==============================================================================

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo " Running Cache-Gated Continuous TTA Experiments"
echo "=========================================================="

BACKBONE="resnet18"
DATASET="cifar10"
CORRUPTION="gaussian_noise"
SEVERITY=5
STREAM_LEN=1000
REPEAT_PROB=0.6
LOG_DIR="./runs/gated"

# Sweep thresholds: tau = 0.80, 0.85, 0.90
THRESHOLDS=(0.80 0.85 0.90)

for TAU in "${THRESHOLDS[@]}"; do
    echo "----------------------------------------------------------"
    echo "--> Running Cache-Gated TENT (SSIM, tau=$TAU)..."
    python main.py \
        --model tent \
        --backbone "$BACKBONE" \
        --dataset "$DATASET" \
        --corruption "$CORRUPTION" \
        --severity "$SEVERITY" \
        --stream_length "$STREAM_LEN" \
        --repeat_prob "$REPEAT_PROB" \
        --gating ssim \
        --threshold "$TAU" \
        --cache_policy keyframe \
        --log_dir "$LOG_DIR" \
        --exp_name "gated_tent_ssim_tau${TAU}"

    echo "--> Running Cache-Gated EATA (SSIM, tau=$TAU)..."
    python main.py \
        --model eata \
        --backbone "$BACKBONE" \
        --dataset "$DATASET" \
        --corruption "$CORRUPTION" \
        --severity "$SEVERITY" \
        --stream_length "$STREAM_LEN" \
        --repeat_prob "$REPEAT_PROB" \
        --gating ssim \
        --threshold "$TAU" \
        --cache_policy keyframe \
        --log_dir "$LOG_DIR" \
        --exp_name "gated_eata_ssim_tau${TAU}"
done

# Run Perceptual Hash (pHash) Comparison
echo "----------------------------------------------------------"
echo "--> Running Cache-Gated TENT with Perceptual Hash (pHash, tau=0.85)..."
python main.py \
    --model tent \
    --backbone "$BACKBONE" \
    --dataset "$DATASET" \
    --corruption "$CORRUPTION" \
    --severity "$SEVERITY" \
    --stream_length "$STREAM_LEN" \
    --repeat_prob "$REPEAT_PROB" \
    --gating phash \
    --threshold 0.85 \
    --cache_policy keyframe \
    --log_dir "$LOG_DIR" \
    --exp_name "gated_tent_phash_tau0.85"

echo "=========================================================="
echo " Gated runs complete! Check results in $LOG_DIR"
echo "=========================================================="

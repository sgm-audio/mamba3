#!/usr/bin/env bash

# ==============================================================================
# Mamba-3 Native Ubuntu Training Loop Runner
# ==============================================================================
# Automates ROCm GPU configuration, memory tuning for 16GB UMA, automatic
# checkpoint detection, and logging natively on Ubuntu.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${SCRIPT_DIR}/.venv/bin/activate"

# Native configuration defaults
# Override with DATASET_DIR=/path/to/musdb18hq/train ./run_train.sh
DEFAULT_DATASET_DIR="${DATASET_DIR:-./datasets/musdb18hq/train}"
DEFAULT_BATCH_SIZE=2       # Safer default for 16GB UMA compared to 24GB
DEFAULT_CKPT_EVERY=1000
DEFAULT_VAL_EVERY=2000
LOG_FILE="${SCRIPT_DIR}/train.log"

# 1. Check for virtual environment
if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: Virtual environment not found at ${SCRIPT_DIR}/.venv"
    echo "Run ./setup_env.sh first (see docs/training/ubuntu-rocm.md)."
    exit 1
fi

# 2. Activate environment
source "${VENV_ACTIVATE}"

# 3. Configure AMD ROCm Environment Variables for Radeon 860M (gfx1152)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# Disable SDMA to prevent memory pinning conflicts on APU shared RAM
export HSA_ENABLE_SDMA=0

# Optimize PyTorch memory allocator to avoid fragmentation in a 16GB UMA buffer
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
export MIOPEN_FIND_MODE=5

echo "======================================================================"
echo " Starting Mamba-3 Native GPU Training"
echo " Spoofer active: HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
echo " UMA Limit: 16 GB Optimized (max_split_size_mb=256)"
echo "======================================================================"

# 4. Check for checkpoints to resume automatically
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
LATEST_CKPT=""

if [ -d "${CKPT_DIR}" ]; then
    # Look for both regular checkpoints and interrupt recovery checkpoints, sorted by modification time (newest first)
    LATEST_CKPT=$(ls -t "${CKPT_DIR}"/checkpoint_*.pt "${CKPT_DIR}"/interrupt_*.pt 2>/dev/null | head -n 1 || true)
fi

# Build execution arguments
ARGS=(
  --data "${DEFAULT_DATASET_DIR}"
  --device cuda
  --batch "${DEFAULT_BATCH_SIZE}"
  --ckpt-every "${DEFAULT_CKPT_EVERY}"
  --val-every "${DEFAULT_VAL_EVERY}"
)

if [ -n "${LATEST_CKPT}" ]; then
    echo "--> [Resume] Auto-detected latest checkpoint: ${LATEST_CKPT}"
    echo "--> Resuming training..."
    ARGS+=(--resume "${LATEST_CKPT}")
else
    echo "--> [Fresh Start] No existing checkpoints found in checkpoints/."
    echo "--> Starting fresh training run..."
fi

# Append any user overrides passed to this script
ARGS+=("$@")

# 5. Run the training script with active logging
echo "--> Command: python3 train.py ${ARGS[*]} 2>&1 | tee -a ${LOG_FILE}"
echo "----------------------------------------------------------------------"

python3 "${SCRIPT_DIR}/train.py" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

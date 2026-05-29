#!/usr/bin/env bash

# ==============================================================================
# Mamba-3 Native Ubuntu Smoke Test Launcher
# ==============================================================================
# Activates the local virtual environment, configures AMD ROCm variables,
# and runs test_smoke.py natively on the GPU.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${SCRIPT_DIR}/.venv/bin/activate"

# 1. Check for virtual environment
if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: Virtual environment not found at ${SCRIPT_DIR}/.venv"
    echo "Please run './setup_env.sh' first to set up the environment."
    exit 1
fi

# 2. Activate environment
source "${VENV_ACTIVATE}"

# 3. Configure AMD ROCm Environment Variables for Radeon 860M (gfx1152)
echo "--> Configuring AMD ROCm env variables..."

# Spoof gfx1152 (RDNA3/RDNA3.5) as gfx1103 (supported Phoenix APU in ROCm 6.x)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# Disable SDMA to prevent memory pinning conflicts on APU shared RAM
export HSA_ENABLE_SDMA=0

# Optimize PyTorch memory allocator to avoid fragmentation in a 16GB UMA buffer
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

# MIOpen fine-grained search tuning
export MIOPEN_FIND_MODE=5

echo "    HSA_OVERRIDE_GFX_VERSION = ${HSA_OVERRIDE_GFX_VERSION}"
echo "    PYTORCH_ROCM_ARCH        = ${PYTORCH_ROCM_ARCH}"
echo "    PYTORCH_ALLOC_CONF       = ${PYTORCH_ALLOC_CONF}"

# 4. Execute test_smoke.py natively on GPU
echo "--> Launching native smoke tests..."
python3 "${SCRIPT_DIR}/test_smoke.py" --device cuda "$@"

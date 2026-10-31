#!/usr/bin/env bash
# ==============================================================================
# Mamba-3 Offline CREPE f0 Extraction (train + test splits)
# ==============================================================================
# Wraps extract_crepe.py. Resumable — already-extracted stems are skipped.
#
#   ./extract_f0.sh                       # uses $DATASET_ROOT or the default below
#   DATASET_ROOT=~/datasets/musdb18hq ./extract_f0.sh --model tiny
# Extra args are forwarded to extract_crepe.py.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${SCRIPT_DIR}/.venv/bin/activate"
DATASET_ROOT="${DATASET_ROOT:-./datasets/musdb18hq}"

if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "ERROR: Virtual environment not found at ${SCRIPT_DIR}/.venv — run ./setup_env.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_ACTIVATE}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export HSA_ENABLE_SDMA=0

python -c "import torchcrepe" 2>/dev/null || pip install torchcrepe

for split in train test; do
    if [ -d "${DATASET_ROOT}/${split}" ]; then
        echo "--> Extracting f0 for ${DATASET_ROOT}/${split}"
        python "${SCRIPT_DIR}/extract_crepe.py" --data "${DATASET_ROOT}/${split}" --device cuda --model full "$@"
    else
        echo "--> Skipping ${split}: ${DATASET_ROOT}/${split} not found"
    fi
done

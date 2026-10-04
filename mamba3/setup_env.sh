#!/usr/bin/env bash
# ==============================================================================
# Mamba-3 Native Linux Environment Setup
# ==============================================================================
# Creates ./.venv next to this script and installs the training stack.
#
#   ./setup_env.sh            # ROCm (default: PyTorch rocm6.4 wheels)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 ./setup_env.sh   # CUDA
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu   ./setup_env.sh   # CPU only
#
# Re-running is safe (pip is idempotent). Delete .venv to start over.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}"

if [ -d "${VENV_DIR}" ] && grep -qE "home = (/usr/bin|/home/)" "${VENV_DIR}/pyvenv.cfg" 2>/dev/null; then
    echo "--> Reusing existing venv at ${VENV_DIR}"
else
    echo "--> Creating venv at ${VENV_DIR} with ${PYTHON}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel

echo "--> Installing PyTorch from ${TORCH_INDEX}"
pip install torch torchaudio --index-url "${TORCH_INDEX}"

echo "--> Installing training requirements"
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "--> Sanity check"
python - <<'PY'
import torch, torchaudio, soundfile, auraloss
print(f"torch {torch.__version__}  torchaudio {torchaudio.__version__}")
print(f"accelerator available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)}, hip={torch.version.hip})" if torch.cuda.is_available() else ""))
PY

echo "======================================================================"
echo " Done. Next: ./run_smoke_test.sh   then   ./extract_f0.sh   then   ./run_train.sh"
echo "======================================================================"

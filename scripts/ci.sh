#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml (python-cpu job), plus the Rust
# checks with --rust. Needs torch/torchaudio (CPU is fine), numpy, soundfile,
# auraloss, ruff and compiler/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
ROOT="$PWD"

"$PY" -m ruff check .
(cd mamba3 && "$PY" smoke_test.py --device cpu)
(cd mamba3 && "$PY" test_smoke.py --device cpu)
(cd mamba3/edge_dsp_meta_prompt/compiler && "$PY" test_controller.py)

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"$PY" - "$TMP" <<'PY'
import numpy as np, soundfile as sf, os, sys
root = sys.argv[1]; sr = 44100
for i in range(4):
    d = f"{root}/ds/train/track_{i}"; os.makedirs(d, exist_ok=True)
    t = np.arange(sr * 2) / sr
    for j, s in enumerate(["bass", "drums", "vocals", "other"]):
        sf.write(f"{d}/{s}.wav", (0.3 * np.sin(2 * np.pi * (110 * (i + 1) + 70 * j) * t)).astype("float32"), sr)
PY
(cd mamba3 && "$PY" train.py --data "$TMP/ds/train" --save "$TMP/ck" --device cpu --batch 2 --steps 4 \
  --val-every 2 --ckpt-every 2 --log-every 1 --num-workers 0 --d-model 64 --n-layers 1 --val-split 0.25)
test -f "$TMP/ck/final.pt" && test -f "$TMP/ck/best.pt"

if [[ "${1:-}" == "--rust" ]]; then
  cd "$ROOT/mamba3/edge_dsp_meta_prompt"
  cargo check --workspace --all-targets   # known to fail: 26 errors in edge_dsp_plugin (2026-10-31)
  cargo test -p edge_dsp_core             # known: 8 pass, 3 fail (2026-10-31)
fi
echo "ci.sh: python gates passed"

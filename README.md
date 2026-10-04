# mamba3 — Mamba-3 MIMO per-track audio processor + edge DSP controller

Two related sub-projects live in this repository:

| Path | What it is | Entry points |
|---|---|---|
| `mamba3/` | **Per-track adaptive filter bank** — a pure-PyTorch Mamba-3 MIMO model (`model.py`) trained as a self-supervised *restorer* on MUSDB18-HQ stems (`train.py`, `degrade.py`, `extract_crepe.py`). Targets AMD ROCm (Radeon 860M / gfx1152) and CUDA cloud GPUs. | `smoke_test.py`, `test_smoke.py`, `train.py` |
| `mamba3/edge_dsp_meta_prompt/` | **Edge DSP controller toolchain** — distils a rule-based "GoldenPolicy" into a tiny recurrent Mamba-3 controller, exports it to ONNX, INT8-quantises it, and embeds it in a Rust CLAP plugin (`core/`, `plugin/`). | `compiler/train_controller.py`, `compiler/quantize.py`, `compiler/fit_profile.py` |
| `mamba3/mamba/` | Partial, **unmodified reference copy** of [state-spaces/mamba](https://github.com/state-spaces/mamba) (Apache-2.0) kept for the Mamba-3 reference kernels/tests. Not imported by anything here. | — |

## Quick start (CPU is enough for the gates)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchaudio            # pick the index for your accelerator (ROCm/CUDA/CPU)
pip install -r mamba3/requirements.txt
pip install -r mamba3/edge_dsp_meta_prompt/compiler/requirements.txt   # only for the ONNX toolchain

# Gates (what CI runs — see .github/workflows/ci.yml)
ruff check .                                              # pyflakes-only lint (ruff.toml)
(cd mamba3 && python smoke_test.py --device cpu)          # 12 sections, ~10 s on CPU
(cd mamba3 && python test_smoke.py --device cpu)          # 10 sections
(cd mamba3/edge_dsp_meta_prompt/compiler && python test_controller.py)
```

## Training

| Platform | Guide |
|---|---|
| Windows + ROCm (Radeon 860M, Python 3.14/3.12) | [`mamba3/TRAINING_WINDOWS.md`](mamba3/TRAINING_WINDOWS.md) — `env_314.ps1`, `run_smoke_test.ps1`, `run_train.ps1` |
| Bazzite / distrobox + ROCm | [`mamba3/TRAINING.md`](mamba3/TRAINING.md) |
| Native Ubuntu + ROCm | [`mamba3/TRAINING_UBUNTU.md`](mamba3/TRAINING_UBUNTU.md) *(status: in progress — see note at top)* |
| RunPod / CUDA | [`runpod_guide.md`](runpod_guide.md) |

Typical cloud run: `python train.py --device cuda --batch 128 --data /workspace/datasets/musdb18hq/train`.
`train.py --help` lists every flag; all documented flags exist. Dataset layout is
`<root>/<track>/{bass,drums,vocals,other}.wav` (+ optional `<stem>.crepe.npy` / `<stem>.pitched.npy`
produced by `extract_crepe.py --model {tiny,full}`).

## Edge controller (ONNX → plugin)

```bash
cd mamba3/edge_dsp_meta_prompt/compiler
python train_controller.py --steps 2000 --export controller.onnx --quantized ../mamba3_controller_quant.onnx
python fit_profile.py --data client_features/ --weights checkpoints/controller.pt --profile-out profile.npy
```

`quantize.py` refuses any graph that does not expose exactly
`observer_features (B,1,12)`, `session_state (B,16)`, `hidden_state_in (B,3,64)` →
`dsp_parameters (B,32)`, `hidden_state_out (B,3,64)`. The committed `mamba3_controller_quant.onnx` satisfies this
contract (regenerated 2026-10-04; details in [`mamba3/REFERENCE_THIS_SHIT.md`](mamba3/REFERENCE_THIS_SHIT.md) §2).

Plugin build: `cd mamba3/edge_dsp_meta_prompt && cargo xtask bundle edge_dsp_plugin --release` (NIH-plug bundler, `xtask/`).
The Rust crates are **not built in CI yet** (the review sandbox had no Rust toolchain) — the first successful `cargo build` still needs to be recorded; see the status note in the reference doc.

## Repo conventions

- `AGENTS.md` (root and `mamba3/`) hold operator notes for the Windows/ROCm workflow.
- No pytest suite: the smoke tests + `compiler/test_controller.py` are the test gate and all exit non-zero on failure.
- Checkpoints (`*.pt`), datasets, logs and venvs are git-ignored; the only tracked binaries are the
  plugin's ONNX artifact and the vendored Mamba README image.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). The vendored reference copy under `mamba3/mamba/` is
Copyright 2023 Tri Dao, Albert Gu, Apache-2.0 (its upstream [`LICENSE`](mamba3/mamba/LICENSE) is included).
Third-party training deps (auraloss, torchcrepe) are Apache-2.0 / MIT.

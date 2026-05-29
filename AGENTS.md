# Mamba-3 Repo Guide

## Windows env setup
- **`mamba3/env_314.ps1`** — one-shot env setup (Python 3.14→3.12, ROCm 7.13.0 for gfx1152, checks Smart App Control, username spaces, VC++ redist, GPU driver ≥31.x). Only for native gfx1152 ROCm wheels at `repo.amd.com/rocm/whl/gfx1152/`.
- **`mamba3/setup_env.ps1`** — multi-backend fallback (auto/rocm/dml/cpu). Prefer `env_314.ps1` first.
- Venv activation must use dot-sourcing (`.`), not call (`&`): `. .\.venv\Scripts\Activate.ps1`
- PowerShell backtick (`` ` ``) for line continuation; `Tee-Object` for log tee.
- HSA override NOT needed on Windows (native gfx1152 wheels exist). Linux Bazzite needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

## Training commands
- **Windows**: `mamba3/run_train.ps1` — auto-detects checkpoints for resume, sets `HSA_ENABLE_SDMA=0`, `PYTORCH_ALLOC_CONF`.
- **Linux**: `mamba3/train.py` directly via `run_train.sh`.
- **Cloud GPU (RunPod)**: `python train.py --device cuda --batch 128 --num-workers 4`
- `--ckpt-every 500` on Windows (gfx1152 GPU hang bug mitigation), `--ckpt-every 1000` on Linux.
- `train.py --device cuda` on ROCm (uses `torch.cuda` namespace like NVIDIA).
- `train.py --no-amp` to disable bf16 (if NaN/Inf every step).
- `train.py --no-grad-ckpt` to disable gradient checkpointing (uses more VRAM, ~20% faster on H100/A100).
- `train.py --grad-accum N` for gradient accumulation (effective batch = batch × N).

## Smoke tests
- `mamba3/smoke_test.py` (793 lines, full) — referenced by `run_smoke_test.ps1`.
- `mamba3/test_smoke.py` (764 lines, reduced) — referenced by training docs.
- Both exist; docs reference `test_smoke.py`, scripts reference `smoke_test.py`.

## File layout
```
mamba3/
  env_314.ps1        — Windows ROCm 7.13.0 env setup (Python 3.14)
  setup_env.ps1      — generic Windows env setup (any backend)
  run_train.ps1      — Windows training launcher
  run_smoke_test.ps1 — Windows smoke test launcher
  model.py           — Mamba-3 MIMO architecture (Pure PyTorch)
  train.py           — 3-phase training loop (100k steps default)
  smoke_test.py      — 9-section pre-training smoke test
  test_smoke.py      — alternative smoke test (7 sections)
  TRAINING.md        — Linux (Bazzite/ROCm) workflow
  TRAINING_WINDOWS.md — Windows workflow
  edge_dsp_meta_prompt/ — separate Rust+Python ONNX export toolchain
  Mamba\ Based/       — separate project (CLAP plugin, has its own AGENTS.md)
```

## ROCm on Windows caveats
- Only PyTorch-with-ROCm acceleration, no full ROCm stack (no rocprofiler, no HIP tools).
- Smart App Control must be OFF (one-way disable, requires Windows reinstall to re-enable).
- GPU hang bug `MES:0x82` on Ryzen AI 7 350 / Radeon 860M — save checkpoints often.
- `torch.cuda.is_available()` returns True on ROCm (API-compat layer, not real CUDA).

## Testing
- No formal test framework — smoke tests (`smoke_test.py` / `test_smoke.py`) are the gate.
- Run smoke test before training: `python smoke_test.py --device cuda` (venv activated).

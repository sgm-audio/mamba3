# mamba3

A per-track audio restorer and a small edge DSP controller, both built on a
pure-PyTorch state-space layer that follows the Mamba-3 paper's MIMO update.
It is not the reference kernel and hasn't been parity-tested against the
vendored `mamba3_MIMO_step_ref`.

Status (2026-10-31): the Python side trains and its gates pass on CPU. The Rust
plugin does not compile yet (26 errors in `edge_dsp_plugin`). No trained
restorer checkpoint is published.

| Path | What it is |
|---|---|
| `mamba3/` | Per-track restorer. `model.py` is the network, `train.py` trains it self-supervised on MUSDB18-HQ stems (degrade with EQ/comp/sat/reverb, learn to undo it), `extract_crepe.py` adds optional pitch features. Written for AMD ROCm (Radeon 860M, gfx1152) and CUDA. |
| `mamba3/edge_dsp_meta_prompt/compiler/` | Controller toolchain. Distils a rule-based `GoldenPolicy` into a small recurrent controller, exports ONNX and INT8-quantises it. |
| `mamba3/edge_dsp_meta_prompt/core/`, `plugin/` | Rust RT core and a NIH-plug CLAP plugin meant to run that ONNX graph. Core builds; 8 of 11 tests pass. Plugin doesn't build. |
| `mamba3/mamba/` | Unmodified partial copy of [state-spaces/mamba](https://github.com/state-spaces/mamba) (Apache-2.0), kept for the reference kernels. Nothing here imports it. |
| `docs/` | Training guides per platform, the edge controller reference notes, and early concept specs. |
| `scripts/ci.sh` | Runs the same gates as CI. |

## Quick start (CPU is enough for the gates)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchaudio            # pick the index for your accelerator
pip install numpy soundfile auraloss ruff
pip install -r mamba3/requirements.txt
pip install -r mamba3/edge_dsp_meta_prompt/compiler/requirements.txt
bash scripts/ci.sh       # add --rust to also run cargo check / cargo test (known failures, below)
```

`scripts/ci.sh` runs ruff (pyflakes rules only), `smoke_test.py`,
`test_smoke.py`, `compiler/test_controller.py`, and a 4-step `train.py` run on
a synthetic 4-track dataset. There is no pytest suite; each script exits
non-zero on failure.

Results on 2026-10-31, CPU only (8-thread Xeon, shared, torch 2.15, Python
3.12): ruff clean, `smoke_test.py` exit 0 (one warning: no GPU), `test_smoke.py`
10/10, `test_controller.py` 9/9 including the ONNX round-trip, and the 4-step
train run writes `final.pt` and `best.pt`.

GitHub Actions: no run has completed yet; the account's Actions billing is
blocked. Until that's fixed, the numbers above are local runs only.

## Training

| Platform | Guide |
|---|---|
| Windows + ROCm (Radeon 860M) | [`docs/training/windows-rocm.md`](docs/training/windows-rocm.md) |
| Bazzite / distrobox + ROCm | [`docs/training/bazzite-distrobox-rocm.md`](docs/training/bazzite-distrobox-rocm.md) |
| Ubuntu + ROCm | [`docs/training/ubuntu-rocm.md`](docs/training/ubuntu-rocm.md) (not finished) |
| RunPod / CUDA | [`docs/training/runpod.md`](docs/training/runpod.md) |

Example: `python train.py --device cuda --batch 128 --data <musdb18hq>/train`.
`train.py --help` lists the flags. Dataset layout is
`<root>/<track>/{bass,drums,vocals,other}.wav`, plus optional
`<stem>.crepe.npy` / `<stem>.pitched.npy` from `extract_crepe.py`.

MUSDB18-HQ is licensed for non-commercial research use. A model trained on it
inherits that restriction; don't ship one in a commercial product.

## Edge controller (ONNX)

```bash
cd mamba3/edge_dsp_meta_prompt/compiler
python train_controller.py --steps 2000 --export controller.onnx --quantized ../mamba3_controller_quant.onnx
python fit_profile.py --data client_features/ --weights checkpoints/controller.pt --profile-out profile.npy
```

`quantize.py` rejects any graph that doesn't expose exactly
`observer_features (B,1,12)`, `session_state (B,16)`, `hidden_state_in (B,3,64)`
→ `dsp_parameters (B,32)`, `hidden_state_out (B,3,64)`. The committed
`mamba3_controller_quant.onnx` matches that contract (regenerated 2026-10-04;
sha256 `0c78075e0f35bd7fc2967d5796281420cd0d1ab3c3fb52a8bbfc2f45456464f3`). It
was distilled from the rule policy, so it imitates `GoldenPolicy`; it hasn't
been judged on audio. Notes are in
[`docs/edge-controller-reference.md`](docs/edge-controller-reference.md).

## Rust crates (not working yet)

Checked 2026-10-31 with Rust 1.97.1 on Linux x86-64:

- `cargo check --workspace --all-targets`: dependency resolution and the core
  crate are fixed, and the plugin crate fails with 26 errors (API drift against
  the pinned nih-plug revision and `ort 2.0.0-rc.13`).
- `cargo test -p edge_dsp_core`: 8 pass, 3 fail (`test_svf_lowpass`,
  `test_adaa_clipper`, `loud_signal_is_attenuated`).

The intended bundle command is
`cd mamba3/edge_dsp_meta_prompt && cargo xtask bundle edge_dsp_plugin --release`;
it won't work until the plugin compiles. The CI Rust job is advisory for that
reason.

## Repo notes

- Checkpoints (`*.pt`), datasets, logs and venvs are git-ignored. The tracked
  binaries are the controller ONNX file and the vendored Mamba README image.
- `docs/specs/` holds early product and architecture concepts. They describe
  plans, not current behaviour.

## License

Apache License 2.0, see [`LICENSE`](LICENSE). The vendored copy under
`mamba3/mamba/` is Copyright 2023 Tri Dao and Albert Gu, Apache-2.0 (its
[`LICENSE`](mamba3/mamba/LICENSE) is included). Training deps auraloss and
torchcrepe are Apache-2.0 / MIT.

# MASTER REFERENCE GUIDE: Edge-Inference DSP Meta-Prompt Stack & CLAP Plugin Framework

This document serves as the quick-reference for the **Edge-Inference DSP Meta-Prompt Stack** and **CLAP Plugin Framework** codebase under `mamba3/edge_dsp_meta_prompt/` (paths below are relative to that directory).

<!-- STATUS: in progress -->
> **Status (review 2026-10-04):** the Python toolchain (`compiler/`) is runnable and tested; the Rust crates have **not been built in CI** (no `Cargo.lock`, `nih_plug` is a git dependency, `ort` is pinned to a 2.0 release candidate) and the committed ONNX artifact is the legacy one (see §2). Treat the Rust side as unverified until a `cargo build` is on record.

---

## 1. Project Directory Mapping

```
edge_dsp_meta_prompt/
├── Cargo.toml (Workspace definition)
│
├── core/ (Rust Crate: Real-time safe DSP & lock-free queues)
│   ├── Cargo.toml (Dependencies: realfft, rtrb)
│   └── src/
│       ├── lib.rs (Module exports)
│       ├── ring_buffer.rs (Lock-free SPSC Ring Buffer)
│       ├── double_buffer.rs (Atomic swap Triple-Buffer)
│       ├── observer.rs (Zero-allocation features extractor)
│       ├── dsp_engine.rs (ZDF SVF EQ filter & ADAA Saturator)
│       └── slice_engine.rs (Zero-crossing segmenter)
│
├── plugin/ (Rust Crate: CLAP plugin shell)
│   ├── Cargo.toml (Dependencies: nih-plug, ort)
│   └── src/
│       ├── lib.rs (CLAP macro exports)
│       ├── plugin.rs (Plugin callback process loop)
│       ├── worker.rs (Inference worker executing ONNX)
│       └── auditor.rs (Auditor agent validating parameters)
│
├── compiler/ (Python: Prompt Stack & Exporter Toolchain)
│   ├── requirements.txt
│   ├── prompt_stack.py (5-Layer compiler structures)
│   ├── train_controller.py (Mamba3Controller, GoldenPolicy distillation, ONNX export)
│   ├── fit_profile.py (6C per-customer 8-D profile prompt-tuning)
│   ├── quantize.py (INT8 dynamic quantizer + signature/parity gate)
│   └── test_controller.py (regression tests: recurrence, state layout, ONNX round-trip)
│
└── specs/ (Product and Architectural Reference specifications)
    ├── prd_cynical_engineer.md
    ├── prd_intelligent_mix_engine.md
    ├── prd_precision_surgical_tool.md
    ├── ard_cynical_engineer.md
    ├── ard_intelligent_mix_engine.md
    └── ard_precision_surgical_tool.md
```

---

## 2. Model Tensor Interface & Shapes

The quantized Mamba3 Controller ONNX model (`mamba3_controller_quant.onnx`) expects these inputs/outputs:

| Tensor Name | Shape | Data Type | Description |
| :--- | :--- | :---: | :--- |
| **`observer_features`** (Input) | `[1, 1, 12]` | `f32` | ONE Observer frame per 10 ms tick (Way 5A). The SSM hidden state already summarizes history — the legacy 100-frame sliding window `[1, 100, 12]` is obsolete (it unrolled ~100 traced steps per layer and wasted 100× compute per tick). |
| **`session_state`** (Input) | `[1, 16]` | `f32` | L1 Session context (track type one-hot, BPM, beat offset, SR) at `[0:8]`; **6C customer/tone profile embedding** at `[8:16]` (zero = neutral, fitted by `compiler/fit_profile.py`). |
| **`hidden_state_in`** (Input) | `[1, 3, 64]` | `f32` | Packed recurrent states for all 3 Mamba layers (Way 5A). |
| **`dsp_parameters`** (Output) | `[1, 32]` | `f32` | 32 tanh-bounded control coefficients mapped to classical DSP (`train_controller.py::PARAM_LAYOUT`). |
| **`hidden_state_out`** (Output) | `[1, 3, 64]` | `f32` | Next packed recurrent state, fed into the next inference cycle. |

> ⚠️ **Legacy artifact warning — the checked-in `mamba3_controller_quant.onnx` does NOT match this contract.** Inspected with onnxruntime (review 2026-10-04) it declares `observer_features [1, 100, 12]` (the old 100-frame window), `hidden_state_in [1, 64]` (one layer, not `[1, 3, 64]`), plus the leaked `onnx::Mul_3` / `onnx::Mul_4` inputs and `onnx::Mul_3168` / `onnx::Mul_4689` outputs from the tuple-unpacking export bug. `plugin/src/worker.rs` zero-fills unknown inputs and reads outputs by name, **but it fills `observer_features` with 12 floats and `hidden_state_in` with 192 floats regardless of the declared shape**, so `Tensor::from_array` fails on every tick against the legacy artifact and the plugin silently stays on neutral DSP parameters. The artifact must be regenerated before the plugin is useful: `python compiler/train_controller.py --steps 2000 --export controller.onnx --quantized mamba3_controller_quant.onnx` (the export path itself was broken until this review — see `compiler/test_controller.py`). <!-- STATUS: in progress -->

---

## 3. Core API Quick-Reference

### 3.1 Lock-Free SPSC Ring Buffer (`core::ring_buffer`)
Pushes real-time features from RT thread to Worker. Safe to execute in audio callback.
```rust
let queue = SpscRingBuffer::<ObserverFrame>::new(512);
// Producer (RT Thread):
queue.push(frame).map_err(|f| "Queue Full");
// Consumer (Worker Thread):
if let Some(frame) = queue.pop() { ... }
```

### 3.2 Atomic Triple-Buffer Parameter Swap (`core::double_buffer`)
Delivers parameters from Worker back to RT thread callback in $<5\text{ ns}$ with no locking.
```rust
let buffer = TripleBuffer::<DynamicDspParams>::new(DynamicDspParams::default());
// Producer (Worker/Auditor):
buffer.write(verified_params);
// Consumer (RT Thread):
let current_params = buffer.read(); // Fast, lock-free swap
```

### 3.3 Zero-Allocation Observer (`core::observer`)
Performs feature extraction inside the audio callback with zero heap allocations.
```rust
let mut observer = Observer::new(block_size, sample_rate);
// In process loop:
if let Some(frame) = observer.process_sample(input_sample) {
    let _ = queue.push(frame); // Block completed, push features
}
```

### 3.4 ZDF SVF Filterbank (`core::dsp_engine`)
Peaking band Dynamic EQ calculation inside process loop. Safe for sample-rate sweeps.
```rust
let mut filter = ZdfSvf::new();
let filter_type = FilterType::Bell { gain_db: -3.4 };
let filtered_sample = filter.process(sample, cutoff_hz, q_factor, &filter_type, sample_rate);
```

### 3.5 ADAA Saturator (`core::dsp_engine`)
Continuous antiderivative waveshaper to prevent Nyquist frequency aliasing.
```rust
let mut saturator = AdaaSaturator::new();
let saturated_sample = saturator.process(sample, drive_db, wet_dry_mix);
```

---

## 4. Compilation & Verification Execution Guide

All compilation, model training simulations, and dynamic quantization passes must be executed inside the virtual environment `.venv`:

*   **Initialize Virtual Environment**:
    ```bash
    source .venv/bin/activate          # any venv with compiler/requirements.txt installed
    ```
*   **Run Prompt Stack Compiler**:
    ```bash
    python3 compiler/prompt_stack.py
    ```
*   **Run Controller Model & Loss Checks**:
    ```bash
    python3 compiler/train_controller.py
    ```
*   **Run ONNX Exporter & Dynamic Quantizer**:
    ```bash
    python3 compiler/quantize.py
    ```

---

## 5. TODO'S FOR NEXT SESSION (After Model Training Completes)

Once you complete training the `Mamba3Controller` model on your specific datasets, follow this step-by-step roadmap to integrate and deploy the plugin:

### 🟩 Step 1: Export Your Trained PyTorch Weights
1.  Save your finalized training checkpoint inside PyTorch:
    ```python
    torch.save(model.state_dict(), "mamba3_controller_final.pt")
    ```
2.  Move the saved checkpoint `mamba3_controller_final.pt` into the `edge_dsp_meta_prompt/compiler/` folder.

### 🟩 Step 2: Compile the Trained Model to Quantized ONNX
1.  Run the quantizer tool using the final checkpoint to compile the model down to INT8:
    ```bash
    python3 compiler/quantize.py
    ```
    *   This compiles the model and saves a dynamically quantized, edge-ready graph `mamba3_controller_quant.onnx` inside the compiler folder.
2.  Move `mamba3_controller_quant.onnx` into the plugin assets folder:
    ```bash
    mkdir -p plugin/assets/
    mv mamba3_controller_quant.onnx plugin/assets/
    ```

### 🟩 Step 3: Integrate ONNX Model inside the CLAP Crate
1.  Open [plugin/src/worker.rs](file:///home/sgm/Desktop/mamba3/edge_dsp_meta_prompt/plugin/src/worker.rs).
2.  Locate the model load line inside `spawn()` and ensure the path points to the assets folder copy:
    ```rust
    let session = Session::builder()
        .unwrap()
        .commit_from_file("plugin/assets/mamba3_controller_quant.onnx")
    ```

### 🟩 Step 4: Compile the CLAP Plugin Bundle
1.  Ensure Rust and Cargo are available on your system shell PATH.
2.  Build and bundle the workspace:
    ```bash
    cargo xtask bundle edge_dsp_plugin --release
    ```
    *   This compiles the optimized Rust code and bundles it as a standard CLAP plugin (`.clap` library file).

### 🟩 Step 5: Scan & Deploy in the Host (e.g. REAPER)
1.  Copy the compiled `.clap` plugin file into your system's CLAP search path (typically `/usr/lib/clap/` or `~/.clap/` on Linux).
2.  Open REAPER, open Preferences $\rightarrow$ Plug-ins $\rightarrow$ CLAP, and run **Re-scan**.
3.  Add "The Cynical Engineer" to a vocal or instrument track. Open the operation log display on the UI to audit the real-time parameter sweeps as the session plays!

---

## 11. Implementation Status & Change Log (2026-10-04)

Completed per the architecture verification audit (Ways 4B + 5A + 6C):

### Files created
| File | Purpose |
|------|---------|
| `edge_dsp_meta_prompt/core/src/lib.rs` | Crate root, module declarations + re-exports (`TripleBuffer`, `Observer`, …) — `cargo test` for `edge_dsp_core` is now runnable |
| `edge_dsp_meta_prompt/core/src/dynamics.rs` | `DynamicsProcessor` (feed-forward compressor + gentle downward gate, control-rate coefficient smoothing) + unit tests |
| `edge_dsp_meta_prompt/plugin/src/lib.rs` | Plugin crate root |
| `edge_dsp_meta_prompt/plugin/src/plugin.rs` | nih-plug `Plugin` impl, `DynamicDspParams` (32-D denormalization mirroring `train_controller.py::PARAM_LAYOUT`), RT DSP chain, worker thread lifecycle |
| `edge_dsp_meta_prompt/plugin/src/worker.rs` | Background controller thread (Way 4B): embedded ONNX bytes, single-step recurrent inference (100 Hz tick), adaptive input feeding (legacy 5-input compatible), AuditorAgent commit |
| `edge_dsp_meta_prompt/compiler/train_controller.py` | Way 5A: `Mamba3Controller` (3 layers, d_model=64, trapezoidal complex poles, no conv1d), `GoldenPolicy` teacher, distillation loop, clean single-step ONNX export (3 inputs / 2 outputs) |
| `edge_dsp_meta_prompt/compiler/quantize.py` | INT8 dynamic quantization + **signature gate** (rejects the legacy `onnx::Mul_*` leak) + fp32↔int8 parity check |
| `edge_dsp_meta_prompt/compiler/fit_profile.py` | Way 6C: prompt-tunes the 8-D customer profile (`session_state[8:16]`) with all model weights frozen |
| `edge_dsp_meta_prompt/specs/prd_intelligent_mix_engine.md`, `specs/ard_cynical_engineer.md` | Missing spec docs restored |

### Behavioral changes
1. **ONNX contract fixed (Way 5A):** true single-step recurrent export
   `observer_features (1,1,12) + session_state (1,16) + hidden_state_in (1,3,64) →
   dsp_parameters (1,32) + hidden_state_out (1,3,64)`. The old 100-step unrolled
   window wasted ~100× compute per tick and broke state continuity.
2. **Customer profiles without LoRA (Way 6C):** `PromptStack.compile_session_state`
   now accepts an 8-D `profile` written to `session_state[8:16]` (previously
   dead padding). `compiler/fit_profile.py` fits it per customer in seconds on
   CPU; runtime cost is zero extra model bytes and zero extra ops.
3. **Dependency fix:** `plugin/Cargo.toml` moved `ort` from the **yanked**
   `1.16.3` to the maintained `2.0.0-rc.13` line (the old pin is unbuildable
   with a fresh `Cargo.lock`).
4. **DSP engine gap closed:** a compressor/gate (`core/src/dynamics.rs`) now
   actually implements the dynamics parameters `auditor.rs` has always clamped.
5. **Training objective corrected (Recipe 1A, `mamba3/train.py` + `mamba3/degrade.py`):**
   the model now trains as a restorer — synthetic EQ/compression/saturation/reverb
   damage in, clean stem out — instead of the previous identity autoencoder.

### Known limitations (tracked from the audit, not yet addressed)
- `mamba3/model.py`: `Mamba3MIMOLayer.in_proj` is instantiated but unused;
  `_ssm_scan_sequential` drives `B_proj` from raw `x_r8` only (hidden state `x`
  modulates only `dt` + residual). Candidate follow-up: Way 2A/2B recurrence rewiring.
- `mamba3/model.py` STFT path is non-causal (`center=True`, symmetric convs) —
  irrelevant for the Way 4B control plane, but blocks direct streaming (Way 4A).
- `mamba3/mamba/mamba_ssm/modules/mamba3.py` is still missing from the vendored
  upstream snapshot (ops + tests are present).

### Verification gates (run on a machine with torch/cargo)
```bash
# Training pipeline (Recipe 1A: degraded input → clean target)
python mamba3/smoke_test.py --device cuda        # §10 degradation, §11 f0 gating + grad-ckpt
python mamba3/test_smoke.py --device cuda        # §9  degradation + f0 gating

# Controller + quantization + profile (Way 5A/6C)
cd mamba3/edge_dsp_meta_prompt
python compiler/train_controller.py --steps 2000 --export controller.onnx \
    --quantized mamba3_controller_quant.onnx
python compiler/fit_profile.py --data <client_features_dir> --profile-out profile.npy

# Rust (once a toolchain is available)
cargo test --manifest-path core/Cargo.toml
cargo check --manifest-path plugin/Cargo.toml && cargo xtask bundle cynical-engineer
```

---

## 12. Way 2B + 2C: Mamba-3 MIMO recurrence rewrite (2026-10-04)

`mamba3/model.py::Mamba3MIMOLayer` now implements the **official arXiv:2603.15569
MIMO formulation** (matching `mamba3/mamba/tests/ops/tilelang/test_mamba3_mimo.py::
mamba3_MIMO_step_ref`):

1. **Data-dependent rotary angle state** — θ_t = θ_{t-1} + π·tanh(Angles_t)·Δ_t
   applied to Q/K (the paper's complex-valued transitions in RoPE form).
2. **Exponential-trapezoidal update** — S_t = α_t·S_{t-1} + β_t·(K_{t-1}⊗V_{t-1})
   + γ_t·(K_t⊗V_t) with α_t = exp(Δ_t·A_h), β_t = (1−σ(τ_t))·Δ_t·α_t,
   γ_t = σ(τ_t)·Δ_t. The previous-step K/V outer product (trapezoidal lag term)
   replaces the old bilinear filterbank; `x_r8` enters as additive K/V coupling.
3. **Rank-R_mimo MIMO** (`cfg.mimo_rank`, default 2) — per-rank Q/K axis,
   MIMO_V/MIMO_Z gain copies, MIMO_O readout, D skip + Swish z·σ(z) gate.

**Way 2C**: `_scan_chunked` parallelizes the recurrence inside 16-step chunks
(angle cumsum + lower-triangular decay GEMM), cutting Python loop overhead
~16× vs the per-step reference scan. `layer.use_chunked = False` selects the
verbatim sequential reference. Recurrent state carry (θ, S, K, V) is exposed
via `forward(..., state=, return_state=)` for future streaming (Way 4A).

### Resolved audit findings (from §11)
- ✅ `in_proj` dead parameters: removed — Q/K/V/Z projections all live and
  gradient-verified (log_a_real, MIMO_V/O, q_proj, k_r8, angle_proj, trap_proj
  all receive grads with and without checkpointing).
- ✅ Hidden state `x` now drives the full recurrence (Q/K/V/Z are projections
  of `x`; `x_r8` couples additively into K/V).

### Verification (run 2026-10-04, torch 2.14.1 CPU)
| Gate | Result |
|------|--------|
| Scan parity chunked vs sequential (T=77, non-multiple of chunk) | max Δ **3.6e-7** |
| State-carry streaming vs full-window | max Δ **4.8e-7** |
| End-state equality across scan paths | max Δ **1.0e-6** |
| `smoke_test.py --device cpu` (13 sections incl. [10][11][12]) | **0 failures** |
| `test_smoke.py --device cpu` (9 sections incl. [9]) | **9/9 passed** |
| `train.py` micro-run (6 steps, --grad-accum 2, synthetic 4-track set) | all 3 phases entered; logs ONLY at optimizer steps 2/4/6; validation + best.pt + resume-from-checkpoint OK |

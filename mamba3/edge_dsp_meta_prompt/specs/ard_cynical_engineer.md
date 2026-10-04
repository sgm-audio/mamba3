# ARD: The Cynical Engineer (Concept A)

**Status:** Architecture baseline (restored 2026-10-04; referenced by REFERENCE_THIS_SHIT.md § docs index)
**Owner:** sgm-audio
**Related:** `prd_cynical_engineer.md`, `../core/src/*`, `../plugin/src/*`, `../compiler/*`

## 1. Architecture Overview

Asynchronous hybrid control plane ("Way 4B"): the neural model NEVER touches
the audio callback. A quantized Mamba-3 controller observes real-time features
and steers a classical DSP chain at control rate.

```
AUDIO THREAD (hard RT, zero alloc, no locks)
  Observer (12-D features) ──► SpscRingBuffer ─────────────┐
  DspChain:                                               │
    low-cut HPF → 6× ZdfSvf Bell EQ                       │
    → DynamicsProcessor (comp/gate)                       │
    → AdaaSaturator (ADAA) → output gain → master mix     │
        ▲                                                 │
        │ TripleBuffer<DynamicDspParams> (latest wins)    │
        │                                                 ▼
WORKER THREAD (100 Hz, allocation OK)              ControllerWorker
  AuditorAgent (clamps ±9 dB EQ, ratio, ≤8 dB drive)   │
        ▲                                             │
        └── mamba3_controller_quant.onnx (INT8, single-step recurrent)
```

## 2. Module Responsibilities

| Module | Responsibility | RT Contract |
|--------|----------------|-------------|
| `core/src/observer.rs` | 12-D feature extraction (true peak, RMS, crest, ZCR, centroid, 6 mel bands, transient slope) | zero-alloc, O(1)/sample |
| `core/src/ring_buffer.rs` | SPSC feature transport, audio → worker | lock-free, droppable |
| `core/src/double_buffer.rs` | `TripleBuffer` parameter transport, worker → audio | lock-free, latest-wins |
| `core/src/dsp_engine.rs` | `ZdfSvf` (bilinear ZDF SVF, Bell/LP/HP/BP/Notch/AP) + `AdaaSaturator` (1st-order ADAA cubic) | zero-alloc, O(1)/sample |
| `core/src/dynamics.rs` | Feed-forward compressor + gentle downward gate; coefficients precomputed per block | zero-alloc |
| `plugin/src/worker.rs` | ONNX single-step recurrent inference (state (1,3,64)), adaptive input feeding | worker thread only |
| `plugin/src/auditor.rs` | Root Constitution clamps + operation log | worker thread only |
| `plugin/src/plugin.rs` | nih-plug glue, `DynamicDspParams` (32-D denorm), per-block param snapshot | hard RT |

## 3. ONNX Contract (Way 5A)

- Inputs: `observer_features (1,1,12)`, `session_state (1,16)`,
  `hidden_state_in (1,3,64)` — one recurrent step per 10 ms tick.
- Outputs: `dsp_parameters (1,32)` (tanh-bounded), `hidden_state_out (1,3,64)`.
- Quantization: INT8 dynamic (`compiler/quantize.py`), signature-gated (the
  legacy artifact's `onnx::Mul_*` leak is rejected by `verify_signature`).
- 32-D parameter layout: see `compiler/train_controller.py::PARAM_LAYOUT`,
  mirrored by `plugin/src/plugin.rs::DynamicDspParams::from_model_output`.

## 4. Session State & Customer Profile (6C)

`session_state[0:8]`: track type one-hot, tempo, beat offset, sample rate,
timeline position. `session_state[8:16]`: 8-D customer/tone profile, fitted
per customer by `compiler/fit_profile.py` (all weights frozen) and shipped in
preset JSON. Zero vector = neutral.

## 5. Failure Modes & Fallbacks

| Failure | Behavior |
|---------|----------|
| ONNX session fails to load (missing ORT dylib) | Worker exits; audio thread keeps neutral `DynamicDspParams::default()` (transparent chain) |
| Worker stalls | `TripleBuffer` keeps last committed params; no audio-thread blocking ever |
| Model proposes extreme params | `AuditorAgent` clamps (±9 dB EQ, ratio/threshold interlock, ≤8 dB drive) and logs |
| Legacy 5-input ONNX artifact | Worker zero-fills leaked inputs; outputs read by name (compatible both ways) |

## 6. Verification Gates

- `cargo test` (core): ring buffer, triple buffer, observer, DSP, dynamics.
- `compiler/quantize.py --verify`: 3/2 signature gate + fp32↔int8 parity.
- `smoke_test.py` §10–11 / `test_smoke.py` §9: degradation, f0 gating, grad-ckpt.

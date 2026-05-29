# MASTER REFERENCE GUIDE: Edge-Inference DSP Meta-Prompt Stack & CLAP Plugin Framework

This document serves as the absolute quick-reference for the completed **Edge-Inference DSP Meta-Prompt Stack** and **CLAP/ARA Plugin Framework** codebase under `/home/sgm/Desktop/mamba3/edge_dsp_meta_prompt/`.

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
│   ├── train_controller.py (Mamba3Controller & dynamic losses)
│   └── quantize.py (ONNX exporter & 8-bit dynamic quantizer)
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
| **`observer_features`** (Input) | `[1, 100, 12]` | `f32` | Sliding sequence window of the last 100 Observer frames. |
| **`session_state`** (Input) | `[1, 16]` | `f32` | L1 Session context (track type one-hot, BPM, beat offset, SR). |
| **`hidden_state_in`** (Input) | `[1, 64]` | `f32` | Recurrent state carry-over for Mamba layer memory. |
| **`dsp_parameters`** (Output) | `[1, 32]` | `f32` | 32 dynamic control coefficients mapped to classical DSP. |
| **`hidden_state_out`** (Output) | `[1, 64]` | `f32` | Next recurrent state to be fed into the next inference cycle. |

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
    source /home/sgm/Desktop/mamba3/.venv/bin/activate
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

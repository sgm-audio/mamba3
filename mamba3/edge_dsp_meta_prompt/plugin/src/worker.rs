//! Background controller worker (Way 4B / 5A).
//!
//! Runs the quantized Mamba-3 controller at a ~100 Hz control tick:
//!   Observer frames (SPSC ring, audio → here) → single-step recurrent ONNX
//!   → `AuditorAgent::verify_and_commit` (clamps) → `TripleBuffer` (here → audio).
//!
//! The ONNX contract (Way 5A export from `compiler/train_controller.py`):
//!   inputs : observer_features (1,1,12), session_state (1,16),
//!            hidden_state_in (1,3,64)
//!   outputs: dsp_parameters (1,32), hidden_state_out (1,3,64)
//!
//! Backward compatibility: the LEGACY `mamba3_controller_quant.onnx` export
//! leaked extra unnamed inputs/outputs (`onnx::Mul_3`, `onnx::Mul_4`, ...) —
//! a torch.onnx.export tuple-unpacking bug.  `run_step` feeds ANY extra input
//! declared by the graph with zeros and reads outputs BY NAME, so the worker
//! runs correctly against both the legacy artifact and the clean re-export
//! produced by `compiler/train_controller.py --export`.
//!
//! This file runs on a normal thread: allocation and locking are fine here.
//! Nothing in this module may ever be called from `process()`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use edge_dsp_core::{ObserverFrame, SpscRingBuffer, TripleBuffer};

use crate::auditor::AuditorAgent;
use crate::plugin::{DynamicDspParams, PARAM_COUNT};

/// Embed the quantized controller with the plugin binary (937 KB) so the
/// plugin never depends on an external file path at runtime.  Rebuild the
/// plugin after `compiler/train_controller.py` regenerates the artifact.
const MODEL_BYTES: &[u8] = include_bytes!("../../mamba3_controller_quant.onnx");

const OBS_DIM: usize = 12;
const SESS_DIM: usize = 16;
const N_LAYERS: usize = 3;
const STATE_PER_LAYER: usize = 64;
const STATE_DIM: usize = N_LAYERS * STATE_PER_LAYER; // 192
const TICK_MS: u64 = 10; // 100 Hz control rate

/// Thin wrapper over an `ort` session with packed recurrent state.
pub struct ControllerWorker {
    session: ort::session::Session,
    session_state: [f32; SESS_DIM],
    hidden: [f32; STATE_DIM],
}

impl ControllerWorker {
    pub fn new(session_state: [f32; SESS_DIM]) -> Result<Self, ort::Error> {
        // `load-dynamic`: ORT is dlopen'd at runtime (ORT_DYLIB_PATH or the
        // dylib shipped alongside the plugin). See plugin/Cargo.toml.
        let session = ort::session::Session::builder()?
            .commit_from_memory(MODEL_BYTES)?;

        // Validate the contract up front so a bad artifact fails here, loudly,
        // instead of silently mis-feeding tensors at tick 1.
        let in_names: Vec<String> = session
            .inputs
            .iter()
            .map(|i| i.name.clone())
            .collect();
        for required in ["observer_features", "session_state", "hidden_state_in"] {
            if !in_names.iter().any(|n| n == required) {
                return Err(ort::Error::new(format!(
                    "controller ONNX missing input `{required}` — got {in_names:?}"
                )));
            }
        }

        Ok(Self {
            session,
            session_state,
            hidden: [0.0; STATE_DIM],
        })
    }

    /// One recurrent step. Returns the 32 tanh-space DSP parameters.
    pub fn run_step(&mut self, obs: &[f32; OBS_DIM]) -> Result<[f32; PARAM_COUNT], ort::Error> {
        // Pack inputs by DECLARED graph inputs: the three contract tensors are
        // filled from state; any extra input (legacy `onnx::Mul_*` leaks) is
        // zero-filled at its declared shape. This keeps the worker compatible
        // with both the legacy 5-input artifact and the clean 5A export.
        let mut feeds: Vec<(String, ort::value::Value)> = Vec::new();

        for input in &self.session.inputs {
            let dims: Vec<usize> = input
                .shape
                .iter()
                .map(|d| match d {
                    ort::session::InputDimensions::Fixed(n) => *n as usize,
                    ort::session::InputDimensions::Symbolic(_) => 1,
                    ort::session::InputDimensions::Unknown => 1,
                })
                .collect();
            let n: usize = dims.iter().product();

            let data: Vec<f32> = match input.name.as_str() {
                "observer_features" => obs.to_vec(),
                "session_state" => self.session_state.to_vec(),
                "hidden_state_in" => self.hidden.to_vec(),
                _ => vec![0.0f32; n.max(1)], // legacy leaked state inputs
            };
            let tensor = ort::value::Tensor::from_array((dims, data))?;
            feeds.push((input.name.clone(), tensor));
        }

        let outputs = self.session.run(feeds)?;

        // Read outputs BY NAME: `dsp_parameters` + `hidden_state_out`; any
        // legacy extra outputs (`onnx::Mul_*`) are ignored on purpose.
        let mut dsp = [0.0f32; PARAM_COUNT];
        {
            let (_, data) = outputs["dsp_parameters"].try_extract_tensor::<f32>()?;
            let n = data.len().min(PARAM_COUNT);
            dsp[..n].copy_from_slice(&data[..n]);
        }
        {
            let (_, data) = outputs["hidden_state_out"].try_extract_tensor::<f32>()?;
            let n = data.len().min(STATE_DIM);
            self.hidden[..n].copy_from_slice(&data[..n]);
        }
        Ok(dsp)
    }
}

/// Pack an `ObserverFrame` into the 12-D controller input, EXACTLY matching
/// `prompt_stack.compile_input_block()` normalization:
///   [0 mode(=Processor 4), 1 true_peak, 2 rms, 3 crest/10, 4 centroid/20000,
///    5 zcr/1024, 6 transient_slope, 7..11 mel band energies 0..4]
fn pack_frame(frame: &ObserverFrame) -> [f32; OBS_DIM] {
    [
        4.0, // operator mode: Processor (corrective EQ/comp/sat)
        frame.true_peak,
        frame.rms,
        frame.crest_factor / 10.0,
        frame.centroid / 20000.0,
        frame.zero_crossings as f32 / 1024.0,
        frame.transient_slope,
        frame.band_energies[0],
        frame.band_energies[1],
        frame.band_energies[2],
        frame.band_energies[3],
        frame.band_energies[4],
    ]
}

/// Worker thread entry point (spawned by `plugin.rs::initialize`).
pub fn run_worker(
    stop: Arc<AtomicBool>,
    feature_ring: Arc<SpscRingBuffer<ObserverFrame>>,
    param_buffer: Arc<TripleBuffer<DynamicDspParams>>,
    session_state: [f32; SESS_DIM],
) {
    let mut worker = match ControllerWorker::new(session_state) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("[cynical-controller] ONNX session failed to start: {e}");
            eprintln!("[cynical-controller] audio thread keeps neutral DSP params");
            return;
        }
    };
    let auditor = AuditorAgent::new(param_buffer);

    while !stop.load(Ordering::Relaxed) {
        let tick_start = Instant::now();

        // Drain the ring; the newest frame wins (control data is droppable).
        let mut latest: Option<ObserverFrame> = None;
        while let Some(frame) = feature_ring.pop() {
            latest = Some(frame);
        }

        if let Some(frame) = latest {
            let obs = pack_frame(&frame);
            if let Ok(out) = worker.run_step(&obs) {
                let params = DynamicDspParams::from_model_output(&out);
                auditor.verify_and_commit(params); // clamps + TripleBuffer write
            }
        }

        // Sleep out the rest of the 10 ms tick (worker thread — sleeping is fine)
        let elapsed = tick_start.elapsed();
        if elapsed < Duration::from_millis(TICK_MS) {
            std::thread::sleep(Duration::from_millis(TICK_MS) - elapsed);
        }
    }
}

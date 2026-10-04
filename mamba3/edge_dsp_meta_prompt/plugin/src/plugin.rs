//! The Cynical Engineer — CLAP plugin (nih-plug) with a hybrid control plane.
//!
//! Hard-RT audio thread (this file):
//!   zero-allocation Observer feature extraction → SPSC ring → classical DSP
//!   chain (low-cut, 6× ZDF bell EQ, dynamics, ADAA saturation) steered by the
//!   latest `DynamicDspParams` from a lock-free `TripleBuffer`.
//!
//! Background worker thread (`worker.rs`):
//!   runs the quantized Mamba-3 controller ONNX (single-step recurrent,
//!   100 Hz control rate) and commits verified parameters through the
//!   `AuditorAgent` (auditor.rs).
//!
//! This split is the Way 4B architecture: no neural inference ever touches the
//! audio callback, so the plugin adds 0 samples of latency and can never xrun
//! because of the model.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use edge_dsp_core::{
    AdaaSaturator, DynamicsProcessor, FilterType, Observer, ObserverFrame, SpscRingBuffer,
    TripleBuffer, ZdfSvf,
};
use nih_plug::prelude::*;

use crate::worker;

// ---------------------------------------------------------------------------
// DynamicDspParams — MUST stay in sync with
// compiler/train_controller.py::PARAM_LAYOUT and auditor.rs clamps.
// Model outputs are tanh-bounded [-1, 1]; from_model_output maps to physical.
// ---------------------------------------------------------------------------

pub const PARAM_COUNT: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct DynamicDspParams {
    pub eq_gains_db: [f32; 6],
    pub eq_freqs_hz: [f32; 6],
    pub eq_q: [f32; 6],
    pub comp_threshold_db: f32,
    pub comp_ratio: f32,
    pub comp_attack_ms: f32,
    pub comp_release_ms: f32,
    pub comp_mix: f32,
    pub sat_drive_db: f32,
    pub sat_mix: f32,
    pub output_gain_db: f32,
    pub low_cut_hz: f32,
    pub tilt_db: f32,
    pub presence_db: f32,
    pub gate_threshold_db: f32,
    pub warmth: f32,
    pub master_mix: f32,
}

impl Default for DynamicDspParams {
    fn default() -> Self {
        Self {
            eq_gains_db: [0.0; 6],
            eq_freqs_hz: [80.0, 240.0, 700.0, 2000.0, 5000.0, 11000.0],
            eq_q: [1.0; 6],
            comp_threshold_db: 0.0,
            comp_ratio: 1.0,
            comp_attack_ms: 10.0,
            comp_release_ms: 120.0,
            comp_mix: 0.0,
            sat_drive_db: 0.0,
            sat_mix: 0.0,
            output_gain_db: 0.0,
            low_cut_hz: 20.0,
            tilt_db: 0.0,
            presence_db: 0.0,
            gate_threshold_db: -80.0,
            warmth: 0.0,
            master_mix: 1.0,
        }
    }
}

fn lin_map(v: f32, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * (v.clamp(-1.0, 1.0) + 1.0) / 2.0
}

fn log_map(v: f32, lo: f32, hi: f32) -> f32 {
    let t = (v.clamp(-1.0, 1.0) + 1.0) / 2.0;
    lo * (hi / lo).powf(t)
}

impl DynamicDspParams {
    /// Denormalize the 32 tanh-space model outputs → physical parameters.
    /// Mirrors `train_controller.py::denormalize` exactly.
    pub fn from_model_output(out: &[f32; PARAM_COUNT]) -> Self {
        let mut p = Self::default();
        let mut i = 0usize;
        {
            let s = &out[i..i + 6];
            i += 6;
            for (k, v) in s.iter().enumerate() {
                p.eq_gains_db[k] = lin_map(*v, -9.0, 9.0);
            }
        }
        {
            let s = &out[i..i + 6];
            i += 6;
            for (k, v) in s.iter().enumerate() {
                p.eq_freqs_hz[k] = log_map(*v, 40.0, 12000.0);
            }
        }
        {
            let s = &out[i..i + 6];
            i += 6;
            for (k, v) in s.iter().enumerate() {
                p.eq_q[k] = log_map(*v, 0.3, 8.0);
            }
        }
        p.comp_threshold_db = lin_map(out[i], -60.0, 0.0);
        i += 1;
        p.comp_ratio = lin_map(out[i], 1.0, 12.0);
        i += 1;
        p.comp_attack_ms = log_map(out[i], 0.1, 100.0);
        i += 1;
        p.comp_release_ms = log_map(out[i], 5.0, 1000.0);
        i += 1;
        p.comp_mix = lin_map(out[i], 0.0, 1.0);
        i += 1;
        p.sat_drive_db = lin_map(out[i], 0.0, 12.0); // auditor clamps ≤ 8
        i += 1;
        p.sat_mix = lin_map(out[i], 0.0, 1.0);
        i += 1;
        p.output_gain_db = lin_map(out[i], -12.0, 12.0);
        i += 1;
        p.low_cut_hz = log_map(out[i], 20.0, 120.0);
        i += 1;
        p.tilt_db = lin_map(out[i], -6.0, 6.0);
        i += 1;
        p.presence_db = lin_map(out[i], -6.0, 6.0);
        i += 1;
        p.gate_threshold_db = lin_map(out[i], -80.0, -20.0);
        i += 1;
        p.warmth = lin_map(out[i], 0.0, 1.0);
        i += 1;
        p.master_mix = lin_map(out[i], 0.0, 1.0);
        p
    }

    /// Tilt (broad spectral slope) + presence bias applied on top of the six
    /// band gains before hitting the EQ — keeps the model's 32-D contract while
    /// reusing the 6-band bell engine.
    pub fn effective_eq_gains(&self) -> [f32; 6] {
        let mut g = self.eq_gains_db;
        for (k, band) in g.iter_mut().enumerate() {
            let tilt = self.tilt_db * ((k as f32) / 5.0 - 0.5) * 2.0; // -tilt..+tilt
            let presence = if k == 4 { self.presence_db } else { 0.0 };
            *band = (band + tilt + presence).clamp(-9.0, 9.0);
        }
        g
    }
}

// ---------------------------------------------------------------------------
// DSP chain (audio thread)
// ---------------------------------------------------------------------------

struct DspChain {
    low_cut: ZdfSvf,
    bells: [ZdfSvf; 6],
    dynamics: DynamicsProcessor,
    sat: edge_dsp_core::AdaaSaturator,
}

impl DspChain {
    fn new() -> Self {
        Self {
            low_cut: ZdfSvf::new(),
            bells: Default::default(),
            dynamics: DynamicsProcessor::new(),
            sat: AdaaSaturator::new(),
        }
    }

    fn reset(&mut self) {
        self.low_cut.reset();
        for b in self.bells.iter_mut() {
            b.reset();
        }
        self.dynamics.reset();
        self.sat.reset();
    }

    /// Push control-rate params once per block.
    fn set_params(&mut self, p: &DynamicDspParams, sample_rate: f32) {
        self.dynamics.set_params(
            p.comp_threshold_db,
            p.comp_ratio,
            p.comp_attack_ms,
            p.comp_release_ms,
            p.gate_threshold_db,
            p.comp_mix,
            sample_rate,
        );
    }

    /// Per-sample chain: low-cut → 6× bell → dynamics → saturation → output.
    #[inline(always)]
    fn process_sample(&mut self, x: f32, p: &DynamicDspParams, gains: &[f32; 6], sr: f32) -> f32 {
        let mut y = self
            .low_cut
            .process(x, p.low_cut_hz.clamp(20.0, 200.0), 0.707, &FilterType::HighPass, sr);
        for k in 0..6 {
            y = self.bells[k].process(
                y,
                p.eq_freqs_hz[k].clamp(20.0, sr * 0.45),
                p.eq_q[k].clamp(0.2, 10.0),
                &FilterType::Bell {
                    gain_db: gains[k],
                },
                sr,
            );
        }
        y = self.dynamics.process_sample(y);
        y = self.sat.process(y, p.sat_drive_db.min(8.0) + 2.0 * p.warmth, p.sat_mix);
        let out_gain = 10.0f32.powf(p.output_gain_db / 20.0);
        y * out_gain
    }
}

// ---------------------------------------------------------------------------
// Plugin parameters (user-facing)
// ---------------------------------------------------------------------------

#[derive(Params)]
pub struct PluginParams {
    #[id = "bypass"]
    pub bypass: BoolParam,

    /// Global wet ceiling on top of the model's master_mix (audition safety).
    #[id = "intensity"]
    pub intensity: FloatParam,

    /// Session type hint (compiled into session_state[0:4] by the worker).
    #[id = "tracktype"]
    pub track_type: EnumParam<TrackType>,
}

#[derive(Enum, PartialEq, Eq, Clone, Copy, Debug)]
pub enum TrackType {
    #[id = "vocal"]
    Vocal,
    #[id = "drums"]
    Drums,
    #[id = "bass"]
    Bass,
    #[id = "melodic"]
    Melodic,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct CynicalEnginePlugin {
    params: Arc<PluginParams>,
    sample_rate: f32,

    observer: Observer,
    feature_ring: Arc<SpscRingBuffer<ObserverFrame>>,
    param_buffer: Arc<TripleBuffer<DynamicDspParams>>,
    chain: DspChain,

    worker_stop: Option<Arc<AtomicBool>>,
    worker_join: Option<std::thread::JoinHandle<()>>,
}

impl Default for CynicalEnginePlugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        Self {
            params,
            sample_rate: 48000.0,
            // Observer block = 512 samples ≈ 10.7 ms @ 48 kHz ≈ the 100 Hz
            // control tick of the worker.
            observer: Observer::new(512, 48000.0),
            feature_ring: Arc::new(SpscRingBuffer::new(256)),
            param_buffer: Arc::new(TripleBuffer::new(DynamicDspParams::default())),
            chain: DspChain::new(),
            worker_stop: None,
            worker_join: None,
        }
    }
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
            bypass: BoolParam::new("Bypass", false),
            intensity: FloatParam::new("Intensity", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(20.0))
                .with_unit("%"),
            track_type: EnumParam::new("Track Type", TrackType::Vocal),
        }
    }
}

impl CynicalEnginePlugin {
    fn stop_worker(&mut self) {
        if let Some(stop) = self.worker_stop.take() {
            stop.store(true, Ordering::SeqCst);
        }
        if let Some(join) = self.worker_join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for CynicalEnginePlugin {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

impl Plugin for CynicalEnginePlugin {
    const CLAP_ID: &'static str = "com.sgm-audio.cynical-engineer";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Real-time Mamba-3 steered adaptive channel strip");
    const CLAP_MANUAL_URL: Option<&'static str> = Some("https://github.com/sgm-audio/mamba3");
    const CLAP_SUPPORT_URL: Option<&'static str> = Some("https://github.com/sgm-audio/mamba3/issues");
    const CLAP_VENDOR: &'static str = "sgm-audio";
    const CLAP_URL: &'static str = "https://github.com/sgm-audio/mamba3";
    const CLAP_VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&mut self) -> &mut impl Params {
        &mut self.params
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> Result<(), BufferConfigSupportedError> {
        self.sample_rate = buffer_config.sample_rate;
        self.observer = Observer::new(512, buffer_config.sample_rate);
        self.chain.reset();

        // ── Spawn the background worker (Way 4B: model never touches audio) ──
        self.stop_worker();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let ring = self.feature_ring.clone();
        let params_buf = self.param_buffer.clone();

        // session_state skeleton: context slots (track type, tempo neutral,
        // 44.1k-normalized SR) + 6C profile slots 8..16 (zeros until a preset
        // carries a fitted profile).
        let mut session_state = [0.0f32; 16];
        match self.params.track_type.value() {
            TrackType::Vocal => session_state[0] = 1.0,
            TrackType::Drums => session_state[1] = 1.0,
            TrackType::Bass => session_state[2] = 1.0,
            TrackType::Melodic => session_state[3] = 1.0,
        }
        session_state[4] = 120.0 / 200.0;
        session_state[6] = buffer_config.sample_rate / 96000.0;
        // session_state[8..16] = customer profile (6C) — zeros = neutral

        let join = std::thread::Builder::new()
            .name("cynical-controller".into())
            .spawn(move || {
                worker::run_worker(stop2, ring, params_buf, session_state);
            })
            .expect("failed to spawn controller worker thread");

        self.worker_stop = Some(stop);
        self.worker_join = Some(join);
        Ok(())
    }

    fn reset(&mut self) {
        self.observer = Observer::new(512, self.sample_rate);
        self.chain.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _ctx: &mut impl ProcessContext,
    ) -> ProcessStatus {
        if self.params.bypass.value() {
            return ProcessStatus::Normal; // dry passthrough
        }

        // Snapshot control params ONCE per block (control-rate world).
        let p = *self.param_buffer.read();
        let intensity = self.params.intensity.value();
        let mut effective = p;
        effective.master_mix = (p.master_mix * intensity).clamp(0.0, 1.0);
        let gains = effective.effective_eq_gains();
        self.chain.set_params(&effective, self.sample_rate);

        for mut channel_samples in buffer.iter_samples() {
            let left = channel_samples[0];
            let right = channel_samples[1];

            // Observer feed: mono analysis on (L+R)/2
            let mid = 0.5 * (left + right);
            if let Some(frame) = self.observer.process_sample(mid) {
                // Control data is droppable: on a full ring the newest frame
                // wins and the stale one is discarded — never blocks.
                let _ = self.feature_ring.push(frame);
            }

            // Parallel stereo chain (shared params; per-channel filter state)
            let dry_l = left;
            let dry_r = right;
            let wet_l = self.chain.process_sample(left, &effective, &gains, self.sample_rate);
            let wet_r = self.chain.process_sample(right, &effective, &gains, self.sample_rate);
            let mix = effective.master_mix;
            channel_samples[0] = (1.0 - mix) * dry_l + mix * wet_l;
            channel_samples[1] = (1.0 - mix) * dry_r + mix * wet_r;
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for CynicalEnginePlugin {
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

nih_export_clap!(CynicalEnginePlugin);

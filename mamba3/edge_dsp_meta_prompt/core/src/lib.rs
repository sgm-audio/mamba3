//! `edge_dsp_core` — hard real-time DSP primitives for the Cynical Engineer plugin.
//!
//! Everything in this crate must obey the audio-thread contract (see
//! REFERENCE_THIS_SHIT.md § "Hard RT Domain"):
//!   - zero heap allocation after construction
//!   - no locks, no syscalls, no logging
//!   - O(1) per sample, no unbounded loops
//!
//! Modules:
//!   - [`ring_buffer`]  — lock-free cache-padded SPSC ring (audio → worker)
//!   - [`double_buffer`] — lock-free `TripleBuffer` parameter swap (worker → audio)
//!   - [`observer`]     — zero-allocation 12-D feature extractor
//!   - [`dsp_engine`]   — `ZdfSvf` (bilinear ZDF SVF) + `AdaaSaturator` (ADAA)
//!   - [`dynamics`]     — feed-forward compressor/gate (control-rate smoothed)
//!   - [`slice_engine`] — zero-crossing waveform slice segmenter

pub mod ring_buffer;
pub mod double_buffer;
pub mod observer;
pub mod dsp_engine;
pub mod dynamics;
pub mod slice_engine;

// Re-exports used across crate boundaries (plugin/auditor.rs does
// `use edge_dsp_core::TripleBuffer;`).
pub use double_buffer::TripleBuffer;
pub use dsp_engine::{AdaaSaturator, FilterType, ZdfSvf};
pub use dynamics::DynamicsProcessor;
pub use observer::{Observer, ObserverFrame};
pub use ring_buffer::SpscRingBuffer;
pub use slice_engine::{SliceDescriptor, SliceEngine};

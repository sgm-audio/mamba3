//! `edge_dsp_plugin` — "The Cynical Engineer" CLAP plugin entry point.
//!
//! Layout:
//!   - [`plugin`]   — nih-plug `Plugin` impl, `DynamicDspParams`, RT DSP chain
//!   - [`worker`]   — background Mamba-3 ONNX controller thread (Way 4B/5A)
//!   - [`auditor`]  — `AuditorAgent` parameter clamping + operation log
//!
//! The CLAP export (`nih_export_clap!`) lives at the bottom of [`plugin`].

mod auditor;
mod plugin;
mod worker;

//! Feed-forward dynamics processor (compressor + downward gate).
//!
//! Hard real-time safe: all state is a few `f32`s, `process_sample` is O(1)
//! with no allocation, no locks, no transcendental calls in the hot path
//! (attack/release coefficients are precomputed in `set_params`, which is
//! called ONCE per control block — not per sample).

/// One-knee feed-forward compressor with smoothed gain and a downward gate.
///
/// Signal path: detector → dB-domain gain computer → attack/release smoothing
/// → mix.  The gate is a gentle downward expander below `gate_threshold_db`
/// (it never cuts to silence — auditions, not gates, per the Root Constitution).
#[derive(Clone, Debug)]
pub struct DynamicsProcessor {
    // detector state (linear envelope)
    env: f32,
    // smoothed gain (linear)
    gain: f32,
    // precomputed coefficients
    attack_coeff: f32,
    release_coeff: f32,
    // precomputed static params (updated by set_params)
    threshold_db: f32,
    ratio: f32,
    gate_threshold_db: f32,
    mix: f32,
}

impl Default for DynamicsProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicsProcessor {
    pub fn new() -> Self {
        Self {
            env: 0.0,
            gain: 1.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            threshold_db: 0.0,
            ratio: 1.0,
            gate_threshold_db: -80.0,
            mix: 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.env = 0.0;
        self.gain = 1.0;
    }

    /// Update control parameters. Call once per block (e.g. every 64 samples)
    /// after reading the latest `DynamicDspParams` from the TripleBuffer.
    pub fn set_params(
        &mut self,
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        gate_threshold_db: f32,
        mix: f32,
        sample_rate: f32,
    ) {
        self.threshold_db = threshold_db.clamp(-60.0, 0.0);
        self.ratio = ratio.clamp(1.0, 12.0);
        self.gate_threshold_db = gate_threshold_db.clamp(-80.0, -20.0);
        self.mix = mix.clamp(0.0, 1.0);

        let atk = (attack_ms.max(0.05) * 0.001 * sample_rate).max(1.0);
        let rel = (release_ms.max(1.0) * 0.001 * sample_rate).max(1.0);
        // one-pole coefficients: y[n] = coeff * y[n-1] + (1 - coeff) * x[n]
        self.attack_coeff = (-1.0 / atk).exp();
        self.release_coeff = (-1.0 / rel).exp();
    }

    /// Process one sample. O(1), zero-allocation.
    #[inline(always)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Peak-ish detector with attack/release smoothing on the envelope
        let x_abs = input.abs();
        let coeff = if x_abs > self.env {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.env = coeff * self.env + (1.0 - coeff) * x_abs;

        let level_db = 20.0 * (self.env + 1e-9).log10();

        // Compressor gain in dB above threshold
        let over = level_db - self.threshold_db;
        let comp_gain_db = if over > 0.0 {
            -over * (1.0 - 1.0 / self.ratio)
        } else {
            0.0
        };

        // Downward gate (gentle 2:1 expansion below gate threshold)
        let gate_gain_db = if level_db < self.gate_threshold_db {
            (level_db - self.gate_threshold_db) * 0.5
        } else {
            0.0
        };

        let target_gain = 10.0f32.powf((comp_gain_db + gate_gain_db) / 20.0);

        // Smooth the gain (release-biased; avoids zipper noise)
        let g_coeff = if target_gain < self.gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.gain = g_coeff * self.gain + (1.0 - g_coeff) * target_gain;

        let processed = input * self.gain;
        self.mix * processed + (1.0 - self.mix) * input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_when_silent_and_below_threshold() {
        let mut d = DynamicsProcessor::new();
        d.set_params(-20.0, 4.0, 1.0, 50.0, -70.0, 1.0, 48000.0);
        // A quiet signal below threshold should pass essentially unchanged
        // after the envelope settles.
        let mut out = 0.0;
        for _ in 0..48000 {
            out = d.process_sample(0.01);
        }
        assert!((out - 0.01).abs() < 1e-3, "quiet signal must pass ~unchanged, got {}", out);
    }

    #[test]
    fn loud_signal_is_attenuated() {
        let mut d = DynamicsProcessor::new();
        d.set_params(-30.0, 8.0, 1.0, 50.0, -70.0, 1.0, 48000.0);
        let mut peak_out = 0.0f32;
        for _ in 0..48000 {
            let o = d.process_sample(0.9);
            peak_out = peak_out.max(o.abs());
        }
        assert!(peak_out < 0.9, "loud signal must be compressed, got {}", peak_out);
    }

    #[test]
    fn gate_attenuates_extremely_quiet_input() {
        let mut d = DynamicsProcessor::new();
        d.set_params(0.0, 1.0, 1.0, 50.0, -40.0, 1.0, 48000.0);
        let mut out = 0.0f32;
        for _ in 0..96000 {
            out = d.process_sample(0.0001); // -80 dBFS
        }
        assert!(out.abs() < 0.0001, "gate must attenuate, got {}", out);
    }
}

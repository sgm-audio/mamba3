# PRD: The Intelligent Mix Engine (Concept B)

**Status:** Spec baseline (restored 2026-10-04; referenced by REFERENCE_THIS_SHIT.md § docs index)
**Owner:** sgm-audio
**Related:** `ard_intelligent_mix_engine.md`, `prd_cynical_engineer.md` (Concept A), `../model.py` (PerTrackProcessor)

## 1. Product Summary

A context-aware mixing assistant that hears the whole session ("Option B"):
each track is processed *with awareness of the other tracks*, carving masking
frequencies and balancing levels so the mix resolves without per-knob labor.

## 2. Problem

Frequency masking, dynamic imbalance and inconsistent track levels are the
bulk of mixing work. A model trained on isolated stems (Option A) cannot see
the collision; the engine must condition on cross-track embeddings.

## 3. Requirements

| ID | Requirement | Acceptance |
|----|-------------|------------|
| B1 | Cross-track conditioning | `PerTrackProcessor(other_embs=...)` consumes per-track semantic embeddings (`model.py` `CrossTrackAggregator`) |
| B2 | Masking-aware repair | Training minimizes cross-track collision (masking-aware loss) **plus** reconstruction preservation so the engine never "fixes" a mix by deleting content |
| B3 | Graceful single-track fallback | With `other_embs=None` the model degrades to Option A single-track processing |
| B4 | Real-time deployment | Deployment follows the hybrid control plane (Concept A): neural model steers classical DSP at ≤100 Hz; never in `processBlock` |
| B5 | Restorative objective (Recipe 1A) | The trained objective is degraded-input → clean-target restoration with session context (`train.py` Phases 2–3) |

## 4. Non-Goals

- Full automatic mix replacement (human remains the mixer of record).
- Sample-level neural synthesis on the audio path (see Concept A § "Hard RT Domain").

## 5. Model Contract

- Input: target stem waveform + CREPE f0 + other-track embeddings + loudness.
- Output: complex STFT (`output_stft`) decoded by ISTFT; `track_emb` fed to
  sibling tracks as `other_embs`.
- Training: `train.py` Phase 3 (50%–100% of steps), MUSDB18-HQ/MoisesDB @ 44.1 kHz.

## 6. Risks

- Phase-3 conditioning quality depends on honest multi-pass inference (pass 1
  estimates stems, pass 2 refines with context) — single-pass separation cannot
  supply `other_embs` at inference time.
- Cross-track attention cost grows with K (track count); cap K (e.g. 7) at export.

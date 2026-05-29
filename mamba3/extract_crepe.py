"""
CREPE Offline F0 Extraction for MUSDB18-HQ
============================================
Pre-computes 360-D pitch probability tensors for every stem in MUSDB18-HQ,
saves as float16 .npy files alongside the WAV files.

Why offline:
  - Zero training-loop overhead (vs ~2x slowdown if done online)
  - Resampling 44.1k→16k done once, not per epoch
  - Resumable: skips files that already have .crepe.npy
  - One-time ~30-60 min cost on 860M for full dataset

Usage:
  python extract_crepe.py --data ~/datasets/musdb18hq/train --device cuda
  python extract_crepe.py --data ~/datasets/musdb18hq/test  --device cuda

Output:
  For each <stem>.wav, writes <stem>.crepe.npy of shape (T_model, 360) float16,
  where T_model matches the model's STFT frame rate (1 + L/hop @ 44100 Hz, hop=512).

  Also writes <stem>.pitched.npy as a single float (1.0 or 0.0) indicating
  whether the stem is pitched (used by FiLM to gate f0 contribution).

Stems handled:
  - vocals.wav, bass.wav, other.wav     → real CREPE inference, pitched=1.0
  - drums.wav                            → zeros tensor, pitched=0.0 (percussive)
  - mixture.wav                          → skipped (model only sees stems)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf

try:
    import torchcrepe
except ImportError:
    print("ERROR: torchcrepe not installed. Run: pip install torchcrepe")
    sys.exit(1)


# ── constants ─────────────────────────────────────────────────────────────

CREPE_SR     = 16000      # CREPE's native sample rate
CREPE_HOP_MS = 10         # 10ms hop → 100 Hz frame rate
CREPE_BINS   = 360        # 20-cent resolution from C1 to B7

MODEL_SR     = 44100      # MUSDB18-HQ native rate
MODEL_HOP    = 512        # train.py default hop_length
# Model frame rate: 44100/512 ≈ 86.13 Hz

# Stems that get real CREPE (pitched)
PITCHED_STEMS = {'vocals', 'bass', 'other', 'guitar', 'piano', 'keys'}
# Stems that get zero tensor (percussion / unpitched)
UNPITCHED_STEMS = {'drums', 'percussion'}
# Stems to skip entirely (not used by training)
SKIP_STEMS = {'mixture'}


# ── audio loading ─────────────────────────────────────────────────────────

def load_wav_mono_16k(path: Path) -> Optional[torch.Tensor]:
    """Load WAV, mix to mono, resample to 16 kHz for CREPE.

    Returns float32 tensor of shape (L,) at 16 kHz, or None on failure.
    """
    try:
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception as e:
        print(f"    ! Failed to read {path.name}: {e}")
        return None

    # Mix to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    wav = torch.from_numpy(data)

    # Resample to 16 kHz
    if sr != CREPE_SR:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, CREPE_SR
        ).squeeze(0)

    return wav


# ── CREPE inference ───────────────────────────────────────────────────────

@torch.no_grad()
def crepe_extract(
    wav_16k: torch.Tensor,
    device:  torch.device,
    model_capacity: str = 'full',
    batch_size: int = 512,
) -> np.ndarray:
    """Run CREPE inference, return (T_crepe, 360) float16 probability tensor.

    wav_16k: (L,) float32 at 16 kHz
    model_capacity: 'tiny' (6 MB, fast) or 'full' (85 MB, highest accuracy).
                    These are the only two sizes torchcrepe ships.
    batch_size: how many 1024-sample frames to process at once on GPU

    Implementation note: torchcrepe.preprocess returns a generator that yields
    batches of frames. We iterate it and call infer(embed=False) on each batch,
    then stack the per-batch (B, 360) outputs into (T_total, 360).
    """
    # torchcrepe expects (1, L)
    wav_16k = wav_16k.to(device).unsqueeze(0)

    hop_length = CREPE_SR // 100  # 10 ms = 160 samples
    all_probs = []

    generator = torchcrepe.preprocess(
        wav_16k,
        sample_rate=CREPE_SR,
        hop_length=hop_length,
        batch_size=batch_size,
        device=str(device),
        pad=True,
    )

    for frames in generator:
        # frames: (B, 1024) where B is batch_size of frames
        probs = torchcrepe.infer(frames, model=model_capacity, device=str(device), embed=False)
        # probs: (B, 360)
        all_probs.append(probs.cpu())

    # Concatenate along time axis: (T_total, 360)
    return torch.cat(all_probs, dim=0).half().numpy()


def resample_frame_rate(
    crepe_probs: np.ndarray,   # (T_crepe, 360) @ 100 Hz
    wav_44k_length: int,       # original 44.1k sample count
) -> np.ndarray:
    """Resample CREPE's 100 Hz frames to model's 86.13 Hz frames.

    Output T_model = 1 + wav_44k_length // MODEL_HOP  (matches torch.stft center=True)
    """
    T_model = 1 + wav_44k_length // MODEL_HOP
    T_crepe = crepe_probs.shape[0]

    # Linear interpolation along time axis
    # x_old: original frame positions (0 to T_crepe-1)
    # x_new: positions to sample at (0 to T_crepe-1 mapped to T_model points)
    x_old = np.linspace(0, T_crepe - 1, T_crepe)
    x_new = np.linspace(0, T_crepe - 1, T_model)

    # Interpolate each of 360 bins independently
    # Result: (T_model, 360)
    out = np.zeros((T_model, CREPE_BINS), dtype=np.float16)
    for bin_idx in range(CREPE_BINS):
        out[:, bin_idx] = np.interp(x_new, x_old, crepe_probs[:, bin_idx]).astype(np.float16)

    return out


# ── per-stem processing ───────────────────────────────────────────────────

def process_stem(
    stem_path:  Path,
    device:     torch.device,
    model_size: str,
    force:      bool = False,
) -> str:
    """Process one stem file. Returns status string for logging."""
    name = stem_path.stem.lower()  # 'vocals', 'drums', etc.

    if name in SKIP_STEMS:
        return 'skip-mixture'

    crepe_path   = stem_path.parent / f"{stem_path.stem}.crepe.npy"
    pitched_path = stem_path.parent / f"{stem_path.stem}.pitched.npy"

    if not force and crepe_path.exists() and pitched_path.exists():
        return 'cached'

    # Load original 44.1k to know the length (for frame rate matching)
    try:
        info = sf.info(str(stem_path))
        wav_44k_length = info.frames
    except Exception as e:
        return f'fail-info ({e})'

    # Unpitched stems: write zeros tensor + pitched=0
    if name in UNPITCHED_STEMS:
        T_model = 1 + wav_44k_length // MODEL_HOP
        zeros = np.zeros((T_model, CREPE_BINS), dtype=np.float16)
        np.save(crepe_path, zeros)
        np.save(pitched_path, np.array(0.0, dtype=np.float32))
        return 'unpitched (zeros)'

    # Pitched stems: real CREPE
    if name not in PITCHED_STEMS:
        # Unknown stem name — treat as pitched (best effort)
        print(f"    ? Unknown stem name '{name}', treating as pitched")

    wav_16k = load_wav_mono_16k(stem_path)
    if wav_16k is None:
        return 'fail-load'

    try:
        crepe_probs = crepe_extract(wav_16k, device, model_capacity=model_size)
    except Exception as e:
        return f'fail-crepe ({type(e).__name__}: {e})'

    # Resample 100 Hz → 86 Hz to match model frame rate
    crepe_resampled = resample_frame_rate(crepe_probs, wav_44k_length)

    np.save(crepe_path, crepe_resampled)
    np.save(pitched_path, np.array(1.0, dtype=np.float32))

    return f'extracted ({crepe_resampled.shape[0]} frames)'


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline CREPE f0 extraction")
    parser.add_argument('--data',       required=True,
                        help='Path to MUSDB18-HQ train or test directory')
    parser.add_argument('--device',     default='cuda',
                        help='cuda (ROCm) or cpu')
    parser.add_argument('--model',      default='full',
                        choices=['tiny', 'full'],
                        help='torchcrepe only ships tiny (~6 MB) and full (~85 MB). '
                             'full is the highest accuracy available.')
    parser.add_argument('--force',      action='store_true',
                        help='Re-extract even if .crepe.npy already exists')
    parser.add_argument('--limit',      type=int, default=0,
                        help='Process only N tracks (0=all, for testing)')
    args = parser.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        sys.exit(1)

    # Discover tracks
    tracks = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.limit > 0:
        tracks = tracks[:args.limit]

    print(f"Found {len(tracks)} tracks under {root}")
    print(f"CREPE model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: <stem>.crepe.npy (fp16) + <stem>.pitched.npy alongside each WAV")
    print("─" * 60)

    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not available, falling back to CPU")
        args.device = 'cpu'
    dev = torch.device(args.device)

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Warm up CREPE (first call triggers download + MIOpen kernel compile)
    print("\nWarming up CREPE (first call triggers ~30s kernel compile)...")
    t0 = time.time()
    dummy = torch.zeros(CREPE_SR, device=dev)  # 1 second of silence
    _ = crepe_extract(dummy, dev, model_capacity=args.model)
    print(f"  warmup done in {time.time()-t0:.1f}s\n")

    # Process all stems
    total_stems = 0
    extracted   = 0
    cached      = 0
    failed      = 0
    t_start     = time.time()

    for i, track_dir in enumerate(tracks):
        wavs = sorted(track_dir.glob("*.wav"))
        print(f"[{i+1:3d}/{len(tracks)}] {track_dir.name}")

        for wav_path in wavs:
            total_stems += 1
            t0 = time.time()
            status = process_stem(wav_path, dev, args.model, args.force)
            dt = time.time() - t0

            if 'extracted' in status:
                extracted += 1
                marker = '✓'
            elif status == 'cached':
                cached += 1
                marker = '·'
            elif 'unpitched' in status:
                extracted += 1
                marker = '○'
            elif status == 'skip-mixture':
                marker = '–'
            else:
                failed += 1
                marker = '✗'

            print(f"    {marker} {wav_path.name:25s} {status:30s} ({dt:.1f}s)")

    elapsed = time.time() - t_start
    print("─" * 60)
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Extracted: {extracted}")
    print(f"  Cached:    {cached}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total_stems}")

    if failed > 0:
        print("\n! Some stems failed. Re-run to retry (cached ones skip automatically).")
        sys.exit(1)


if __name__ == '__main__':
    main()

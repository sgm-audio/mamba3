"""
Training recipe for Mamba-3 Per-Track Adaptive Filter Bank.

Loss stack:
  - Multi-resolution STFT (auraloss) — tonal + transient accuracy
  - Mel-L1 — perceptual spectral shape
  - SI-SDR — time-domain signal ratio (ramped in after warmup)
  - NT-Xent contrastive — semantic encoder pretraining

Dataset: MUSDB18-HQ stems (vocals/drums/bass/other)
         MoisesDB extended stems (guitar, piano, etc.)
         Synthetic degradation pairs (EQ + compression + reverb)

AMD ROCm gfx1103 safe: no CUDA-specific ops.
All deps: Apache 2.0 / MIT licensed.
"""

from __future__ import annotations

import random
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import torchaudio

# pip install auraloss — Apache 2.0
# pip install torchcrepe — MIT
# pip install musdb — MIT (dataset loader)
try:
    import auraloss.freq as alf
    HAS_AURALOSS = True
except ImportError:
    HAS_AURALOSS = False
    print("WARNING: auraloss not found. Install: pip install auraloss")

from model import PerTrackProcessor, TrackProcessorConfig


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TrackProcessorLoss(nn.Module):
    """
    Multi-objective loss for context-aware per-track processing.

    Weights schedule:
      Steps 0-5k:    MR-STFT + Mel-L1 only (stable foundation)
      Steps 5k-15k:  + SI-SDR (ramped 0→0.1)
      Steps 15k+:    full weighted sum

    This matches L-Acoustics HS-TasNet finding that SI-SDR destabilizes
    early training when used at full weight from step 0.
    """

    def __init__(self, cfg: TrackProcessorConfig, sample_rate: int = 44100):
        super().__init__()
        self.cfg = cfg
        self.sr  = sample_rate

        # Multi-resolution STFT loss (auraloss)
        if HAS_AURALOSS:
            self.mr_stft = alf.MultiResolutionSTFTLoss(
                fft_sizes   = [512, 1024, 2048, 4096],
                hop_sizes   = [128, 256,  512,  1024],
                win_lengths = [512, 1024, 2048, 4096],
                w_sc = 1.0,   # spectral convergence weight
                w_log_mag = 1.0,
                sample_rate = sample_rate,
            )
        else:
            self.mr_stft = None

        # Mel filterbank for mel-L1 (3 scales)
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate = sample_rate,
                n_fft       = n,
                hop_length  = n // 4,
                n_mels      = n_mels,
            )
            for n, n_mels in [(512, 64), (1024, 80), (2048, 128)]
        ])

        self.global_step = 0

    def _si_sdr(self, pred: Tensor, target: Tensor) -> Tensor:
        """SI-SDR in dB. pred/target: (B, L)"""
        eps = 1e-8
        target_energy = (target ** 2).sum(dim=-1, keepdim=True)
        dot = (pred * target).sum(dim=-1, keepdim=True)
        proj = dot / (target_energy + eps) * target
        noise = pred - proj
        si_sdr = 10 * torch.log10(
            (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps
        )
        return -si_sdr.mean()  # negate: minimize = maximize SI-SDR

    def _mel_l1(self, pred_wave: Tensor, target_wave: Tensor) -> Tensor:
        loss = 0.0
        for mel_tf in self.mel_transforms:
            mel_tf = mel_tf.to(pred_wave.device)
            p = torch.log1p(mel_tf(pred_wave))
            t = torch.log1p(mel_tf(target_wave))
            loss = loss + F.l1_loss(p, t)
        return loss / len(self.mel_transforms)

    def _si_sdr_weight(self) -> float:
        """Ramp SI-SDR weight from 0 → 0.1 between steps 5k and 15k."""
        if self.global_step < 5000:
            return 0.0
        elif self.global_step > 15000:
            return 0.1
        else:
            return 0.1 * (self.global_step - 5000) / 10000

    def forward(
        self,
        pred_stft:   Tensor,    # (B, T, F) complex — model output
        target_wave: Tensor,    # (B, L)
        model:       nn.Module, # for ISTFT
    ) -> dict[str, Tensor]:

        # Reconstruct waveform from predicted STFT
        pred_wave = model.waveform_from_stft(pred_stft)

        # Align lengths (ISTFT may differ by a few samples)
        L = min(pred_wave.shape[-1], target_wave.shape[-1])
        pred_wave   = pred_wave[..., :L]
        target_wave = target_wave[..., :L]

        losses = {}

        # 1. Multi-resolution STFT
        if self.mr_stft is not None:
            losses['mr_stft'] = self.mr_stft(
                pred_wave.unsqueeze(1),
                target_wave.unsqueeze(1)
            )
        else:
            # Fallback: single-res STFT L1
            window = torch.hann_window(2048, device=pred_wave.device)
            p_stft = torch.stft(pred_wave, 2048, 512, window=window, return_complex=True)
            t_stft = torch.stft(target_wave, 2048, 512, window=window, return_complex=True)
            losses['mr_stft'] = F.l1_loss(p_stft.abs(), t_stft.abs())

        # 2. Mel-L1 (3 scales)
        losses['mel_l1'] = self._mel_l1(pred_wave, target_wave)

        # 3. SI-SDR (ramped)
        si_sdr_w = self._si_sdr_weight()
        if si_sdr_w > 0:
            losses['si_sdr'] = self._si_sdr(pred_wave, target_wave) * si_sdr_w
        else:
            losses['si_sdr'] = torch.tensor(0.0, device=pred_wave.device)

        # Total
        losses['total'] = (
            1.0 * losses['mr_stft'] +
            0.5 * losses['mel_l1'] +
            losses['si_sdr']
        )

        self.global_step += 1
        return losses


class NTXentLoss(nn.Module):
    """
    NT-Xent contrastive loss for semantic encoder pretraining.
    Two augmented views of same track = positive pair.
    All other tracks in batch = negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """z1, z2: (B, d) — L2-normalized embeddings"""
        B = z1.shape[0]
        z  = torch.cat([z1, z2], dim=0)           # (2B, d)
        sim = torch.mm(z, z.T) / self.T            # (2B, 2B)

        # Labels: for each z1[i], positive is z2[i] = z[B+i]
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels])  # (2B,)

        # Mask self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StemDataset(Dataset):
    """
    Multi-stem dataset for per-track processing training.

    Loads stems from MUSDB18-HQ or MoisesDB directory structure.
    Returns (target_stem, mixture, other_stems_list, metadata).

    Expected directory layout:
      root/
        track_001/
          bass.wav
          drums.wav
          vocals.wav
          other.wav
          mixture.wav
        track_002/
          ...

    Augmentations (applied during __getitem__):
      - Random gain ±6 dB
      - Random polarity flip
      - Pitch shift ±2 semitones (via resampling approximation, cheap)
      - Random circular crop (avoids silence at boundaries)
    """

    STEM_NAMES = ['bass', 'drums', 'vocals', 'other', 'guitar', 'piano', 'keys']

    def __init__(
        self,
        root: str,
        cfg: TrackProcessorConfig,
        clip_seconds: float = 4.0,
        augment: bool = True,
        target_stems: Optional[list[str]] = None,
    ):
        self.root      = Path(root)
        self.cfg       = cfg
        self.clip_len  = int(clip_seconds * cfg.sample_rate)
        self.augment   = augment
        self.stems     = target_stems or ['bass', 'drums', 'vocals', 'other']

        # Discover tracks
        self.tracks = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and any((d / f"{s}.wav").exists() for s in self.stems)
        ])
        assert len(self.tracks) > 0, f"No tracks found in {root}"

    def __len__(self) -> int:
        return len(self.tracks) * len(self.stems) * 4  # 4 random crops per stem per track

    def _load_stem(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        """Load and resample stem to target sample rate. Returns (L,) or None.

        Uses soundfile (libsndfile) instead of torchaudio.load to avoid the
        torchcodec/ffmpeg dependency chain that's broken in PyTorch 2.9+ROCm.
        soundfile reads WAV/FLAC/OGG natively, which covers MUSDB18-HQ.
        """
        import soundfile as sf
        import numpy as np

        path = track_dir / f"{stem_name}.wav"
        if not path.exists():
            return None

        # soundfile returns (L, C) for stereo, (L,) for mono
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        # data: (L, C). Convert to mono if needed.
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        wav = torch.from_numpy(data)  # (L,)

        if sr != self.cfg.sample_rate:
            # Use torchaudio.functional.resample on tensor — doesn't hit torchcodec
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.cfg.sample_rate
            ).squeeze(0)
        return wav

    def _load_crepe(self, track_dir: Path, stem_name: str, target_len: int):
        """Load pre-extracted CREPE f0 probabilities for a stem.

        Returns:
          probs:    (T_model, 360) float32 tensor or zeros if not found
          pitched:  scalar 0.0 or 1.0 (gates f0 contribution in FiLM)

        target_len: waveform length used to compute expected T (must match
                    the audio crop length so f0 frames align with STFT frames).
        """
        import numpy as np
        crepe_path   = track_dir / f"{stem_name}.crepe.npy"
        pitched_path = track_dir / f"{stem_name}.pitched.npy"

        T_expected = 1 + target_len // self.cfg.hop_length

        if not crepe_path.exists() or not pitched_path.exists():
            # No CREPE extracted — return zeros (training works, f0 conditioning inert)
            return torch.zeros(T_expected, 360, dtype=torch.float32), 0.0

        # Load as float16 (storage) → cast to float32 (model expects)
        probs = np.load(str(crepe_path)).astype(np.float32)  # (T_full, 360)
        pitched = float(np.load(str(pitched_path)))

        return torch.from_numpy(probs), pitched

    def _random_crop(self, wav: Tensor) -> Tensor:
        L = wav.shape[-1]
        if L <= self.clip_len:
            return F.pad(wav, (0, self.clip_len - L))
        start = random.randint(0, L - self.clip_len)
        return wav[start:start + self.clip_len]

    def _crop_with_crepe(
        self, wav: Tensor, crepe: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Sync-crop waveform and CREPE tensor so frames stay aligned.

        wav:   (L,) waveform at sample_rate
        crepe: (T_full, 360) f0 probs at model frame rate (1 + L_full // hop)

        Returns:
          wav_crop:   (clip_len,)
          crepe_crop: (T_clip, 360) where T_clip = 1 + clip_len // hop
        """
        L = wav.shape[-1]
        T_clip = 1 + self.clip_len // self.cfg.hop_length

        if L <= self.clip_len:
            # Pad waveform
            wav_padded = F.pad(wav, (0, self.clip_len - L))
            # Pad / truncate CREPE to T_clip
            T_have = crepe.shape[0]
            if T_have >= T_clip:
                crepe_out = crepe[:T_clip]
            else:
                pad_amt = T_clip - T_have
                crepe_out = F.pad(crepe, (0, 0, 0, pad_amt))  # pad time dim
            return wav_padded, crepe_out

        # Pick a random sample-aligned start
        start = random.randint(0, L - self.clip_len)
        wav_crop = wav[start:start + self.clip_len]

        # Convert sample start to frame start.
        # CREPE was generated with T = 1 + L_full // hop, so frame i covers
        # samples [i*hop - n_fft/2, i*hop + n_fft/2] (center=True).
        # Closest frame to sample `start` is start // hop.
        frame_start = start // self.cfg.hop_length
        frame_end   = frame_start + T_clip
        T_have = crepe.shape[0]

        if frame_end <= T_have:
            crepe_crop = crepe[frame_start:frame_end]
        else:
            # Edge case: tail of crepe is shorter — pad
            crepe_crop = crepe[frame_start:]
            pad_amt = T_clip - crepe_crop.shape[0]
            crepe_crop = F.pad(crepe_crop, (0, 0, 0, pad_amt))

        return wav_crop, crepe_crop

    def _augment(self, wav: Tensor) -> Tensor:
        # Random gain ±6 dB
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))

        # Random polarity
        if random.random() < 0.5:
            wav = -wav

        return wav.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        track_idx = (idx // (len(self.stems) * 4)) % len(self.tracks)
        stem_idx  = (idx // 4) % len(self.stems)
        track_dir = self.tracks[track_idx]
        target_name = self.stems[stem_idx]

        # Load target stem
        target = self._load_stem(track_dir, target_name)
        if target is None:
            # Fallback: return zeros (will be filtered by collate_fn)
            T_clip = 1 + self.clip_len // self.cfg.hop_length
            return {
                'target':         torch.zeros(self.clip_len),
                'other_stems':    torch.zeros(len(self.stems) - 1, self.clip_len),
                'target_crepe':   torch.zeros(T_clip, 360),
                'other_crepe':    torch.zeros(len(self.stems) - 1, T_clip, 360),
                'target_pitched': 0.0,
                'other_pitched':  torch.zeros(len(self.stems) - 1),
                'stem_name':      target_name,
                'valid':          False,
            }

        # Load CREPE for target (full length, will be sync-cropped with waveform)
        target_crepe, target_pitched = self._load_crepe(
            track_dir, target_name, target.shape[-1]
        )
        target, target_crepe = self._crop_with_crepe(target, target_crepe)

        # Load other stems + their CREPE
        other_wavs       = []
        other_crepes     = []
        other_pitched_l  = []
        for name in self.stems:
            if name == target_name:
                continue
            wav = self._load_stem(track_dir, name)
            if wav is not None:
                cr, pit = self._load_crepe(track_dir, name, wav.shape[-1])
                w_crop, c_crop = self._crop_with_crepe(wav, cr)
                other_wavs.append(w_crop)
                other_crepes.append(c_crop)
                other_pitched_l.append(pit)
            else:
                T_clip = 1 + self.clip_len // self.cfg.hop_length
                other_wavs.append(torch.zeros(self.clip_len))
                other_crepes.append(torch.zeros(T_clip, 360))
                other_pitched_l.append(0.0)

        if self.augment:
            target = self._augment(target)
            other_wavs = [self._augment(w) for w in other_wavs]
            # NOTE: augmentation does NOT touch CREPE — gain/polarity flips
            # don't affect pitch content, so f0 conditioning stays valid.

        return {
            'target':         target,                                    # (L,)
            'other_stems':    torch.stack(other_wavs, dim=0),            # (K, L)
            'target_crepe':   target_crepe,                              # (T, 360)
            'other_crepe':    torch.stack(other_crepes, dim=0),          # (K, T, 360)
            'target_pitched': target_pitched,
            'other_pitched':  torch.tensor(other_pitched_l),             # (K,)
            'stem_name':      target_name,
            'valid':          True,
        }


def collate_valid(batch: list[dict]) -> dict:
    """Filter invalid samples and collate."""
    batch = [b for b in batch if b['valid']]
    if not batch:
        return None
    return {
        'target':         torch.stack([b['target'] for b in batch]),
        'other_stems':    torch.stack([b['other_stems'] for b in batch]),
        'target_crepe':   torch.stack([b['target_crepe'] for b in batch]),
        'other_crepe':    torch.stack([b['other_crepe'] for b in batch]),
        'target_pitched': torch.tensor([b['target_pitched'] for b in batch]),
        'other_pitched':  torch.stack([b['other_pitched'] for b in batch]),
        'stem_names':     [b['stem_name'] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    data_root:  str,
    save_dir:   str,
    cfg:        Optional[TrackProcessorConfig] = None,
    n_steps:    int  = 100_000,
    batch_size: int  = 128,     # H100/80GB: 128-512; iGPU: 2-4
    lr:         float = 3e-4,
    grad_clip:  float = 1.0,
    device:     str  = 'cuda',  # ROCm uses 'cuda' device string
    amp:        bool  = True,   # bf16 AMP — better than fp16 for SSMs
    resume_from: Optional[str] = None,
    val_split:  float = 0.1,
    ckpt_every: int = 1000,
    log_every:  int = 50,
    val_every:  int = 2000,
    num_workers: int = 4,       # 0 for iGPU shared-RAM; 4+ for cloud GPU
    grad_accum:  int = 1,       # gradient accumulation steps (effective batch = batch_size * grad_accum)
    grad_ckpt:   bool = True,   # gradient checkpointing (saves VRAM, costs ~20% compute)
    early_stop_patience: int = 10,  # val checks with no improvement
):
    """
    Staged training:
      Phase 1 (0%-20%):   Pretrain semantic encoder (NT-Xent only)
      Phase 2 (20%-50%):  Single-track processing (no cross-track conditioning)
      Phase 3 (50%+):     Full cross-track conditioning

    Phase boundaries scale with n_steps. At default n_steps=100k:
      Phase 1 ends at 20k, Phase 2 ends at 50k, Phase 3 runs to 100k.

    Memory strategy for shared-RAM iGPU:
      - bf16 AMP (NOT fp16: complex SSM recurrence needs precision)
      - Gradient checkpointing on Mamba layers
      - num_workers=0 default (avoids RAM contention with main process)

    Resume:
      Pass resume_from='checkpoints/checkpoint_005000.pt' to continue training.
      Loads model + optimizer + scheduler + step counter.

    Validation:
      val_split fraction of tracks held out, never seen during training.
      Best checkpoint is selected by val loss, not train loss (much more stable).
    """
    if cfg is None:
        cfg = TrackProcessorConfig()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────
    # Device setup
    # ────────────────────────────────────────────────────────────────
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not available, falling back to CPU")
        device = 'cpu'
        amp = False

    dev = torch.device(device)

    # ────────────────────────────────────────────────────────────────
    # Model
    # ────────────────────────────────────────────────────────────────
    model = PerTrackProcessor(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    for layer in model.mamba_layers:
        layer.gradient_checkpointing = grad_ckpt

    # ────────────────────────────────────────────────────────────────
    # Losses
    # ────────────────────────────────────────────────────────────────
    task_loss   = TrackProcessorLoss(cfg, cfg.sample_rate).to(dev)
    contrastive = NTXentLoss(temperature=0.1).to(dev)

    # ────────────────────────────────────────────────────────────────
    # Optimizer + LR schedule
    # ────────────────────────────────────────────────────────────────
    decay_params  = [p for n, p in model.named_parameters()
                     if 'bias' not in n and 'norm' not in n and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if ('bias' in n or 'norm' in n) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': decay_params,   'weight_decay': 1e-4},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step: int) -> float:
        warmup = 2000
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    autocast_dtype = torch.bfloat16 if amp else torch.float32

    # ────────────────────────────────────────────────────────────────
    # Resume from checkpoint (if requested)
    # ────────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float('inf')
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            print(f"ERROR: --resume path does not exist: {ckpt_path}")
            sys.exit(1)
        print(f"Resuming from {ckpt_path}")
        # weights_only=False needed to load TrackProcessorConfig object
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step']
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
        print(f"  → resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")

    # ────────────────────────────────────────────────────────────────
    # Dataset: train/val split by TRACK (no leakage)
    # ────────────────────────────────────────────────────────────────
    full_dataset = StemDataset(data_root, cfg, clip_seconds=4.0, augment=True)
    val_dataset  = StemDataset(data_root, cfg, clip_seconds=4.0, augment=False)

    n_tracks = len(full_dataset.tracks)
    n_val    = max(1, int(n_tracks * val_split))
    # Deterministic split: last n_val tracks (alphabetical) are val
    random.Random(42).shuffle(full_dataset.tracks)
    val_tracks   = full_dataset.tracks[-n_val:]
    train_tracks = full_dataset.tracks[:-n_val]

    # Apply split
    full_dataset.tracks = train_tracks
    val_dataset.tracks  = val_tracks

    print(f"Tracks: {len(train_tracks)} train, {len(val_tracks)} val")

    dataloader = DataLoader(
        full_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = (device == 'cuda'),
        collate_fn  = collate_valid,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_valid,
        drop_last   = False,
    )

    # ────────────────────────────────────────────────────────────────
    # Phase boundaries (scale with n_steps)
    # ────────────────────────────────────────────────────────────────
    PHASE_1_END = int(n_steps * 0.20)
    PHASE_2_END = int(n_steps * 0.50)

    # ────────────────────────────────────────────────────────────────
    # Signal handler: write a final checkpoint on Ctrl+C
    # ────────────────────────────────────────────────────────────────
    import signal
    _state = {'step': start_step, 'interrupted': False}

    def signal_handler(signum, frame):
        if _state['interrupted']:
            print("\n! Second interrupt — exiting hard.")
            sys.exit(130)
        print(f"\n! Caught signal {signum} — will save checkpoint after current step.")
        _state['interrupted'] = True

    signal.signal(signal.SIGINT, signal_handler)
    # SIGTERM exists on Windows but behaves differently; register defensively.
    # On Windows console processes only SIGINT (Ctrl+C) and SIGBREAK reliably fire.
    if hasattr(signal, 'SIGTERM') and sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    if sys.platform == 'win32' and hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    # ────────────────────────────────────────────────────────────────
    # Atomic checkpoint write
    # ────────────────────────────────────────────────────────────────
    def save_checkpoint(step: int, loss_val: float, is_best: bool = False,
                        name: Optional[str] = None):
        """Write checkpoint atomically: save to .tmp, then rename."""
        ckpt = {
            'step':           step,
            'model':          model.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict(),
            'cfg':            cfg,
            'loss':           loss_val,
            'best_val_loss':  best_val_loss,
        }
        fname = name if name else f"checkpoint_{step:06d}.pt"
        tmp_path  = save_path / f".{fname}.tmp"
        final_path = save_path / fname

        torch.save(ckpt, tmp_path)
        tmp_path.replace(final_path)  # atomic on POSIX

        if is_best:
            best_tmp = save_path / ".best.pt.tmp"
            torch.save(ckpt, best_tmp)
            best_tmp.replace(save_path / "best.pt")

    # ────────────────────────────────────────────────────────────────
    # Validation
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(current_phase: int) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for v_batch in val_loader:
            if v_batch is None:
                continue
            v_target = v_batch['target'].to(dev)
            v_crepe  = v_batch['target_crepe'].to(dev)
            if current_phase == 1:
                # Phase 1 validates with contrastive loss on target
                z1 = _encode_for_contrastive(model, v_target, cfg, dev)
                # use a slightly augmented copy for the second view
                z2 = _encode_for_contrastive(model, _augment_for_contrastive(v_target), cfg, dev)
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                    vl = contrastive(z1, z2)
            else:
                # Phase 2/3 validates with reconstruction
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):
                    if current_phase == 3:
                        # Build other_embs from val stems
                        v_other = v_batch['other_stems'].to(dev)
                        K = v_other.shape[1]
                        embs = []
                        T_frames = 1 + v_target.shape[-1] // cfg.hop_length
                        for k in range(K):
                            v_crepe_k = v_batch['other_crepe'][:, k, :, :].to(dev)
                            out_k = model(v_other[:, k, :], v_crepe_k, other_embs=None)
                            embs.append(out_k['track_emb'])
                        other_embs = torch.stack(embs, dim=1)
                        out = model(v_target, v_crepe, other_embs=other_embs)
                    else:
                        out = model(v_target, v_crepe, other_embs=None)
                    vl_dict = task_loss(out['output_stft'], v_target, model)
                    vl = vl_dict['total']
            if not (torch.isnan(vl) or torch.isinf(vl)):
                total_loss += vl.item()
                n_batches += 1
        model.train()
        return total_loss / max(1, n_batches)

    # ────────────────────────────────────────────────────────────────
    # Training loop
    # ────────────────────────────────────────────────────────────────
    step = start_step
    plateau_count = 0
    last_log_time = time.time()
    last_log_step = step
    accum_step = 0  # gradient accumulation counter

    print(f"\nStarting training: {n_steps} steps, batch={batch_size}, device={device}")
    if grad_accum > 1:
        print(f"Gradient accumulation: {grad_accum} (effective batch = {batch_size * grad_accum})")
    if not grad_ckpt:
        print(f"Gradient checkpointing: OFF (max VRAM, fastest compute)")
    print(f"Phase 1 (contrastive):  step 0       → {PHASE_1_END}")
    print(f"Phase 2 (single-track): step {PHASE_1_END} → {PHASE_2_END}")
    print(f"Phase 3 (cross-track):  step {PHASE_2_END} → {n_steps}")
    if start_step > 0:
        print(f"Resuming at step {start_step}")
    print()

    last_phase = None

    while step < n_steps:
        for batch in dataloader:
            if batch is None:
                continue

            target       = batch['target'].to(dev)
            other_stems  = batch['other_stems'].to(dev)
            target_crepe = batch['target_crepe'].to(dev)
            other_crepe  = batch['other_crepe'].to(dev)
            B = target.shape[0]
            # FIX: use 1 + L/hop to match torch.stft(center=True), NOT (L-n_fft)//hop+1
            T_frames = 1 + target.shape[-1] // cfg.hop_length

            current_phase = (1 if step < PHASE_1_END
                             else 2 if step < PHASE_2_END
                             else 3)

            # Log phase transitions
            if current_phase != last_phase:
                print(f"\n{'─'*60}")
                print(f"  ENTERING PHASE {current_phase} at step {step}")
                print(f"{'─'*60}\n")
                last_phase = current_phase

            # Scale loss by 1/grad_accum for accumulation
            loss_scale = 1.0 / grad_accum

            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=amp):

                if current_phase == 1:
                    aug1 = _augment_for_contrastive(target)
                    aug2 = _augment_for_contrastive(target)
                    z1 = _encode_for_contrastive(model, aug1, cfg, dev)
                    z2 = _encode_for_contrastive(model, aug2, cfg, dev)
                    loss = contrastive(z1, z2)
                    loss_dict = {'total': loss, 'contrastive': loss}

                elif current_phase == 2:
                    out = model(target, target_crepe, other_embs=None)
                    loss_dict = task_loss(out['output_stft'], target, model)

                else:  # Phase 3
                    K = other_stems.shape[1]
                    other_embs_list = []
                    for k in range(K):
                        stem_k  = other_stems[:, k, :]
                        crepe_k = other_crepe[:, k, :, :]
                        out_k = model(stem_k, crepe_k, other_embs=None)
                        other_embs_list.append(out_k['track_emb'])
                    other_embs = torch.stack(other_embs_list, dim=1)
                    out = model(target, target_crepe, other_embs=other_embs)
                    loss_dict = task_loss(out['output_stft'], target, model)

            loss = loss_dict['total']
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[step {step}] NaN/Inf loss — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                continue

            (loss * loss_scale).backward()
            accum_step += 1

            # Step optimizer every grad_accum steps (or at end of accumulation)
            if accum_step >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0

            # ── Logging ──
            if step % log_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                now = time.time()
                steps_per_sec = (step - last_log_step) / max(1e-6, now - last_log_time)
                last_log_time = now
                last_log_step = step

                log_str = (
                    f"[step {step:6d}] phase={current_phase} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e} "
                    f"{steps_per_sec:.2f} step/s"
                )
                if 'mr_stft' in loss_dict:
                    log_str += f" mr_stft={loss_dict['mr_stft'].item():.4f}"
                if 'si_sdr' in loss_dict and loss_dict['si_sdr'].item() != 0:
                    log_str += f" si_sdr={loss_dict['si_sdr'].item():.4f}"
                print(log_str)

            # ── Validation + best checkpoint ──
            if step > 0 and step % val_every == 0:
                print(f"  Running validation at step {step}...")
                val_loss = validate(current_phase)
                print(f"  val_loss = {val_loss:.4f}  (best={best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(step, loss.item(), is_best=True)
                    print(f"  ✓ new best val loss — saved best.pt")
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= early_stop_patience:
                        print(f"  ! Validation plateaued for {early_stop_patience} checks — early stopping")
                        save_checkpoint(step, loss.item(), name="final_earlystop.pt")
                        return model

            # ── Regular checkpoint ──
            if step > 0 and step % ckpt_every == 0:
                save_checkpoint(step, loss.item())
                # Compact: print only every 5 ckpts to avoid spam
                if step % (ckpt_every * 5) == 0:
                    print(f"  ✓ checkpoint at step {step}")

            # ── Signal-triggered exit ──
            if _state['interrupted']:
                print(f"\n! Saving interrupt checkpoint at step {step}...")
                save_checkpoint(step, loss.item(), name=f"interrupt_{step:06d}.pt")
                print(f"  ✓ saved. Resume with: --resume {save_path}/interrupt_{step:06d}.pt")
                return model

            step += 1
            _state['step'] = step
            if step >= n_steps:
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    save_checkpoint(step, loss.item(), name="final.pt")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _augment_for_contrastive(wav: Tensor) -> Tensor:
    """
    Random augmentation for contrastive training (NT-Xent positive pairs).
    Pitch shift approximated via resampling (no torchaudio sox required).
    """
    # Random gain ±4 dB
    gain_db = random.uniform(-4, 4)
    wav = wav * (10 ** (gain_db / 20))

    # Random polarity
    if random.random() < 0.5:
        wav = -wav

    # Random time shift (circular)
    shift = random.randint(0, wav.shape[-1] // 4)
    wav = torch.roll(wav, shift, dims=-1)

    # Additive noise
    noise_level = random.uniform(0, 0.005)
    wav = wav + noise_level * torch.randn_like(wav)

    return wav.clamp(-1.0, 1.0)


def _encode_for_contrastive(
    model: PerTrackProcessor,
    wav:   Tensor,
    cfg:   TrackProcessorConfig,
    dev:   torch.device,
) -> Tensor:
    """
    Run semantic encoder on waveform to get contrastive embedding.
    Returns: (B, d_semantic) L2-normalized
    """
    mag_sq, _ = model._compute_stft(wav)
    mel  = model._compute_mel(mag_sq)
    r8   = model._extract_r8_raw(wav, mag_sq)
    return model.semantic_enc(mel, r8, return_sequence=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Mamba-3 Per-Track Adaptive Filter Bank training"
    )
    # Data + I/O
    parser.add_argument('--data',       default=r'/workspace/datasets/musdb18hq/train',  help='Path to MUSDB18-HQ train/ root')
    parser.add_argument('--save',       default='./checkpoints', help='Where to write checkpoints')
    parser.add_argument('--resume',     default=None,
                        help='Resume from checkpoint path (e.g. checkpoints/checkpoint_005000.pt)')

    # Training schedule
    parser.add_argument('--steps',      type=int,   default=100_000,
                        help='Total training steps (phase boundaries scale: 20%% / 50%% / 100%%)')
    parser.add_argument('--batch',      type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--no-amp',     action='store_true')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch × grad-accum)')
    parser.add_argument('--no-grad-ckpt', action='store_true',
                        help='Disable gradient checkpointing (uses more VRAM, ~20%% faster)')

    # Validation + checkpointing cadence
    parser.add_argument('--val-split',   type=float, default=0.1,
                        help='Fraction of tracks held out for validation (default 0.1)')
    parser.add_argument('--val-every',   type=int,   default=2000,
                        help='Run validation every N steps (default 2000)')
    parser.add_argument('--ckpt-every',  type=int,   default=1000,
                        help='Save checkpoint every N steps (default 1000)')
    parser.add_argument('--log-every',   type=int,   default=50,
                        help='Print log line every N steps (default 50)')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Stop if val loss does not improve for N val checks')

    # DataLoader
    parser.add_argument('--num-workers', type=int,   default=4,
                        help='DataLoader workers. 0 for iGPU shared RAM; 4+ for cloud GPU.')

    # Model architecture
    parser.add_argument('--d-model',    type=int,   default=256)
    parser.add_argument('--d-state',    type=int,   default=32)
    parser.add_argument('--n-layers',   type=int,   default=6)

    args = parser.parse_args()

    cfg = TrackProcessorConfig(
        d_model  = args.d_model,
        d_state  = args.d_state,
        n_layers = args.n_layers,
    )

    train(
        data_root           = args.data,
        save_dir            = args.save,
        cfg                 = cfg,
        n_steps             = args.steps,
        batch_size          = args.batch,
        lr                  = args.lr,
        device              = args.device,
        amp                 = not args.no_amp,
        resume_from         = args.resume,
        val_split           = args.val_split,
        ckpt_every          = args.ckpt_every,
        log_every           = args.log_every,
        val_every           = args.val_every,
        num_workers         = args.num_workers,
        grad_accum          = args.grad_accum,
        grad_ckpt           = not args.no_grad_ckpt,
        early_stop_patience = args.early_stop_patience,
    )

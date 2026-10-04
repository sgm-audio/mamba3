"""
Mamba-3 MIMO R=8 Per-Track Adaptive Filter Bank
================================================
Option B: Context-aware per-track transform.
"A per-track processor that hears your whole session."

Architecture:
  - R=8 structured spectral feature extractor (learnable band boundaries)
  - Mamba-3 MIMO recurrence (complex poles, trapezoidal discretization)
  - FiLM conditioning from cross-track aggregator (CREPE f0 + semantic)
  - Waveform decoder (overlap-add synthesis)

Constraints:
  - Pure PyTorch: no causal_conv1d, no Triton, no CUDA extensions
  - AMD ROCm gfx1103 safe (Bazzite distrobox)
  - 8 GB shared VRAM budget
  - Apache 2.0 clean (Mamba-3, auraloss, torchcrepe all permissive)

Target: MUSDB18-HQ / MoisesDB stems @ 44100 Hz, non-RT training.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrackProcessorConfig:
    # Audio
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512          # ~11.6 ms @ 44.1 kHz
    n_mels: int = 128

    # R=8 band structure (Hz boundaries, learnable centers initialized here)
    band_boundaries_hz: tuple = (
        (20,   80),    # sub_bass
        (80,   250),   # bass
        (250,  800),   # low_mid
        (800,  2500),  # mid
        (2500, 8000),  # high_mid
        (8000, 20000), # air
    )
    # slots 7,8 = transient (computed) + centroid (computed)
    R: int = 8

    # Mamba-3 MIMO core
    d_model: int = 256             # embedding dim
    d_state: int = 32              # N: state dim (Mamba-3 needs N/2 vs M2 for same quality)
    n_heads: int = 16              # H: filter bank width
    n_layers: int = 6              # depth
    chunk_size: int = 128          # SSD chunk (must divide T on fast path)
    dt_min: float = 0.001
    dt_max: float = 0.1

    # Conditioning
    d_semantic: int = 128          # semantic encoder output dim
    d_f0: int = 32                 # CREPE 360-D → 32-D projection
    d_cross: int = 64              # cross-track attention output
    d_cond: int = 128              # final FiLM conditioning dim (= d_semantic + d_f0 + d_cross + 4)

    # Training
    max_tracks: int = 8            # max stems in a session
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# R=8 Structured Feature Extractor
# ---------------------------------------------------------------------------

class R8FeatureExtractor(nn.Module):
    """
    Extracts R=8 structured spectral features per frame:
      [sub_bass, bass, low_mid, mid, high_mid, air, transient, centroid]

    Band boundaries are learnable (initialized from config).
    Learnable = soft triangular filterbank with trainable center freqs.
    Avoids hard-coded FFT bin slicing which loses gradient flow.
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.cfg = cfg
        sr = cfg.sample_rate
        n_fft = cfg.n_fft

        # Frequency axis (positive freqs only)
        self.register_buffer(
            "freqs",
            torch.linspace(0, sr / 2, n_fft // 2 + 1)
        )

        # Learnable band centers (Hz), initialized at geometric mean of boundaries
        centers_hz = [
            math.sqrt(lo * hi)
            for (lo, hi) in cfg.band_boundaries_hz
        ]
        # Store in log-Hz space for unconstrained optimization
        self.log_centers = nn.Parameter(
            torch.log(torch.tensor(centers_hz, dtype=torch.float32))
        )
        # Learnable log-bandwidths (initialized to ~1/3 octave)
        self.log_bandwidths = nn.Parameter(
            torch.full((6,), math.log(0.5))
        )

        # Transient detector: tiny 1-D conv on waveform envelope
        self.transient_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            padding=2,
            bias=False
        )

        # Centroid normalization (log scale, max = log(sr/2))
        self.log_max_freq = math.log(sr / 2)

        # Project R=8 features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.R, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

    def _soft_band_energy(self, mag_sq: Tensor) -> Tensor:
        """
        mag_sq: (B, T, F) — squared magnitude spectrogram
        returns: (B, T, 6) — soft-band energies
        """
        # freqs: (F,) → (1, 1, F, 1) so it broadcasts cleanly against bands at dim 3
        freqs = self.freqs.view(1, 1, -1, 1)           # (1, 1, F, 1)
        centers = torch.exp(self.log_centers).view(1, 1, 1, -1)  # (1, 1, 1, 6)
        bws     = torch.exp(self.log_bandwidths).view(1, 1, 1, -1)  # (1, 1, 1, 6)

        # Soft triangular filterbank: weight = max(0, 1 - |log(f/c)| / bw)
        log_ratio = torch.log(freqs.clamp(min=1.0) / centers)  # (1, 1, F, 6)
        weights = F.relu(1.0 - log_ratio.abs() / bws)           # (1, 1, F, 6)
        weights = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-8))

        energies = (mag_sq.unsqueeze(-1) * weights).sum(dim=2)  # (B, T, 6)
        return torch.log1p(energies)  # log-compress

    def _transient_density(self, waveform: Tensor, T: int) -> Tensor:
        """
        waveform: (B, L)
        T: target number of frames (must match STFT output T)
        returns: (B, T, 1) — per-frame transient density

        Uses adaptive pooling to guarantee output frame count matches STFT,
        regardless of center-padding or hop arithmetic differences.
        """
        envelope = waveform.abs().unsqueeze(1)               # (B, 1, L)
        diff = self.transient_conv(envelope)                  # (B, 1, L)
        diff = F.relu(diff)
        # Adaptive avg pool to T frames — handles any L/T ratio cleanly
        pooled = F.adaptive_avg_pool1d(diff, T)               # (B, 1, T)
        return pooled.transpose(1, 2)                         # (B, T, 1)

    def _spectral_centroid(self, mag_sq: Tensor) -> Tensor:
        """
        mag_sq: (B, T, F)
        returns: (B, T, 1) — normalized log spectral centroid
        """
        freqs = self.freqs.clamp(min=1.0)
        weights = mag_sq / (mag_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        centroid_hz = (weights * freqs).sum(dim=-1, keepdim=True)  # (B, T, 1)
        return torch.log(centroid_hz.clamp(min=1.0)) / self.log_max_freq

    def forward(self, waveform: Tensor, stft_mag_sq: Tensor) -> Tensor:
        """
        waveform:    (B, L)
        stft_mag_sq: (B, T, F) — precomputed |STFT|^2

        returns: (B, T, d_model)
        """
        band_energy  = self._soft_band_energy(stft_mag_sq)    # (B, T, 6)
        T_frames = stft_mag_sq.shape[1]
        transient    = self._transient_density(waveform, T_frames)  # (B, T, 1)
        centroid     = self._spectral_centroid(stft_mag_sq)    # (B, T, 1)

        r8 = torch.cat([band_energy, transient, centroid], dim=-1)  # (B, T, 8)
        return self.input_proj(r8)                                    # (B, T, d_model)


# ---------------------------------------------------------------------------
# Mamba-3 Core: Pure PyTorch MIMO with Complex Poles
# ---------------------------------------------------------------------------

class Mamba3MIMOLayer(nn.Module):
    """
    Pure-PyTorch Mamba-3 MIMO layer.

    Three innovations from arXiv 2603.15569:
      1. Exponential-trapezoidal discretization (2nd-order, bilinear-like)
      2. Complex-valued state (poles on complex plane = resonant bandpass heads)
      3. MIMO: B,C matrices operate on R-dim input vector per head

    No causal_conv1d dependency (Mamba-3 drops the short conv entirely).
    No Triton kernels. Runs on any PyTorch backend incl. ROCm gfx1103.

    Filter-bank interpretation:
      - Each of H heads = damped oscillator (bandpass filter)
      - Center freq controlled by Im(A_t), bandwidth by Re(A_t)
      - Both modulated per-timestep by input x_t via Δ_t
      - MIMO B/C: R input channels couple to N-dim complex state

    Memory: O(B * H * N) constant (SSM, not attention).
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.H = cfg.n_heads
        self.N = cfg.d_state        # complex state dim per head
        self.d = cfg.d_model
        self.R = cfg.R
        self.chunk_size = cfg.chunk_size

        head_dim = cfg.d_model // cfg.n_heads
        assert head_dim * cfg.n_heads == cfg.d_model, \
            "d_model must be divisible by n_heads"
        self.head_dim = head_dim

        # --- Input projection (replaces short conv of Mamba-1/2) ---
        # Projects d_model → [x_inner, dt, BCgate] for all heads
        # No conv: Mamba-3 uses bias on B/C instead
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model * 2 + cfg.n_heads, bias=False)

        # --- Complex SSM parameters ---
        # A: real part (decay), imaginary part (frequency) per head
        # Initialize: small negative real (stable), uniform imaginary (spread bands)
        log_a_real = torch.zeros(cfg.n_heads)
        a_imag = torch.linspace(0, math.pi, cfg.n_heads + 1)[:-1]  # spread 0..π
        self.log_a_real = nn.Parameter(log_a_real)   # unconstrained; exp → negative
        self.a_imag     = nn.Parameter(a_imag)        # imaginary part (frequency)

        # --- MIMO B and C projections ---
        # B: R → N (input coupling, complex)
        # C: N → head_dim (readout, complex)
        self.B_proj = nn.Linear(cfg.R, cfg.d_state * 2, bias=True)   # *2 for real+imag
        self.C_proj = nn.Linear(cfg.d_state * 2, self.head_dim, bias=False)

        # --- Δ (step size) projection: d_model → n_heads ---
        self.dt_proj  = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.dt_bias  = nn.Parameter(torch.full((cfg.n_heads,), -4.0))  # softplus(-4)≈0.018
        self.log_dt_min = math.log(cfg.dt_min)
        self.log_dt_max = math.log(cfg.dt_max)

        # --- Output ---
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm     = nn.LayerNorm(cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

        # --- Activation-memory control (honored by forward; set by train.py) ---
        # When True during training, the layer body is wrapped in
        # torch.utils.checkpoint (non-reentrant) so activations are recomputed
        # in backward instead of stored. ~20% slower, much lower peak VRAM.
        self.gradient_checkpointing = False

        # --- FiLM conditioning projections (eagerly initialized) ---
        # Initialized so γ ≈ 0, β ≈ 0 → layer starts as identity in cond pathway.
        # This is the standard FiLM init trick: model behaves as plain Mamba3
        # at step 0, conditioning influence grows during training.
        self.film_proj    = nn.Linear(cfg.d_cond, cfg.d_model * 2, bias=True)
        self.dt_film_proj = nn.Linear(cfg.d_cond, cfg.n_heads * 2, bias=True)
        with torch.no_grad():
            self.film_proj.weight.zero_()
            self.film_proj.bias.zero_()
            self.dt_film_proj.weight.zero_()
            self.dt_film_proj.bias.zero_()

    def _discretize_trapezoidal(
        self, a_real: Tensor, a_imag: Tensor, dt: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Exponential-trapezoidal discretization (Mamba-3 innovation #1).
        Equivalent to bilinear (Tustin) transform for continuous→discrete.

        Continuous: dh/dt = (α + jω) h + B x
        Trapezoidal: h_t = A_d h_{t-1} + B_d x_t
          A_d = (1 + Δ(α+jω)/2) / (1 - Δ(α+jω)/2)
          B_d = Δ / (1 - Δ(α+jω)/2)

        This maps left-half s-plane exactly to unit disk (stable by construction
        for α < 0), preserving frequency response shape. Compare ZOH (Mamba-1/2)
        which has frequency warping at high freqs — critical for 44.1 kHz audio.

        a_real: (H,) — negative decay rates
        a_imag: (H,) — oscillation frequencies
        dt:     (B, T, H) — per-step discretization

        returns: A_d (B, T, H) complex, B_d_scale (B, T, H) complex
        """
        # Continuous eigenvalue: α + jω, shape (H,)
        a_cont = torch.complex(
            -torch.exp(a_real),  # force negative real part (stable)
            a_imag
        )  # (H,)

        # Broadcast dt: (B, T, H)
        half_dt_a = dt.unsqueeze(-1) * a_cont / 2  # will broadcast: (B,T,H)
        # NOTE: dt is real, a_cont is complex → result is complex
        half_dt_a = torch.complex(
            dt * (-torch.exp(a_real)) / 2,
            dt * a_imag / 2
        )  # (B, T, H) complex

        numerator   = 1 + half_dt_a
        denominator = 1 - half_dt_a

        A_d = numerator / denominator          # (B, T, H) complex, |A_d| ≤ 1
        B_d_scale = dt / denominator           # (B, T, H) complex

        return A_d, B_d_scale

    def _ssm_scan_sequential(
        self,
        x_r8: Tensor,      # (B, T, R) — R=8 features (MIMO input)
        A_d:  Tensor,      # (B, T, H) complex
        B_d:  Tensor,      # (B, T, H) complex — scaling factor
        dt:   Tensor,      # (B, T, H) — for B_d application
    ) -> Tensor:
        """
        Sequential SSM scan. O(T) sequential, O(H*N) memory.
        Slow but correct and ROCm-safe. Replace with parallel scan
        when/if ROCm kernel is available.

        Returns: (B, T, H, head_dim) — output sequence
        """
        B_batch, T, R = x_r8.shape
        H, N = self.H, self.N

        # MIMO B: project R → N complex per head (shared across heads for now)
        bc = self.B_proj(x_r8)                         # (B, T, N*2)
        B_real, B_imag = bc.chunk(2, dim=-1)           # (B, T, N) each
        B_vec = torch.complex(B_real, B_imag)          # (B, T, N) complex

        # Initialize hidden state (complex)
        h = torch.zeros(B_batch, H, N, dtype=torch.complex64, device=x_r8.device)

        outputs = []
        for t in range(T):
            # A_d_t: (B, H) complex, B_d_t: (B, H) complex
            A_d_t = A_d[:, t, :]   # (B, H)
            B_d_t = B_d[:, t, :]   # (B, H)

            # State update: h = A_d * h + B_d * (B_vec * x)
            # B_d_t: (B, H) → (B, H, 1), B_vec_t: (B, N) → (B, 1, N)
            B_input = B_d_t.unsqueeze(-1) * B_vec[:, t, :].unsqueeze(1)  # (B, H, N)
            h = A_d_t.unsqueeze(-1) * h + B_input      # (B, H, N) complex

            # Readout: C projects complex state → real head_dim
            h_cat = torch.cat([h.real, h.imag], dim=-1)   # (B, H, N*2)
            y_t = self.C_proj(h_cat)                        # (B, H, head_dim)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)               # (B, T, H, head_dim)
        return out

    def forward(
        self,
        x: Tensor,           # (B, T, d_model) — track embedding
        x_r8: Tensor,        # (B, T, R=8) — raw structured features (MIMO input)
        cond: Optional[Tensor] = None,  # (B, T, d_cond) — FiLM conditioning
    ) -> Tensor:
        """
        Returns: (B, T, d_model)

        If ``self.gradient_checkpointing`` is True and gradients are enabled,
        the body runs under ``torch.utils.checkpoint.checkpoint`` so training
        recomputes activations in backward instead of storing them (set
        ``layer.gradient_checkpointing = True`` from the training loop; the
        ``--no-grad-ckpt`` CLI flag disables it).
        """
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            # cond may be None; non-tensor args pass through non-reentrant
            # checkpointing unchanged.
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, x_r8, cond, use_reentrant=False
            )
        return self._forward_impl(x, x_r8, cond)

    def _forward_impl(
        self,
        x: Tensor,           # (B, T, d_model) — track embedding
        x_r8: Tensor,        # (B, T, R=8) — raw structured features (MIMO input)
        cond: Optional[Tensor] = None,  # (B, T, d_cond) — FiLM conditioning
    ) -> Tensor:
        """
        Returns: (B, T, d_model)
        """
        B, T, D = x.shape
        residual = x

        # --- FiLM on input (from cross-track conditioning) ---
        if cond is not None:
            gamma_beta = self.film_proj(cond)            # (B, T, D*2)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        # --- Δ projection (input-dependent step size) ---
        dt_raw = self.dt_proj(x)                        # (B, T, H)

        # FiLM on Δ from conditioning (modulates pole location per timestep)
        if cond is not None:
            dt_gamma_beta = self.dt_film_proj(cond)     # (B, T, H*2)
            dt_gamma, dt_beta = dt_gamma_beta.chunk(2, dim=-1)
            dt_raw = dt_raw * (1 + dt_gamma) + dt_beta

        # Clamp Δ to [dt_min, dt_max] via softplus + clamp
        dt = F.softplus(dt_raw + self.dt_bias)          # (B, T, H)
        dt = dt.clamp(
            min=math.exp(self.log_dt_min),
            max=math.exp(self.log_dt_max)
        )

        # --- Trapezoidal discretization + MIMO scan ---
        # Force fp32: torch.complex() does not support bf16/fp16. Even under
        # bf16 autocast, the SSM recurrence must compute in fp32 for stability
        # (complex pole arithmetic is precision-sensitive).
        with torch.autocast(device_type=x.device.type, enabled=False):
            dt_fp32 = dt.float()
            x_r8_fp32 = x_r8.float()
            A_d, B_d_scale = self._discretize_trapezoidal(
                self.log_a_real.float(), self.a_imag.float(), dt_fp32
            )
            out = self._ssm_scan_sequential(x_r8_fp32, A_d, B_d_scale, dt_fp32)

        # Restore outer dtype for the rest of the layer
        out = out.to(x.dtype)
        out = out.reshape(B, T, D)                      # (B, T, d_model)

        # --- Output projection + residual ---
        out = self.out_proj(out)
        out = self.drop(out)
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Cross-Track Attention Aggregator
# ---------------------------------------------------------------------------

class CrossTrackAggregator(nn.Module):
    """
    Aggregates semantic embeddings from K other tracks into a
    per-frame context vector for the current track.

    Design: causal multi-head attention over track dimension,
    not time dimension. Each timestep independently attends over
    the K other tracks' features at that timestep.

    Input:  target_emb (B, T, d_semantic), other_embs (B, K, T, d_semantic)
    Output: context (B, T, d_cross)
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_semantic,
            num_heads=4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(cfg.d_semantic, cfg.d_cross)
        self.norm = nn.LayerNorm(cfg.d_cross)

    def forward(
        self,
        target_emb: Tensor,   # (B, T, d_semantic)
        other_embs: Tensor,   # (B, K, T, d_semantic)
    ) -> Tensor:              # (B, T, d_cross)
        B, T, D = target_emb.shape
        K = other_embs.shape[1]

        # Reshape: treat T as batch, attend over K tracks
        # query: (B*T, 1, D), kv: (B*T, K, D)
        q  = target_emb.reshape(B * T, 1, D)
        kv = other_embs.permute(0, 2, 1, 3).reshape(B * T, K, D)

        ctx, _ = self.attn(q, kv, kv)                  # (B*T, 1, D)
        ctx = ctx.squeeze(1).reshape(B, T, D)          # (B, T, D)
        return self.norm(self.proj(ctx))               # (B, T, d_cross)


# ---------------------------------------------------------------------------
# Semantic Encoder (lightweight self-supervised)
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """
    Lightweight contrastive semantic encoder.
    COLA-style: trained with NT-Xent on augmented pairs.

    Architecture: conv stem → 2 Mamba-3 layers → mean pool → projection.
    Input: log-mel spectrogram (B, T, n_mels)
    Output: (B, d_semantic) — per-clip embedding
             OR (B, T, d_semantic) — per-frame if return_sequence=True
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.d_semantic, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.d_semantic, cfg.d_semantic, kernel_size=3, padding=1),
            nn.LayerNorm(cfg.d_semantic),  # applied after transpose
        )
        # Project d_semantic → d_model for Mamba layer compatibility.
        # Mamba3MIMOLayer is configured at d_model=256; encoder works in d_semantic=128.
        # Without these, dt_proj weight (d_model x n_heads) cannot multiply (B, T, d_semantic).
        self.proj_up = nn.Linear(cfg.d_semantic, cfg.d_model)
        self.proj_down = nn.Linear(cfg.d_model, cfg.d_semantic)

        # Two lightweight Mamba-3 layers for temporal context
        self.mamba_layers = nn.ModuleList([
            Mamba3MIMOLayer(cfg) for _ in range(2)
        ])
        # Projection head (for contrastive training)
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.d_semantic, cfg.d_semantic),
            nn.GELU(),
            nn.Linear(cfg.d_semantic, cfg.d_semantic),
        )

    def forward(
        self,
        mel: Tensor,              # (B, T, n_mels)
        r8: Tensor,               # (B, T, R=8) — needed for MIMO layers
        return_sequence: bool = False,
    ) -> Tensor:
        # Conv stem expects (B, C, T)
        x = mel.transpose(1, 2)               # (B, n_mels, T)
        x = self.conv_stem[0](x)              # Conv1d: (B, d_semantic, T)
        x = self.conv_stem[1](x)              # GELU
        x = self.conv_stem[2](x)              # Conv1d
        x = x.transpose(1, 2)                 # (B, T, d_semantic)
        x = self.conv_stem[3](x)              # LayerNorm

        # Project to d_model for Mamba layers
        x = self.proj_up(x)                   # (B, T, d_model)

        for layer in self.mamba_layers:
            x = layer(x, r8)                  # no cross-track cond in encoder

        # Project back to d_semantic
        x = self.proj_down(x)                 # (B, T, d_semantic)

        if return_sequence:
            return x                          # (B, T, d_semantic)

        # Mean pool → projection
        pooled = x.mean(dim=1)                # (B, d_semantic)
        return F.normalize(self.proj_head(pooled), dim=-1)


# ---------------------------------------------------------------------------
# F0 Conditioning Module
# ---------------------------------------------------------------------------

class F0Conditioner(nn.Module):
    """
    Projects CREPE 360-D f0 probability vector → d_f0.
    CREPE covers C1-B7 (32.7-1975.5 Hz) at 20-cent resolution, 100 Hz frame rate.

    Input:  (B, T_crepe, 360) — must be resampled to model frame rate externally
    Output: (B, T, d_f0)
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(360, 128),
            nn.GELU(),
            nn.Linear(128, cfg.d_f0),
            nn.LayerNorm(cfg.d_f0),
        )

    def forward(self, crepe_probs: Tensor) -> Tensor:
        return self.proj(crepe_probs)


# ---------------------------------------------------------------------------
# Full Per-Track Processor
# ---------------------------------------------------------------------------

class PerTrackProcessor(nn.Module):
    """
    Full Option B model.

    Per-track forward pass:
      1. Extract R=8 features + mel spectrogram
      2. Encode semantic embedding (per-frame, return_sequence=True)
      3. Project f0 conditioning
      4. Aggregate cross-track context (if other_embs provided)
      5. Build FiLM conditioning vector
      6. N × Mamba-3 MIMO layers with FiLM
      7. Decode to waveform (overlap-add)

    Cross-track conditioning is optional at inference (graceful degradation
    to single-track enhancement = Option A for free).
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.cfg = cfg

        # Feature extraction
        self.r8_extractor   = R8FeatureExtractor(cfg)
        self.semantic_enc   = SemanticEncoder(cfg)
        self.f0_conditioner = F0Conditioner(cfg)
        self.cross_track    = CrossTrackAggregator(cfg)

        # Conditioning MLP: [semantic(128) + f0(32) + cross(64) + loudness(4)] → d_cond
        cond_in = cfg.d_semantic + cfg.d_f0 + cfg.d_cross + 4
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, cfg.d_cond),
            nn.GELU(),
            nn.Linear(cfg.d_cond, cfg.d_cond),
            nn.LayerNorm(cfg.d_cond),
        )

        # Core Mamba-3 MIMO stack
        self.mamba_layers = nn.ModuleList([
            Mamba3MIMOLayer(cfg) for _ in range(cfg.n_layers)
        ])

        # Decoder: project back to waveform via overlap-add synthesis
        # Output: frame-level waveform chunks → reassembled
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, 2 * (cfg.n_fft // 2 + 1)),   # 2*F_bins for real+imag
        )

        # Mel projection: STFT bins → n_mels. Initialized with abs() so weights
        # are non-negative (mel filterbanks are positive sums). Prevents NaN
        # from log1p(negative) when input is non-negative magnitude squared.
        F_bins = cfg.n_fft // 2 + 1
        self.mel_proj = nn.Linear(F_bins, cfg.n_mels, bias=False)
        with torch.no_grad():
            self.mel_proj.weight.abs_()  # force non-negative init

    def _compute_stft(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """
        waveform: (B, L)
        returns:
          mag_sq: (B, T, F) — |STFT|^2
          stft:   (B, T, F) complex — for reconstruction
        """
        cfg = self.cfg
        window = torch.hann_window(cfg.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=window,
            return_complex=True,
            normalized=False,
        )  # (B, F, T) complex
        stft = stft.transpose(1, 2)            # (B, T, F)
        return stft.abs().pow(2), stft

    def _compute_mel(self, mag_sq: Tensor) -> Tensor:
        """Approximate mel from magnitude squared (no torchaudio dep).
        mel_proj weights are non-negative (initialized via abs), so output is
        non-negative for non-negative input. Clamp added as safety net.
        """
        projected = self.mel_proj(mag_sq).clamp(min=0.0)
        return torch.log1p(projected)

    def _loudness_features(self, waveform: Tensor, T: int) -> Tensor:
        """4-dim loudness descriptor per frame: [rms, peak, crest_factor, lufs_approx]

        Pads waveform if needed to make L divisible by T, then reshapes.
        Handles arbitrary L/T ratio (STFT center-padding makes T = L/hop + 1).
        """
        B = waveform.shape[0]
        L = waveform.shape[-1]
        # Compute effective frame size; pad waveform to T * frame_size if needed
        frame_size = (L + T - 1) // T  # ceiling division
        pad_amount = T * frame_size - L
        if pad_amount > 0:
            waveform = F.pad(waveform, (0, pad_amount))
        frames = waveform.reshape(B, T, frame_size)

        rms    = frames.pow(2).mean(-1, keepdim=True).sqrt()
        peak   = frames.abs().max(-1, keepdim=True).values
        crest  = (peak / (rms + 1e-8)).clamp(min=1e-8).log()
        lufs   = 10 * torch.log10(rms.pow(2) + 1e-8)

        return torch.cat([rms, peak, crest, lufs], dim=-1)  # (B, T, 4)

    def forward(
        self,
        waveform: Tensor,               # (B, L) — target track
        crepe_probs: Tensor,            # (B, T, 360) — f0 (pre-resampled)
        other_embs: Optional[Tensor] = None,   # (B, K, T, d_semantic)
        target_pitched: Optional[Tensor] = None,  # (B,) float 0/1 — gates f0 in FiLM
    ) -> dict:
        """
        Returns dict with:
          'output_stft': (B, T, F) complex — predicted STFT
          'track_emb':   (B, T, d_semantic) — for use as other_embs in parallel tracks
          'r8':          (B, T, R=8) — structured features (debug/viz)
        """
        cfg = self.cfg

        # --- Feature extraction ---
        mag_sq, stft_in = self._compute_stft(waveform)
        T = mag_sq.shape[1]

        r8  = self.r8_extractor(waveform, mag_sq)              # (B, T, d_model)
        # raw r8 before proj (for MIMO input)
        r8_raw = self._extract_r8_raw(waveform, mag_sq)        # (B, T, 8)
        mel = self._compute_mel(mag_sq)                         # (B, T, n_mels)

        # --- Semantic encoding (per-frame) ---
        sem_emb = self.semantic_enc(mel, r8_raw, return_sequence=True)  # (B, T, d_semantic)

        # --- F0 conditioning ---
        f0_emb = self.f0_conditioner(crepe_probs)               # (B, T, d_f0)
        # Gate f0 contribution for unpitched stems (drums/other): the CREPE
        # vector is all zeros there, but F0Conditioner's biases/LayerNorm would
        # still inject a non-zero f0 embedding. target_pitched (0.0/1.0, from
        # <stem>.pitched.npy) zeroes that pathway exactly.
        if target_pitched is not None:
            gate = target_pitched.to(f0_emb.dtype).reshape(-1, 1, 1)  # (B, 1, 1)
            f0_emb = f0_emb * gate

        # --- Cross-track aggregation ---
        if other_embs is not None:
            cross_ctx = self.cross_track(sem_emb, other_embs)  # (B, T, d_cross)
        else:
            cross_ctx = torch.zeros(
                waveform.shape[0], T, cfg.d_cross, device=waveform.device
            )

        # --- Build conditioning vector ---
        loudness = self._loudness_features(waveform, T)         # (B, T, 4)
        cond_in  = torch.cat([sem_emb, f0_emb, cross_ctx, loudness], dim=-1)
        cond     = self.cond_mlp(cond_in)                       # (B, T, d_cond)

        # --- Mamba-3 MIMO stack ---
        x = r8  # (B, T, d_model) — start from R=8 projected embedding
        for layer in self.mamba_layers:
            x = layer(x, r8_raw, cond=cond)

        # --- Decode to STFT ---
        stft_pred_real = self.decoder(x)                        # (B, T, 2*F_bins)
        # Split into real/imag of output STFT. Force fp32: torch.complex()
        # requires float32 or float64 inputs, fails with bf16/fp16.
        F_bins = cfg.n_fft // 2 + 1
        stft_real = stft_pred_real[..., :F_bins].float()
        stft_imag = stft_pred_real[..., F_bins:2*F_bins].float()
        stft_out  = torch.complex(stft_real, stft_imag)         # (B, T, F) complex64

        return {
            'output_stft': stft_out,
            'track_emb':   sem_emb,
            'r8':          r8_raw,
        }

    def _extract_r8_raw(self, waveform: Tensor, mag_sq: Tensor) -> Tensor:
        """Extract raw R=8 features (before d_model projection) for MIMO input."""
        ext = self.r8_extractor
        band_energy = ext._soft_band_energy(mag_sq)             # (B, T, 6)
        T_frames = mag_sq.shape[1]
        transient   = ext._transient_density(waveform, T_frames)  # (B, T, 1)
        centroid    = ext._spectral_centroid(mag_sq)            # (B, T, 1)
        return torch.cat([band_energy, transient, centroid], dim=-1)  # (B, T, 8)

    def waveform_from_stft(self, stft: Tensor) -> Tensor:
        """
        Reconstruct waveform from predicted complex STFT via Griffin-Lim or
        direct ISTFT (if phase is predicted, use ISTFT directly).
        stft: (B, T, F) complex
        """
        cfg = self.cfg
        window = torch.hann_window(cfg.n_fft, device=stft.device)
        # ISTFT expects (B, F, T)
        return torch.istft(
            stft.transpose(1, 2),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=window,
        )

"""
Mamba-3 MIMO R=8 Per-Track Adaptive Filter Bank
================================================
Option B: Context-aware per-track transform.
"A per-track processor that hears your whole session."

Architecture:
  - R=8 structured spectral feature extractor (learnable band boundaries)
  - Mamba-3 MIMO recurrence (Way 2B: data-dependent rotary angle state +
    exponential-trapezoidal K/V outer-product update; Way 2C chunked scan)
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

    # Mamba-3 MIMO core (Way 2B: official arXiv:2603.15569 MIMO formulation —
    # rotary angle state + exponential-trapezoidal K/V outer-product update;
    # Way 2C: chunked parallel scan). State per head is a (P×D) matrix.
    d_model: int = 256             # embedding dim
    d_state: int = 32              # legacy field (kept for checkpoint configs)
    n_heads: int = 16              # H: filter bank width
    n_layers: int = 6              # depth
    chunk_size: int = 128          # SSD chunk (must divide T on fast path)
    dt_min: float = 0.001
    dt_max: float = 0.1
    mimo_rank: int = 2             # R_mimo: MIMO outer-product rank (Q/K rank axis)
    rotary_dim_divisor: int = 4    # rotated pairs = (d_model/n_heads)/divisor

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
    Pure-PyTorch Mamba-3 MIMO layer — Way 2B (arXiv:2603.15569 formulation,
    matching mamba3/mamba/tests/ops/tilelang/test_mamba3_mimo.py::
    mamba3_MIMO_step_ref) + Way 2C chunked parallel scan.

    The official Mamba-3 MIMO recurrence (per head h, step t):

      1. Data-dependent rotary angle state — the paper's complex-valued state
         transitions in their data-dependent-RoPE equivalent form:
              θ_t = θ_{t-1} + π · tanh(Angles_t) · Δ_t
         Q_t and K_t are rotated by θ_t (rotary_dim_divisor of the head_dim).

      2. Exponential-trapezoidal discretization: the state update mixes the
         PREVIOUS key/value outer product (trapezoidal lag term β) with the
         current one (γ), under α_t = exp(Δ_t · A_h) decay (A_h < 0 learnable):
              S_t = α_t·S_{t-1} + β_t·(K_{t-1} ⊗ V_{t-1}) + γ_t·(K_t ⊗ V_t)
              β_t = (1 − σ(τ_t))·Δ_t·α_t,    γ_t = σ(τ_t)·Δ_t
         No causal_conv1d anywhere — Mamba-3 drops the short conv.

      3. Rank-R_mimo MIMO: Q/K carry an mimo_rank axis; V and Z are per-rank
         gain copies (MIMO_V / MIMO_Z of shape (H, R, D)); the readout sums
         ranks with MIMO_O and applies the D skip + Swish gate z·σ(z):
              out_t = MIMO_O · Σ_r [ S_t·Q_t + D·V_t ] ⊙ swish(Z_t)

    x_r8 (the R=8 structured spectral features) enters as additive input
    coupling on the K and V streams (the B/C-bias role that subsumes conv),
    so the layer keeps its (x, x_r8, cond) interface.

    Scan paths (identical math — smoke_test §12 asserts parity):
      - _scan_sequential: per-step loop, a direct transcription of
        mamba3_MIMO_step_ref (always available, reference semantics).
      - _scan_chunked:    Way 2C — parallel prefix inside small chunks
        (angle cumsum + lower-triangular decay GEMM): T/C Python steps
        instead of T, everything else batched matmul/cumsum. Same state
        carry semantics at chunk boundaries (K/V/θ/S all carried).

    Gradient checkpointing (self.gradient_checkpointing, set by train.py):
      the layer body recomputes activations in backward when enabled.
    """

    def __init__(self, cfg: TrackProcessorConfig):
        super().__init__()
        self.H = cfg.n_heads
        self.Rm = cfg.mimo_rank              # MIMO outer-product rank
        self.d = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        assert self.head_dim * cfg.n_heads == cfg.d_model, \
            "d_model must be divisible by n_heads"
        D = self.head_dim
        self.n_angles = max(2, (D // cfg.rotary_dim_divisor) // 2 * 2)  # even, ≤ D//2-ish

        # --- Δ (step size), trap mixture τ, and angle increments ---
        self.dt_proj    = nn.Linear(cfg.d_model, self.H, bias=True)
        self.dt_bias    = nn.Parameter(torch.full((self.H,), -2.0))
        self.trap_proj  = nn.Linear(cfg.d_model, self.H, bias=True)
        self.angle_proj = nn.Linear(cfg.d_model, self.H * self.n_angles, bias=True)
        self.log_dt_min = math.log(cfg.dt_min)
        self.log_dt_max = math.log(cfg.dt_max)

        # --- Rank-R_mimo Q/K/V/Z projections (x + x_r8 input coupling) ---
        qk_dim = self.Rm * self.H * D
        self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
        self.k_r8   = nn.Linear(cfg.R, qk_dim, bias=True)     # x_r8 → K stream
        self.v_proj = nn.Linear(cfg.d_model, self.H * D, bias=False)
        self.v_r8   = nn.Linear(cfg.R, self.H * D, bias=True) # x_r8 → V stream
        self.z_proj = nn.Linear(cfg.d_model, self.H * D, bias=False)

        # --- MIMO per-rank gains + D skip (per head) ---
        self.MIMO_V = nn.Parameter(torch.randn(self.H, self.Rm, D) / math.sqrt(D))
        self.MIMO_Z = nn.Parameter(torch.randn(self.H, self.Rm, D) / self.Rm)
        self.MIMO_O = nn.Parameter(torch.randn(self.H, self.Rm, D) / self.Rm)
        self.D_skip = nn.Parameter(torch.zeros(self.H))

        # --- Decay poles: A_h = -exp(log_a_real) < 0 (per head) ---
        self.log_a_real = nn.Parameter(torch.zeros(self.H))

        # --- Output ---
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm     = nn.LayerNorm(cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

        # --- Activation-memory control (honored; set by train.py) ---
        self.gradient_checkpointing = False

        # --- FiLM conditioning (zero-init: identity at step 0) ---
        self.film_proj    = nn.Linear(cfg.d_cond, cfg.d_model * 2, bias=True)
        self.dt_film_proj = nn.Linear(cfg.d_cond, self.H * 2, bias=True)
        with torch.no_grad():
            self.film_proj.weight.zero_()
            self.film_proj.bias.zero_()
            self.dt_film_proj.weight.zero_()
            self.dt_film_proj.bias.zero_()

        # Way 2C chunk length: small enough that the (C×C) decay matrices
        # stay tiny and exp() spreads stay in fp32 range, large enough to cut
        # Python loop overhead ~16× vs the per-step scan.
        self.scan_chunk = min(16, max(2, cfg.chunk_size // 8))
        self.use_chunked = True

    # ------------------------------------------------------------------ #
    # Rotary helper (from mamba3_MIMO_step_ref::apply_rotary_emb)         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotate(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        tensor: (B, H, R, D)  cos/sin: (B, H, 1, A) with A ≤ D//2.
        Rotates the first A complex pairs; remaining pairs pass through
        (cos padded to 1.0, sin padded to 0.0).
        """
        t = tensor.view(*tensor.shape[:-1], -1, 2)
        t0 = t[..., 0]
        t1 = t[..., 1]
        if cos.shape[-1] < t0.shape[-1]:
            pad = t0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)
        r0 = t0 * cos - t1 * sin
        r1 = t0 * sin + t1 * cos
        return torch.stack([r0, r1], dim=-1).view_as(tensor)

    def _init_state(self, B: int, device, ref: Tensor):
        D = self.head_dim
        return (
            torch.zeros(B, self.H, self.n_angles, device=device, dtype=ref.dtype),
            torch.zeros(B, self.H, D, D, device=device, dtype=ref.dtype),      # S (B,H,P,D)
            torch.zeros(B, self.H, self.Rm, D, device=device, dtype=ref.dtype),  # K state
            torch.zeros(B, self.H, self.Rm, D, device=device, dtype=ref.dtype),  # V state
        )

    # ------------------------------------------------------------------ #
    # Scan path 1: sequential reference (mamba3_MIMO_step_ref)            #
    # ------------------------------------------------------------------ #
    def _scan_sequential(
        self,
        q: Tensor, k: Tensor, v: Tensor, z: Tensor,     # (B,T,Rm,H,D) / (B,T,H,D)
        dt: Tensor, trap_raw: Tensor, angles: Tensor,   # (B,T,H) / (B,T,H,A)
        state=None,
    ) -> tuple[Tensor, tuple]:
        B, T, Rm, H, D = q.shape
        if state is None:
            theta, S, Kst, Vst = self._init_state(B, q.device, q)
        else:
            theta, S, Kst, Vst = (s.clone() for s in state)

        A_decay = -torch.exp(self.log_a_real)            # (H,) negative poles
        outs = []
        for t in range(T):
            dt_t = dt[:, t]                              # (B, H)
            trap_t = torch.sigmoid(trap_raw[:, t])       # (B, H)

            # 1. Data-dependent rotary angle state
            theta = theta + torch.tanh(angles[:, t]) * dt_t.unsqueeze(-1) * math.pi
            cos = torch.cos(theta).unsqueeze(2)          # (B, H, 1, A)
            sin = torch.sin(theta).unsqueeze(2)

            q_t = self._rotate(q[:, t].permute(0, 2, 1, 3), cos, sin)  # (B,H,Rm,D)
            k_t = self._rotate(k[:, t].permute(0, 2, 1, 3), cos, sin)

            # Rank projections (per-rank gain copies of V/Z)
            v_rank = torch.einsum('bhd,hrd->bhrd', v[:, t], self.MIMO_V)  # (B,H,Rm,D)
            z_rank = torch.einsum('bhd,hrd->bhrd', z[:, t], self.MIMO_Z)

            # 2. Exponential-trapezoidal state update
            alpha = torch.exp(dt_t * A_decay)            # (B, H)
            beta  = (1.0 - trap_t) * dt_t * alpha
            gamma = trap_t * dt_t

            prev_kv = torch.einsum('bhrd,bhrp->bhpd', Kst, Vst)
            curr_kv = torch.einsum('bhrd,bhrp->bhpd', k_t, v_rank)
            S = (alpha.unsqueeze(-1).unsqueeze(-1) * S
                 + beta.unsqueeze(-1).unsqueeze(-1) * prev_kv
                 + gamma.unsqueeze(-1).unsqueeze(-1) * curr_kv)

            # 3. Rank-R readout + D skip + Swish gate + MIMO_O
            out = torch.einsum('bhpd,bhrd->bhrp', S, q_t)          # (B,H,Rm,P)
            out = out + self.D_skip[None, :, None, None] * v_rank
            out = out * z_rank * torch.sigmoid(z_rank)
            out = torch.einsum('bhrp,hrp->bhp', out, self.MIMO_O)  # (B,H,P)

            outs.append(out)
            Kst = k_t
            Vst = v_rank

        return torch.stack(outs, dim=1), (theta, S, Kst, Vst)

    # ------------------------------------------------------------------ #
    # Scan path 2: Way 2C chunked parallel scan                           #
    # ------------------------------------------------------------------ #
    def _scan_chunked(
        self,
        q: Tensor, k: Tensor, v: Tensor, z: Tensor,
        dt: Tensor, trap_raw: Tensor, angles: Tensor,
        state=None,
    ) -> tuple[Tensor, tuple]:
        """
        Identical recurrence to _scan_sequential, parallelized within
        chunks of scan_chunk steps:
          - angle state: cumsum (purely additive → trivially parallel)
          - S recurrence: S_t = α_t·S_{t-1} + u_t with scalar α per (B,H)
            solved inside the chunk by a lower-triangular decay GEMM:
            W[t,k] = exp(g_t − g_{k−1}) (g = within-chunk cumsum of log α),
            plus the carried S at the chunk boundary scaled by exp(g_t).
          - readout per chunk, batched over the chunk's steps.
        Chunk boundaries carry (θ, S, K, V) exactly like per-step updates.
        """
        B, T, Rm, H, D = q.shape
        if state is None:
            theta, S, Kst, Vst = self._init_state(B, q.device, q)
        else:
            theta, S, Kst, Vst = (s.clone() for s in state)

        A_decay = -torch.exp(self.log_a_real)            # (H,)
        C = self.scan_chunk
        outs = []

        for t0 in range(0, T, C):
            t1 = min(t0 + C, T)
            Lc = t1 - t0

            dt_c = dt[:, t0:t1]                          # (B, L, H)
            trap_c = torch.sigmoid(trap_raw[:, t0:t1])   # (B, L, H)
            ang_c = angles[:, t0:t1]                     # (B, L, H, A)

            # ── 1. Angle state: within-chunk cumsum + carry ──
            inc = torch.tanh(ang_c) * dt_c.unsqueeze(-1) * math.pi   # (B,L,H,A)
            theta_seq = theta.unsqueeze(1) + torch.cumsum(inc, dim=1)
            theta = theta_seq[:, -1]
            # layout (B, H, L, 1, A) to broadcast over the rank axis of
            # the (B, H, L, Rm, D) tensors below
            cos = torch.cos(theta_seq).permute(0, 2, 1, 3).unsqueeze(3)
            sin = torch.sin(theta_seq).permute(0, 2, 1, 3).unsqueeze(3)

            # ── Rotate + rank projections over the whole chunk ──
            # per-rank layout: (B, H, L, Rm, D)
            q_c = self._rotate(q[:, t0:t1].permute(0, 3, 1, 2, 4), cos, sin)
            k_c = self._rotate(k[:, t0:t1].permute(0, 3, 1, 2, 4), cos, sin)
            v_c = torch.einsum('bhld,hrd->bhlrd',
                               v[:, t0:t1].permute(0, 2, 1, 3), self.MIMO_V)
            z_c = torch.einsum('bhld,hrd->bhlrd',
                               z[:, t0:t1].permute(0, 2, 1, 3), self.MIMO_Z)

            # ── 2. Trapezoidal input terms ──
            prev_kv = torch.einsum(
                'bhlrd,bhlrp->bhlpd',
                torch.cat([Kst.unsqueeze(2), k_c[:, :, :-1]], dim=2),
                torch.cat([Vst.unsqueeze(2), v_c[:, :, :-1]], dim=2),
            )                                            # (B,H,L,P,D)
            curr_kv = torch.einsum('bhlrd,bhlrp->bhlpd', k_c, v_c)

            alpha = torch.exp(dt_c * A_decay)            # (B, L, H)
            beta  = (1.0 - trap_c) * dt_c * alpha
            gamma = trap_c * dt_c
            u = (beta.permute(0, 2, 1)[:, :, :, None, None] * prev_kv
                 + gamma.permute(0, 2, 1)[:, :, :, None, None] * curr_kv)  # (B,H,L,P,D)

            # ── Triangular decay GEMM scan within the chunk ──
            # S_t = α_t·S_{t-1} + u_t  ⇒  u_k enters S_t with weight
            # Π_{i=k+1..t} α_i = exp(g_t − g_k)  (unit diagonal, k = t),
            # and the carry S_{t0-1} enters with weight exp(g_t) (g_{-1} = 0).
            la = (dt_c * A_decay).permute(0, 2, 1)       # (B, H, L) log α
            g = torch.cumsum(la, dim=2)                  # (B, H, L)
            logW = g.unsqueeze(-1) - g.unsqueeze(-2)     # (B,H,L_t,L_k)
            W = torch.tril(torch.exp(logW))              # (B,H,L,L)

            u_flat = u.reshape(B, H, Lc, D * D)          # (B,H,L,P*D)
            S_flat = torch.einsum('bhik,bhkp->bhip', W, u_flat)
            S_seq = S_flat.reshape(B, H, Lc, D, D)
            # carry-in: S at chunk start, decayed to each step
            S_seq = S_seq + torch.exp(g)[:, :, :, None, None] * S.unsqueeze(2)
            S = S_seq[:, :, -1]

            # ── 3. Readout over the chunk ──
            out = torch.einsum('bhlpd,bhlrd->bhlrp', S_seq, q_c)   # (B,H,L,Rm,P)
            out = out + self.D_skip[None, :, None, None, None] * v_c
            out = out * z_c * torch.sigmoid(z_c)
            out = torch.einsum('bhlrp,hrp->bhlp', out, self.MIMO_O)  # (B,H,L,P)
            outs.append(out.permute(0, 2, 1, 3))         # → (B, L, H, P)

            Kst = k_c[:, :, -1]
            Vst = v_c[:, :, -1]

        return torch.cat(outs, dim=1), (theta, S, Kst, Vst)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Tensor,           # (B, T, d_model) — track embedding
        x_r8: Tensor,        # (B, T, R=8) — raw structured features (input coupling)
        cond: Optional[Tensor] = None,   # (B, T, d_cond) — FiLM conditioning
        state: Optional[tuple] = None,   # recurrent state carry-in (streaming)
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, tuple]:
        """
        Returns: (B, T, d_model), or (output, state) when return_state=True.

        If ``self.gradient_checkpointing`` is True and gradients are enabled,
        the body runs under ``torch.utils.checkpoint`` (non-reentrant) so
        training recomputes activations in backward. The streaming path
        (state carry) skips checkpointing.
        """
        if (self.training and self.gradient_checkpointing and torch.is_grad_enabled()
                and state is None and not return_state):
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, x_r8, cond, use_reentrant=False
            )
        return self._forward_impl(x, x_r8, cond, state=state, return_state=return_state)

    def _forward_impl(
        self,
        x: Tensor,
        x_r8: Tensor,
        cond: Optional[Tensor] = None,
        state: Optional[tuple] = None,
        return_state: bool = False,
    ):
        B, T, D = x.shape
        residual = x

        # --- FiLM on input (session / cross-track conditioning) ---
        if cond is not None:
            gamma_beta = self.film_proj(cond)             # (B, T, D*2)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        # --- Δ, trap τ, angle increments ---
        dt_raw = self.dt_proj(x)                          # (B, T, H)
        if cond is not None:
            dt_gamma_beta = self.dt_film_proj(cond)       # (B, T, H*2)
            dt_gamma, dt_beta = dt_gamma_beta.chunk(2, dim=-1)
            dt_raw = dt_raw * (1 + dt_gamma) + dt_beta
        dt = F.softplus(dt_raw + self.dt_bias)
        dt = dt.clamp(min=math.exp(self.log_dt_min), max=math.exp(self.log_dt_max))

        trap_raw = self.trap_proj(x)                      # (B, T, H)
        angles   = self.angle_proj(x).view(B, T, self.H, self.n_angles)

        # --- Rank-R_mimo Q/K/V/Z (x + x_r8 input coupling on K/V) ---
        q = self.q_proj(x).view(B, T, self.Rm, self.H, self.head_dim)
        k = (self.k_proj(x) + self.k_r8(x_r8)).view(B, T, self.Rm, self.H, self.head_dim)
        v = (self.v_proj(x) + self.v_r8(x_r8)).view(B, T, self.H, self.head_dim)
        z = self.z_proj(x).view(B, T, self.H, self.head_dim)

        # --- Scan in fp32 (rotary + decay math is precision-sensitive) ---
        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f, k_f, v_f, z_f = (t.float() for t in (q, k, v, z))
            dt_f, trap_f, ang_f = (t.float() for t in (dt, trap_raw, angles))
            args = (q_f, k_f, v_f, z_f, dt_f, trap_f, ang_f)
            if self.use_chunked:
                out, new_state = self._scan_chunked(*args, state=state)
            else:
                out, new_state = self._scan_sequential(*args, state=state)

        # --- Output projection + residual ---
        out = out.reshape(B, T, D).to(residual.dtype)     # (B, T, d_model)
        out = self.out_proj(out)
        out = self.drop(out)
        out = self.norm(out + residual)
        if return_state:
            return out, new_state
        return out


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

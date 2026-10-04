"""
Synthetic degradation pipeline for self-supervised restoration training (Recipe 1A).

The training objective is:  model(degrade(clean_stem)) -> clean_stem

Each op mimics a realistic "bad capture / bad session edit" failure mode that the
Option B per-track processor (and the downstream edge controller) must repair:

  1. EQ damage      — resonant boosts/cuts on a smooth random log-frequency curve
  2. Compression    — over-compression (low threshold, moderate ratio, pumping)
  3. Saturation     — nonlinear waveshaping (clipping / grit)
  4. Reverb / room  — decaying-noise convolution tail (boxy room capture)

Design constraints:
  - Pure PyTorch (no torchaudio ops) so it runs on CPU, CUDA and ROCm gfx1103.
  - Device-agnostic: all randomness comes from a CPU ``torch.Generator`` so the
    same seed gives the same degradation on any accelerator.
  - Batch-safe on any leading shape (B, C, L) / (C, L) / (L,).
  - Pitch-preserving by construction (no time-stretch / resample), so the
    offline CREPE f0 conditioning of the clean stem remains valid for the
    degraded input.  ``target_pitched`` gating is handled in ``model.forward``.
  - Cheap: one rfft pair per clip plus a handful of pools. Runs on the training
    GPU inside the step loop (after batch collation), zero cost at inference.

Validation uses a reseeded generator (``reseed()``) so the val loss is
deterministic across checks — required for stable early stopping.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class StemDegradation:
    """
    Random EQ + compression + saturation + reverb degradation.

    Usage:
        degrade = StemDegradation(sample_rate=44100, seed=0)
        y = degrade(x)                # x: (B, C, L) or (C, L) or (L,)

        # Deterministic validation pass:
        degrade.reseed(1234)
        y_val = degrade(x_val)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        seed: Optional[int] = None,
        # per-op application probabilities (drawn independently)
        p_eq: float = 0.55,
        p_comp: float = 0.45,
        p_sat: float = 0.40,
        p_reverb: float = 0.35,
        # severity scalars (1.0 = nominal ranges documented per op)
        severity: float = 1.0,
    ):
        self.sr = sample_rate
        self.p_eq = p_eq
        self.p_comp = p_comp
        self.p_sat = p_sat
        self.p_reverb = p_reverb
        self.severity = severity
        self.gen = torch.Generator(device='cpu')
        if seed is None:
            # nondeterministic seed
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        self.gen.manual_seed(int(seed))

    # ------------------------------------------------------------------ #
    # RNG helpers — everything is drawn on CPU, then moved to the device  #
    # ------------------------------------------------------------------ #
    def reseed(self, seed: int) -> None:
        """Reset the RNG (call at the start of each validation run)."""
        self.gen.manual_seed(int(seed))

    def _rand(self, *shape: int, device: torch.device, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        u = torch.rand(*shape, generator=self.gen, device='cpu')
        return (u * (hi - lo) + lo).to(device)

    def _coin(self, p: float) -> bool:
        return bool(torch.rand(1, generator=self.gen, device='cpu').item() < p)

    # ------------------------------------------------------------------ #
    # Public API                                                        #
    # ------------------------------------------------------------------ #
    def __call__(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            raise TypeError("StemDegradation expects a torch.Tensor")
        if x.numel() == 0:
            return x
        orig_shape = x.shape
        # Flatten leading dims → (N, 1, L) for batched processing
        L = orig_shape[-1]
        flat = x.reshape(-1, 1, L).float()
        y = flat

        if self._coin(self.p_eq):
            y = self._eq_damage(y)
        if self._coin(self.p_comp):
            y = self._compress(y)
        if self._coin(self.p_sat):
            y = self._saturate(y)
        if self._coin(self.p_reverb):
            y = self._reverb(y)

        # Keep the overall level sane (restoration must not be a gain task)
        y = self._match_rms(flat, y)
        return y.reshape(orig_shape).to(x.dtype)

    # ------------------------------------------------------------------ #
    # Individual degradations                                           #
    # ------------------------------------------------------------------ #
    def _eq_damage(self, y: Tensor) -> Tensor:
        """Smooth random log-frequency gain curve (±3..±12 dB), applied via rfft."""
        N, _, L = y.shape
        n_bins = L // 2 + 1
        n_ctrl = 7  # control points across the log-frequency axis

        # Control-point gains per clip: mix of mild curves and resonant bumps
        deep = self._coin(0.3)  # occasionally a nasty resonant boost
        amp = 12.0 if deep else 6.0
        amp *= self.severity
        ctrl = self._rand(N, n_ctrl, device=y.device, lo=-amp, hi=amp)

        # Optional resonant spike: one control point gets a big boost
        if deep:
            idx = int(torch.randint(0, n_ctrl, (1,), generator=self.gen).item())
            spike = self._rand(1, device=y.device, lo=6.0, hi=12.0).item()
            ctrl[:, idx] = ctrl[:, idx] + spike

        # Interpolate control points over LINEAR bin index. (Not a log-frequency
        # warp: the 7 control points are spaced evenly in Hz, so most of the
        # curve's resolution lands above ~3 kHz. Good enough for augmentation;
        # revisit if low-frequency EQ damage needs finer control.)
        gains_db = F.interpolate(ctrl.unsqueeze(1), size=n_bins,
                                 mode='linear', align_corners=True).squeeze(1)
        gains = torch.pow(10.0, gains_db / 20.0).to(torch.float32)

        spec = torch.fft.rfft(y.squeeze(1), dim=-1)          # (N, n_bins) complex
        spec = spec * gains
        out = torch.fft.irfft(spec, n=L, dim=-1)             # (N, L)
        return out.unsqueeze(1)

    def _compress(self, y: Tensor) -> Tensor:
        """Over-compression: envelope detector + soft-ratio gain reduction + release smoothing."""
        N, C, L = y.shape
        # Attack envelope (causality irrelevant for augmentation)
        k_att = max(8, int(self.sr * 0.001))   # ~1 ms
        k_rel = max(64, int(self.sr * 0.05))   # ~50 ms release smoothing
        if k_att % 2 == 0:
            k_att += 1
        if k_rel % 2 == 0:
            k_rel += 1

        x_abs = y.abs()
        env = F.avg_pool1d(x_abs, kernel_size=k_att, stride=1, padding=k_att // 2)
        env = F.avg_pool1d(env, kernel_size=k_rel, stride=1, padding=k_rel // 2)

        # Threshold relative to clip RMS: 6..24 dB below (deep = aggressive)
        rms = y.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
        depth_db = self._rand(N, 1, 1, device=y.device, lo=6.0, hi=24.0) * self.severity
        thr = rms * torch.pow(10.0, -depth_db / 20.0)

        ratio = self._rand(N, 1, 1, device=y.device, lo=2.0, hi=8.0)
        over = (env / thr.clamp(min=1e-6)).clamp(min=1.0)
        gain = over.pow(1.0 / ratio - 1.0)                   # ≤ 1.0

        # Smooth the gain curve (release), then apply + light makeup
        gain = F.avg_pool1d(gain, kernel_size=k_rel, stride=1, padding=k_rel // 2)
        makeup = self._rand(N, 1, 1, device=y.device, lo=0.8, hi=1.6)
        return y * gain * makeup

    def _saturate(self, y: Tensor) -> Tensor:
        """Soft-clip waveshaping: tanh(drive · x) / drive (unity small-signal gain)."""
        drive_db = self._rand(y.shape[0], 1, 1, device=y.device, lo=3.0, hi=14.0) * self.severity
        drive = torch.pow(10.0, drive_db / 20.0)
        out = torch.tanh(y * drive) / drive
        # Sometimes blend only partially wet (parallel saturation)
        mix = self._rand(y.shape[0], 1, 1, device=y.device, lo=0.4, hi=1.0)
        return mix * out + (1.0 - mix) * y

    def _reverb(self, y: Tensor) -> Tensor:
        """Exponential-decay noise IR convolution (boxy room), FFT-based, batched."""
        N, C, L = y.shape
        tau = self._rand(1, device=y.device, lo=0.06, hi=0.45).item()  # RT-ish decay
        ir_len = int(self.sr * min(0.8, 4.0 * tau))
        ir_len = max(64, min(ir_len, L))

        t = torch.arange(ir_len, device=y.device, dtype=torch.float32) / self.sr
        decay = torch.exp(-t / tau)
        noise = torch.randn(N, C, ir_len, generator=self.gen, device='cpu').to(y.device)
        ir = noise * decay
        ir[:, :, :8] *= torch.linspace(0.0, 1.0, 8, device=y.device)  # fade-in
        ir = ir / ir.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-6)

        wet = self._rand(N, 1, 1, device=y.device, lo=0.06, hi=0.30)

        # FFT convolution, 'full' truncated to L (crop tail — augmentation only)
        n_fft = 1 << (L + ir_len - 2).bit_length()
        Y = torch.fft.rfft(y.reshape(N * C, L), n=n_fft, dim=-1)
        H = torch.fft.rfft(ir.reshape(N * C, ir_len), n=n_fft, dim=-1)
        conv = torch.fft.irfft(Y * H, n=n_fft, dim=-1)[..., :L].reshape(N, C, L)
        return y * (1.0 - wet) + conv * wet

    @staticmethod
    def _match_rms(ref: Tensor, y: Tensor) -> Tensor:
        """Restore input RMS so restoration is not dominated by gain errors."""
        r_ref = ref.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        r_y = y.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        scale = (r_ref / r_y).clamp(min=0.25, max=4.0)
        return y * scale

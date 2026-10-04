"""
Mamba-3 edge controller: training + single-step ONNX export (Way 5A).

Replaces the legacy `mamba3_controller_quant.onnx` export path, which had two
export bugs traced in the binary:

  1. The recurrent state of all 3 layers was exposed as THREE unnamed ONNX
     inputs/outputs (`onnx::Mul_3`, `onnx::Mul_4`, `onnx::Mul_3168`,
     `onnx::Mul_4689`) instead of one packed state tensor.
  2. `torch.onnx.export` traced a Python `for t in range(100)` sliding-window
     loop, unrolling ~100 Gather/Add nodes per layer — despite an SSM hidden
     state that already summarizes the history in O(1).

This module exports the clean, true-recurrent signature (3 inputs, 2 outputs):

    observer_features : (B, 1, 12)   one Observer frame (single step)
    session_state     : (B, 16)      context[0:8] + 6C profile[8:16]
    hidden_state_in   : (B, 3, 64)   packed per-layer SSM states (real floats)
    ────────────────────────────────────────────────────────────────────────
    dsp_parameters    : (B, 32)      tanh-bounded, see PARAM_LAYOUT
    hidden_state_out  : (B, 3, 64)   fed back as next tick's hidden_state_in

The controller itself is Mamba-3 style: bilinear (trapezoidal) discretization
of complex poles, NO causal conv1d, FiLM conditioning from session_state
(which carries the 6C customer profile), rank-free MIMO input coupling.

Training distills a deterministic classical `GoldenPolicy` teacher (the
"Golden Reference" rules from ard_cynical_engineer.md) so the network learns
temporally-smoothed expert knob moves from Observer features.  The teacher
runs on synthetic feature sequences by default, so this script is runnable
end-to-end on any machine with torch — no dataset required.

CLI:
    python train_controller.py --steps 2000                       # distill
    python train_controller.py --steps 2000 --export controller.onnx
    python train_controller.py --export-only --weights ckpt.pt \
        --export controller.onnx --quantized mamba3_controller_quant.onnx
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 32-parameter DSP layout — MUST stay in sync with
# plugin/src/worker.rs (PARAM_LAYOUT) and auditor.rs clamps.
# Model outputs are tanh-bounded in [-1, 1]; `denormalize` maps to physical.
# ---------------------------------------------------------------------------

PARAM_LAYOUT = [
    ("eq_gains_db",      6, -9.0,     9.0,     "linear"),
    ("eq_freqs_hz",      6, 40.0,    12000.0, "log"),
    ("eq_q",             6, 0.3,      8.0,    "log"),
    ("comp_threshold_db", 1, -60.0,   0.0,    "linear"),
    ("comp_ratio",       1,  1.0,    12.0,    "linear"),
    ("comp_attack_ms",   1,  0.1,   100.0,    "log"),
    ("comp_release_ms",  1,  5.0,   1000.0,   "log"),
    ("comp_mix",         1,  0.0,     1.0,    "linear"),
    ("sat_drive_db",     1,  0.0,    12.0,    "linear"),   # auditor clamps ≤ 8
    ("sat_mix",          1,  0.0,     1.0,    "linear"),
    ("output_gain_db",   1, -12.0,  12.0,     "linear"),
    ("low_cut_hz",       1, 20.0,   120.0,    "log"),
    ("tilt_db",          1, -6.0,    6.0,     "linear"),
    ("presence_db",      1, -6.0,    6.0,     "linear"),
    ("gate_threshold_db", 1, -80.0, -20.0,    "linear"),
    ("warmth",           1,  0.0,    1.0,     "linear"),
    ("master_mix",       1,  0.0,    1.0,     "linear"),
]
PARAM_COUNT = sum(n for _, n, _, _, _ in PARAM_LAYOUT)  # 32


def denormalize(p: Tensor) -> dict[str, Tensor]:
    """Map tanh-space (B, 32) in [-1,1] → physical parameter dict."""
    out: dict[str, Tensor] = {}
    idx = 0
    for name, n, lo, hi, scale in PARAM_LAYOUT:
        chunk = p[..., idx:idx + n]
        if scale == "log":
            phys = lo * (hi / lo) ** ((chunk + 1.0) / 2.0)
        else:
            phys = lo + (hi - lo) * (chunk + 1.0) / 2.0
        out[name] = phys
        idx += n
    return out


# ---------------------------------------------------------------------------
# GoldenPolicy — the deterministic "Golden Reference" teacher
# ---------------------------------------------------------------------------

class GoldenPolicy:
    """
    Classical rule-based mapping: Observer frame + session state → 32 knob
    targets in tanh-space [-1, 1] (ard_cynical_engineer.md "Golden Reference").

    Philosophy (Root Constitution): correct the minimum necessary, preserve
    transients first, never normalize.  These rules only ever *propose*; the
    AuditorAgent clamps the final committed parameters in the plugin.
    """

    @staticmethod
    def _lin(lo: float, hi: float, v: float) -> float:
        """Physical value → tanh-space [-1,1]."""
        v = max(lo, min(hi, v))
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    @staticmethod
    def _log(lo: float, hi: float, v: float) -> float:
        v = max(lo, min(hi, v))
        return 2.0 * (math.log(v / lo) / math.log(hi / lo)) - 1.0

    def __call__(self, obs: Tensor, session: Tensor) -> Tensor:
        """
        obs:     (B, T, 12) observer features (normalized, see prompt_stack)
        session: (B, 16)
        returns: (B, T, 32) tanh-space targets
        """
        B, T, _ = obs.shape
        dev = obs.device
        out = torch.zeros(B, T, PARAM_COUNT, device=dev)

        # Unpack observer features (feature order from prompt_stack.compile_input_block)
        # 0 mode, 1 true_peak, 2 rms, 3 crest/10, 4 centroid/20k, 5 zcr/1024,
        # 6 transient_slope, 7..11 band energies (bands 0-4 of the Observer's
        # 6-band mel bank — prompt_stack packs 5 of them into the 12-D vector).
        peak     = obs[..., 1]
        rms      = obs[..., 2]
        crest    = obs[..., 3] * 10.0
        bands    = obs[..., 7:12]                     # (B, T, 5)
        transient = obs[..., 6]

        band_mean = bands.mean(-1, keepdim=True)
        excess = (bands - band_mean) * 10.0          # normalized energy excess

        # 1. Corrective EQ: cut what pokes out above the mix, gently.
        #    Bands 0-4 get adaptive gains; band 5 (11 kHz bell) stays neutral.
        for i in range(5):
            gain_db = (-4.0 * excess[..., i]).clamp(-6.0, 3.0)   # minimal moves
            out[..., i] = self._batched_lin(-9.0, 9.0, gain_db)
        out[..., 5] = self._batched_lin(-9.0, 9.0, torch.zeros_like(peak))

        # 2. Default EQ placement (log-space params at neutral positions)
        for i, f in enumerate([80.0, 240.0, 700.0, 2000.0, 5000.0, 11000.0]):
            out[..., 6 + i] = self._batched_log(40.0, 12000.0, torch.full_like(peak, f))
        for i in range(6):
            out[..., 12 + i] = self._batched_log(0.3, 8.0, torch.full_like(peak, 1.0))

        # 3. Gentle glue compression when loud + dense (low crest), louder RMS
        density = (1.0 - crest / 12.0).clamp(0.0, 1.0)
        drive_level = (rms * 4.0).clamp(0.0, 1.0)
        out[..., 18] = self._batched_lin(-60.0, 0.0, -18.0 - 10.0 * drive_level)
        out[..., 19] = self._batched_lin(1.0, 12.0, 1.5 + 3.0 * density)
        out[..., 20] = self._batched_log(0.1, 100.0, torch.full_like(peak, 10.0))
        out[..., 21] = self._batched_log(5.0, 1000.0, torch.full_like(peak, 120.0))
        out[..., 22] = self._batched_lin(0.0, 1.0, 0.25 + 0.25 * density)

        # 4. Saturation only for sparse material (preserve transients!)
        sat = (0.35 * (1.0 - density) * (1.0 - transient.clamp(0, 1))).clamp(0, 2.5)
        out[..., 23] = self._batched_lin(0.0, 12.0, sat * 4.0)
        out[..., 24] = self._batched_lin(0.0, 1.0, 0.2 + 0.2 * (1.0 - density))

        # 5. Never normalize: keep output gain at unity, master mix full wet
        out[..., 25] = self._batched_lin(-12.0, 12.0, torch.zeros_like(peak))
        out[..., 26] = self._batched_log(20.0, 120.0, torch.full_like(peak, 30.0))
        out[..., 27] = self._batched_lin(-6.0, 6.0, torch.zeros_like(peak))
        out[..., 28] = self._batched_lin(-6.0, 6.0, torch.zeros_like(peak))
        out[..., 29] = self._batched_lin(-80.0, -20.0, torch.full_like(peak, -70.0))
        out[..., 30] = self._batched_lin(0.0, 1.0, torch.zeros_like(peak))
        out[..., 31] = self._batched_lin(0.0, 1.0, torch.ones_like(peak))
        return out

    # vectorized helpers ------------------------------------------------- #
    @staticmethod
    def _batched_lin(lo: float, hi: float, v: Tensor) -> Tensor:
        v = v.clamp(lo, hi)
        return (2.0 * (v - lo) / (hi - lo) - 1.0)

    @staticmethod
    def _batched_log(lo: float, hi: float, v: Tensor) -> Tensor:
        v = v.clamp(lo, hi)
        return (2.0 * (torch.log(v / lo) / math.log(hi / lo)) - 1.0)


# ---------------------------------------------------------------------------
# Mamba3Controller — 3 layers, d_model=64, per-layer complex state (2×16)
# packed as 64 real floats. Single-step recurrent, export-friendly.
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    obs_dim:  int = 12
    sess_dim: int = 16
    d_model:  int = 64
    n_layers: int = 3
    n_heads:  int = 2      # complex heads per layer
    d_state:  int = 16     # complex state dim per head → 2*16*2 = 64 floats
    dt_min:   float = 1e-3
    dt_max:   float = 0.2
    hidden_dim: int = 64   # per-layer packed state size (heads * d_state * 2)


class Mamba3StepLayer(nn.Module):
    """One recurrent Mamba-3 step: trapezoidal complex poles, no conv1d."""

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        H, N = cfg.n_heads, cfg.d_state
        self.H, self.N = H, N

        # Complex poles per head: α + jω  (α forced negative via -exp)
        self.log_a_real = nn.Parameter(torch.zeros(H))
        self.a_imag = nn.Parameter(torch.linspace(0.1, math.pi, H + 1)[:-1].clone())

        self.B_proj = nn.Linear(cfg.d_model, H * N * 2, bias=True)   # → complex (H, N)
        self.C_proj = nn.Linear(H * N * 2, cfg.d_model, bias=False)
        self.dt_proj = nn.Linear(cfg.d_model, H, bias=True)
        self.dt_bias = nn.Parameter(torch.full((H,), -2.0))

        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # FiLM from session_state (carries the 6C profile) — zero-init
        self.film = nn.Linear(cfg.sess_dim, cfg.d_model * 2, bias=True)
        self.dt_film = nn.Linear(cfg.sess_dim, H * 2, bias=True)
        for m in (self.film, self.dt_film):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def step(self, u: Tensor, session: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        u:       (B, d_model) — this step's features
        session: (B, 16)
        h:       (B, H, N) complex64 — layer state
        returns: (out (B, d_model), h_next (B, H, N) complex64)
        """
        cfg = self.cfg
        residual = u

        gamma, beta = self.film(session).chunk(2, dim=-1)
        u = u * (1.0 + gamma) + beta

        dt_raw = self.dt_proj(u)
        dg, db = self.dt_film(session).chunk(2, dim=-1)
        dt_raw = dt_raw * (1.0 + dg) + db
        dt = F.softplus(dt_raw + self.dt_bias).clamp(cfg.dt_min, cfg.dt_max)  # (B, H)

        # Trits in fp32 for complex arithmetic stability
        with torch.autocast(device_type=u.device.type, enabled=False):
            dt = dt.float()
            a_cont = torch.complex(-torch.exp(self.log_a_real.float()), self.a_imag.float())  # (H,)
            half = dt.unsqueeze(-1) * a_cont / 2.0                        # (B, H) complex
            A_d = (1.0 + half) / (1.0 - half)                             # (B, H) complex
            B_d = dt.to(torch.complex64) / (1.0 - half)                   # (B, H) complex

            bc = self.B_proj(u.float())                                   # (B, H*N*2)
            br, bi = bc.chunk(2, dim=-1)
            b_vec = torch.complex(br, bi).reshape(-1, self.H, self.N)     # (B, H, N)

            h = A_d.unsqueeze(-1) * h + B_d.unsqueeze(-1) * b_vec         # (B, H, N)
            h_cat = torch.cat([h.real, h.imag], dim=-1).reshape(-1, self.H * self.N * 2)
            y = self.C_proj(h_cat)                                        # (B, d_model)

        out = self.out_proj(y.to(residual.dtype))
        return self.norm(out + residual), h


class Mamba3Controller(nn.Module):
    """
    Full controller: input proj → 3× Mamba3StepLayer → tanh parameter head.

    forward_step() is the exported ONNX graph (single time step).
    forward() loops over time for training and carries the state.
    """

    def __init__(self, cfg: Optional[ControllerConfig] = None):
        super().__init__()
        self.cfg = cfg or ControllerConfig()
        c = self.cfg
        self.in_proj = nn.Linear(c.obs_dim, c.d_model)
        self.layers = nn.ModuleList([Mamba3StepLayer(c) for _ in range(c.n_layers)])
        self.param_head = nn.Linear(c.d_model, PARAM_COUNT)
        self.sess_bias = nn.Linear(c.sess_dim, c.d_model, bias=False)  # session steer

    def init_hidden(self, batch: int, device=None) -> Tensor:
        c = self.cfg
        return torch.zeros(batch, c.n_layers, c.n_heads, c.d_state,
                           dtype=torch.complex64, device=device)

    def _flat_hidden(self, h: Tensor) -> Tensor:
        """(B, L, H, N) complex → (B, L, H*N*2) real  (= (B, L, 64))."""
        B, L, H, N = h.shape
        return torch.cat([h.real, h.imag], dim=-1).reshape(B, L, H * N * 2)

    def _unflat_hidden(self, hf: Tensor) -> Tensor:
        """(B, L, 64) real → (B, L, H, N) complex."""
        B, L, _ = hf.shape
        c = self.cfg
        half = c.n_heads * c.d_state
        hr, hi = hf[..., :half], hf[..., half:]
        return torch.complex(hr, hi).reshape(B, L, c.n_heads, c.d_state)

    def forward_step(self, obs: Tensor, session: Tensor, hidden_state_in: Tensor) -> tuple[Tensor, Tensor]:
        """One step. obs: (B, 1, 12); session: (B, 16); state: (B, 3, 64)."""
        u = self.in_proj(obs[:, 0, :]) + self.sess_bias(session)   # (B, d)
        u = F.silu(u)
        h_list = []
        h_states = self._unflat_hidden(hidden_state_in)            # (B, L, H, N)
        for l, layer in enumerate(self.layers):
            u, h_next = layer.step(u, session, h_states[:, l])
            h_list.append(h_next)
        params = torch.tanh(self.param_head(u))                    # (B, 32)
        h_out = torch.stack(
            [torch.cat([h.real, h.imag], dim=-1).reshape(h.shape[0], -1) for h in h_list],
            dim=1,
        )  # (B, 3, 64)
        return params, h_out

    def forward(self, obs_seq: Tensor, session: Tensor,
                hidden_state_in: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Training path. obs_seq: (B, T, 12). Returns (B, T, 32), (B, 3, 64)."""
        B, T, _ = obs_seq.shape
        if hidden_state_in is None:
            hidden_state_in = self._flat_hidden(self.init_hidden(B, obs_seq.device))
        outs = []
        h = hidden_state_in
        for t in range(T):
            p, h = self.forward_step(obs_seq[:, t:t + 1, :], session, h)
            outs.append(p)
        return torch.stack(outs, dim=1), h


# ---------------------------------------------------------------------------
# Synthetic feature sequences (runnable without any dataset)
# ---------------------------------------------------------------------------

def synthetic_batch(batch: int, seq_len: int, device) -> tuple[Tensor, Tensor]:
    """Plausible Observer-feature sequences + random session states."""
    t = torch.linspace(0, 6 * math.pi, seq_len, device=device)
    base = 0.25 + 0.15 * torch.sin(t)[None, :, None]
    obs = base.repeat(batch, 1, 12)
    obs = obs + 0.05 * torch.randn_like(obs)
    # rough normalization ranges used by prompt_stack
    obs[..., 1] = obs[..., 1].clamp(0, 1)         # true_peak
    obs[..., 2] = obs[..., 2].clamp(0, 1) * 0.5   # rms
    obs[..., 3] = obs[..., 3].clamp(0, 1)         # crest/10
    obs[..., 4] = obs[..., 4].clamp(0, 1) * 0.1   # centroid/20k
    obs[..., 6] = obs[..., 6].clamp(0, 1)         # transient slope
    obs[..., 7:12] = torch.rand(batch, seq_len, 5, device=device)  # 5 packed bands
    sess = torch.zeros(batch, 16, device=device)
    sess[:, 4] = 0.6    # tempo
    sess[:, 6] = 0.459  # 44.1k / 96k
    return obs, sess


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------

def train(steps: int, device: str, lr: float, seq_len: int, batch: int,
          smooth: float, save: Optional[str], seed: int = 0) -> Mamba3Controller:
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = Mamba3Controller().to(dev)
    teacher = GoldenPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba3Controller parameters: {n_params:,}")
    print(f"Distilling GoldenPolicy for {steps} steps (batch={batch}, T={seq_len})...")

    for step in range(1, steps + 1):
        obs, sess = synthetic_batch(batch, seq_len, dev)
        with torch.no_grad():
            target = teacher(obs, sess)

        pred, _ = model(obs, sess)
        loss_fit = F.mse_loss(pred, target)
        # temporal smoothness: knob streams should not jitter
        loss_smooth = (pred[:, 1:] - pred[:, :-1]).pow(2).mean()
        loss = loss_fit + smooth * loss_smooth

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % max(1, steps // 10) == 0 or step == 1:
            print(f"  [{step:6d}] loss={loss.item():.5f} fit={loss_fit.item():.5f} "
                  f"smooth={loss_smooth.item():.5f}")

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'cfg': model.cfg.__dict__}, save)
        print(f"Saved weights → {save}")
    return model


# ---------------------------------------------------------------------------
# ONNX export (3 inputs / 2 outputs, dynamic batch) + quantization handoff
# ---------------------------------------------------------------------------

def export_onnx(model: Mamba3Controller, path: str) -> None:
    model = model.cpu().eval()
    c = model.cfg
    obs = torch.zeros(1, 1, c.obs_dim)
    sess = torch.zeros(1, c.sess_dim)
    h = torch.zeros(1, c.n_layers, c.hidden_dim)

    class ExportWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, obs, sess, h):
            p, ho = self.m.forward_step(obs, sess, h)
            return p, ho

    wrapper = ExportWrapper(model).eval()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (obs, sess, h), path,
        input_names=['observer_features', 'session_state', 'hidden_state_in'],
        output_names=['dsp_parameters', 'hidden_state_out'],
        dynamic_axes={
            'observer_features': {0: 'batch'},
            'session_state':     {0: 'batch'},
            'hidden_state_in':   {0: 'batch'},
            'dsp_parameters':    {0: 'batch'},
            'hidden_state_out':  {0: 'batch'},
        },
        opset_version=17,
    )
    print(f"Exported ONNX → {path}")
    print("  inputs : observer_features (B,1,12), session_state (B,16), hidden_state_in (B,3,64)")
    print("  outputs: dsp_parameters (B,32), hidden_state_out (B,3,64)")


def main():
    ap = argparse.ArgumentParser(description="Mamba-3 edge controller: distill + export")
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--smooth', type=float, default=0.5, help='temporal smoothness weight')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save', default='compiler/checkpoints/controller.pt')
    ap.add_argument('--export', default=None, help='path for controller.onnx')
    ap.add_argument('--quantized', default=None, help='path for quantized ONNX (uses quantize.py)')
    ap.add_argument('--export-only', action='store_true', help='skip training, load --save')
    ap.add_argument('--weights', default=None, help='checkpoint for --export-only')
    args = ap.parse_args()

    if args.export_only:
        ckpt_path = args.weights or args.save
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ControllerConfig(**{k: v for k, v in ckpt['cfg'].items()
                                  if k in ControllerConfig.__dataclass_fields__})
        model = Mamba3Controller(cfg)
        model.load_state_dict(ckpt['model'])
    else:
        model = train(args.steps, args.device, args.lr, args.seq_len,
                      args.batch, args.smooth, args.save)

    if args.export:
        export_onnx(model, args.export)
        if args.quantized:
            from quantize import quantize_dynamic
            quantize_dynamic(args.export, args.quantized, verify=True)


if __name__ == '__main__':
    main()

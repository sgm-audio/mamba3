"""
Customer/Tone Profile prompt-tuning (Way 6C).

Instead of LoRA adapters or per-customer checkpoints, each customer gets an
8-D continuous profile embedding carried in `session_state[8:16]` (see
prompt_stack.py).  This script freezes ALL controller weights and optimizes
just those 8 floats so the controller's knob moves match the customer's
preferred reference settings on their own material.

Runtime cost of the approach:
  - Zero extra model weights (the profile is 8 float32 = 32 bytes in a preset).
  - Zero extra compute in the audio thread (profile rides inside the session
    state vector the worker already feeds the ONNX graph).
  - Per-customer "fine-tune" runs in seconds on CPU.

Two label sources for the profile fit:
  1. `--labels ref_params.npy`  — explicit reference knob settings
     (32-D tanh-space vectors, e.g. from the customer's hand-edited sessions).
  2. default                    — the GoldenPolicy teacher applied to the
     customer's Observer feature streams (consistent taste baseline).

Data layout: a directory of `.npy` files, each an Observer feature sequence
of shape (T, 12) as produced by the offline observer dump.

CLI:
    python fit_profile.py --data client_features/ --profile-out profile.npy
    python fit_profile.py --data client_features/ --labels client_targets.npy \
        --weights checkpoints/controller.pt --profile-out profile.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prompt_stack import PROFILE_DIM
from train_controller import GoldenPolicy, Mamba3Controller, ControllerConfig


def load_feature_dir(data_dir: str) -> list[torch.Tensor]:
    files = sorted(Path(data_dir).glob('*.npy'))
    if not files:
        raise SystemExit(f"No .npy feature files in {data_dir}")
    seqs = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[1] != 12:
            raise SystemExit(f"{f}: expected (T, 12), got {arr.shape}")
        seqs.append(torch.from_numpy(arr.astype(np.float32)))
    return seqs


def fit_profile(
    seqs: list[torch.Tensor],
    labels: torch.Tensor | None,
    weights: str | None,
    steps: int,
    lr: float,
    seq_len: int,
    device: str,
) -> np.ndarray:
    dev = torch.device(device)

    # Load the frozen controller (or just the teacher if no weights given)
    model = None
    if weights:
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        cfg = ControllerConfig(**{k: v for k, v in ckpt['cfg'].items()
                                  if k in ControllerConfig.__dataclass_fields__})
        model = Mamba3Controller(cfg)
        model.load_state_dict(ckpt['model'])
        model = model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)          # freeze EVERYTHING (6C contract)
    teacher = GoldenPolicy()

    # Session context skeleton: neutral context + trainable profile slots 8:16
    sess_ctx = torch.zeros(1, 8, device=dev)
    profile = torch.zeros(PROFILE_DIM, device=dev, requires_grad=True)
    opt = torch.optim.Adam([profile], lr=lr)

    print(f"Fitting {PROFILE_DIM}-D profile on {len(seqs)} sequences, "
          f"{steps} steps (weights {'frozen: ' + weights if weights else 'N/A — teacher only'})")

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        # Sample a random crop
        seq = seqs[torch.randint(0, len(seqs), (1,)).item()]
        T = seq.shape[0]
        start = torch.randint(0, max(1, T - seq_len), (1,)).item()
        crop = seq[start:start + seq_len].to(dev).unsqueeze(0)   # (1, t, 12)

        sess = torch.cat([sess_ctx, profile.clamp(-1, 1).unsqueeze(0)], dim=1)

        with torch.no_grad():
            if labels is not None:
                tgt = labels[torch.randint(0, labels.shape[0], (1,)).item()].to(dev)
                tgt = tgt.unsqueeze(0).unsqueeze(0).expand(1, crop.shape[1], -1)
            else:
                tgt = teacher(crop, sess)

        if model is not None:
            pred, _ = model(crop, sess)
        else:
            pred = teacher(crop, sess)   # degenerate: profile cannot affect
            if step == 1:
                print("  (no --weights: teacher-only fit — profile stays neutral)")
        loss = F.mse_loss(pred, tgt) + 1e-3 * profile.pow(2).mean()
        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == 1:
            print(f"  [{step:6d}] loss={loss.item():.6f} |profile|_inf="
                  f"{profile.detach().abs().max().item():.4f}")

    out = profile.detach().clamp(-1, 1).cpu().numpy().astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="6C: prompt-tune the 8-D customer profile")
    ap.add_argument('--data', required=True, help='dir of (T,12) observer .npy dumps')
    ap.add_argument('--labels', default=None, help='optional (K, 32) tanh-space reference params')
    ap.add_argument('--weights', default='compiler/checkpoints/controller.pt')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--profile-out', default='profile.npy')
    args = ap.parse_args()

    seqs = load_feature_dir(args.data)
    labels = None
    if args.labels:
        labels = torch.from_numpy(np.load(args.labels).astype(np.float32))
        if labels.ndim != 2 or labels.shape[1] != 32:
            raise SystemExit(f"--labels must be (K, 32), got {tuple(labels.shape)}")

    weights = args.weights if args.weights and Path(args.weights).exists() else None
    if args.weights and weights is None:
        print(f"  (weights {args.weights} not found — teacher-only mode)")

    profile = fit_profile(seqs, labels, weights, args.steps, args.lr,
                          args.seq_len, args.device)

    np.save(args.profile_out, profile)
    print(f"\nSaved profile → {args.profile_out}")
    print("Profile vector (session_state[8:16]):")
    print(np.array2string(profile, precision=4, separator=', '))
    json_path = str(Path(args.profile_out).with_suffix('.json'))
    with open(json_path, 'w') as f:
        json.dump({"profile": profile.tolist()}, f, indent=2)
    print(f"Preset JSON → {json_path}  (embed as session_state[8:16] at runtime)")


if __name__ == '__main__':
    main()

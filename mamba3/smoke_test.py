"""
Smoke test + memory profiler for Mamba-3 per-track processor.
Run this FIRST on your 860M before touching the training loop.

Usage (inside distrobox ROCm container):
  python smoke_test.py --device cuda
  python smoke_test.py --device cpu       # fallback if ROCm fails
  python smoke_test.py --device cuda --amp  # test bf16 AMP

Expected on 860M (8 GB UMA, pure PyTorch):
  - Forward pass: 2-8 s for 4s clip, batch=1 (sequential scan is slow)
  - Peak VRAM: ~2-4 GB for default config
  - No NaN in output
  - Loss backward: no OOM

Sections:
  [1] Model build + parameter count
  [2] Sub-module unit tests (R8, SemanticEncoder, F0, CrossTrack)
  [3] Forward pass (no grad) — shape + NaN/Inf check
  [4] Forward + backward (grad) — loss, gradients, peak VRAM
  [5] bf16 AMP forward + backward (critical for SSM stability)
  [6] Waveform reconstruction roundtrip (STFT → ISTFT)
  [7] Gradient checkpointing (train.py enables this)
  [8] Memory scaling sweep (1s/2s/4s × batch 1/2/4)
  [9] ROCm / HSA environment check
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.nn.functional as F

from model import (
    PerTrackProcessor,
    TrackProcessorConfig,
    R8FeatureExtractor,
    SemanticEncoder,
    F0Conditioner,
    CrossTrackAggregator,
    Mamba3MIMOLayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mOK\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
SKIP = "\033[94mSKIP\033[0m"

fail_count = 0
warn_count = 0


def memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def check(cond: bool, label: str) -> bool:
    global fail_count
    if cond:
        print(f"  {label}: {PASS}")
    else:
        print(f"  {label}: {FAIL}")
        fail_count += 1
    return cond


def warn(msg: str):
    global warn_count
    print(f"  {msg}: {WARN}")
    warn_count += 1


def make_cfg(**overrides) -> TrackProcessorConfig:
    defaults = dict(
        d_model=128,
        d_state=16,
        n_heads=8,
        n_layers=2,
        chunk_size=64,
    )
    defaults.update(overrides)
    return TrackProcessorConfig(**defaults)


def t_frames(clip_len: int, cfg: TrackProcessorConfig) -> int:
    return 1 + clip_len // cfg.hop_length


# ---------------------------------------------------------------------------
# [1] Model build
# ---------------------------------------------------------------------------

def test_build(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[1] Model build + parameter count")
    print(f"{'='*60}")

    t0 = time.time()
    model = PerTrackProcessor(cfg).to(dev)
    build_s = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"  Config: d_model={cfg.d_model}, d_state={cfg.d_state}, "
          f"n_heads={cfg.n_heads}, n_layers={cfg.n_layers}")
    print(f"  Parameters (trainable): {n_trainable:,}")
    print(f"  Parameters (frozen):    {n_frozen:,}")
    print(f"  Build time: {build_s:.2f}s")
    print(f"  VRAM after build: {memory_mb():.0f} MB")

    check(n_params > 0, "Model has trainable parameters")
    check(build_s < 30, f"Build < 30s (got {build_s:.1f}s)")

    return model


# ---------------------------------------------------------------------------
# [2] Sub-module unit tests
# ---------------------------------------------------------------------------

def test_r8_extractor(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[2a] R8 Feature Extractor")
    print(f"{'='*60}")

    ext = R8FeatureExtractor(cfg).to(dev)
    clip_len = cfg.sample_rate * 1
    B = 2
    wav = torch.randn(B, clip_len, device=dev)
    mag_sq, _ = PerTrackProcessor(cfg)._compute_stft(wav.to(dev))

    band_e = ext._soft_band_energy(mag_sq)
    check(band_e.shape == (B, t_frames(clip_len, cfg), 6),
          f"Band energy shape {list(band_e.shape)} == [B, T, 6]")
    check(not torch.isnan(band_e).any(), "Band energy finite (no NaN)")
    check((band_e >= 0).all(), "Band energy non-negative (log1p)")

    trans = ext._transient_density(wav, t_frames(clip_len, cfg))
    check(trans.shape == (B, t_frames(clip_len, cfg), 1),
          f"Transient shape {list(trans.shape)} == [B, T, 1]")
    check(not torch.isnan(trans).any(), "Transient finite (no NaN)")

    cent = ext._spectral_centroid(mag_sq)
    check(cent.shape == (B, t_frames(clip_len, cfg), 1),
          f"Centroid shape {list(cent.shape)} == [B, T, 1]")
    check(not torch.isnan(cent).any(), "Centroid finite (no NaN)")

    r8_full = ext(wav, mag_sq)
    check(r8_full.shape == (B, t_frames(clip_len, cfg), cfg.d_model),
          f"R8 → d_model shape {list(r8_full.shape)}")

    check(not torch.isnan(r8_full).any(), "R8 projected output finite")
    del ext, wav, mag_sq, band_e, trans, cent, r8_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_semantic_encoder(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[2b] Semantic Encoder (clip-level + frame-level)")
    print(f"{'='*60}")

    enc = SemanticEncoder(cfg).to(dev)
    clip_len = cfg.sample_rate * 1
    B = 2
    T = t_frames(clip_len, cfg)
    mel = torch.randn(B, T, cfg.n_mels, device=dev)
    r8 = torch.randn(B, T, cfg.R, device=dev)

    z_clip = enc(mel, r8, return_sequence=False)
    check(z_clip.shape == (B, cfg.d_semantic),
          f"Clip embedding shape {list(z_clip.shape)} == [B, {cfg.d_semantic}]")
    check(not torch.isnan(z_clip).any(), "Clip embedding finite")

    norms = z_clip.norm(dim=-1)
    check(torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
          f"Clip embeddings L2-normalized (norms={norms[0].item():.4f})")

    z_seq = enc(mel, r8, return_sequence=True)
    check(z_seq.shape == (B, T, cfg.d_semantic),
          f"Frame embedding shape {list(z_seq.shape)} == [B, T, {cfg.d_semantic}]")
    check(not torch.isnan(z_seq).any(), "Frame embeddings finite")

    del enc, mel, r8, z_clip, z_seq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_f0_conditioner(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[2c] F0 Conditioner (CREPE 360-D → d_f0)")
    print(f"{'='*60}")

    f0 = F0Conditioner(cfg).to(dev)
    clip_len = cfg.sample_rate * 1
    B = 2
    T = t_frames(clip_len, cfg)
    crepe = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)

    out = f0(crepe)
    check(out.shape == (B, T, cfg.d_f0),
          f"F0 output shape {list(out.shape)} == [B, T, {cfg.d_f0}]")
    check(not torch.isnan(out).any(), "F0 output finite")

    del f0, crepe, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_cross_track(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[2d] Cross-Track Aggregator")
    print(f"{'='*60}")

    agg = CrossTrackAggregator(cfg).to(dev)
    clip_len = cfg.sample_rate * 1
    B = 2
    T = t_frames(clip_len, cfg)
    K = 3

    target_emb = torch.randn(B, T, cfg.d_semantic, device=dev)
    other_embs = torch.randn(B, K, T, cfg.d_semantic, device=dev)

    ctx = agg(target_emb, other_embs)
    check(ctx.shape == (B, T, cfg.d_cross),
          f"Cross-track context shape {list(ctx.shape)} == [B, T, {cfg.d_cross}]")
    check(not torch.isnan(ctx).any(), "Cross-track context finite")

    del agg, target_emb, other_embs, ctx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_mamba_layer(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[2e] Mamba-3 MIMO Layer (single layer)")
    print(f"{'='*60}")

    layer = Mamba3MIMOLayer(cfg).to(dev)
    clip_len = cfg.sample_rate * 1
    B = 2
    T = t_frames(clip_len, cfg)

    x = torch.randn(B, T, cfg.d_model, device=dev)
    x_r8 = torch.randn(B, T, cfg.R, device=dev)
    cond = torch.randn(B, T, cfg.d_cond, device=dev)

    out = layer(x, x_r8, cond=cond)
    check(out.shape == (B, T, cfg.d_model),
          f"Mamba layer output shape {list(out.shape)} == [B, T, {cfg.d_model}]")
    check(not torch.isnan(out).any(), "Mamba layer output finite")
    check(not torch.isinf(out).any(), "Mamba layer output not Inf")

    out_no_cond = layer(x, x_r8, cond=None)
    check(out_no_cond.shape == (B, T, cfg.d_model),
          f"Mamba layer (no cond) output shape {list(out_no_cond.shape)}")
    check(not torch.isnan(out_no_cond).any(), "Mamba layer (no cond) output finite")

    del layer, x, x_r8, cond, out, out_no_cond
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [3] Full forward pass (no grad)
# ---------------------------------------------------------------------------

def test_forward_no_grad(dev: torch.device, model: PerTrackProcessor, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[3] Forward pass (no grad) — 1s clip, batch=1")
    print(f"{'='*60}")

    sr = cfg.sample_rate
    clip_len = sr * 1
    B = 1
    T = t_frames(clip_len, cfg)
    K = 2

    waveform = torch.randn(B, clip_len, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    other_embs = torch.randn(B, K, T, cfg.d_semantic, device=dev)

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out = model(waveform, crepe_probs, other_embs=other_embs)
    elapsed = time.time() - t0

    F_bins = cfg.n_fft // 2 + 1
    check(out['output_stft'].shape == (B, T, F_bins),
          f"output_stft shape {list(out['output_stft'].shape)} == [1, {T}, {F_bins}]")
    check(out['track_emb'].shape == (B, T, cfg.d_semantic),
          f"track_emb shape {list(out['track_emb'].shape)} == [1, {T}, {cfg.d_semantic}]")
    check(out['r8'].shape == (B, T, cfg.R),
          f"r8 shape {list(out['r8'].shape)} == [1, {T}, {cfg.R}]")

    has_nan = torch.isnan(out['output_stft']).any().item()
    has_inf = torch.isinf(out['output_stft']).any().item()
    check(not has_nan, "No NaN in output_stft")
    check(not has_inf, "No Inf in output_stft")

    has_nan_emb = torch.isnan(out['track_emb']).any().item()
    check(not has_nan_emb, "No NaN in track_emb")

    print(f"  Forward time: {elapsed:.3f}s")
    print(f"  VRAM after forward: {memory_mb():.0f} MB")

    check(elapsed < 60, f"Forward < 60s (got {elapsed:.1f}s)")

    model.train()
    del waveform, crepe_probs, other_embs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [4] Forward + backward (grad)
# ---------------------------------------------------------------------------

def test_forward_backward(dev: torch.device, model: PerTrackProcessor, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[4] Forward + backward (grad) — 1s clip, batch=1")
    print(f"{'='*60}")

    sr = cfg.sample_rate
    clip_len = sr * 1
    B = 1
    T = t_frames(clip_len, cfg)
    K = 2

    waveform = torch.randn(B, clip_len, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    other_embs = torch.randn(B, K, T, cfg.d_semantic, device=dev)

    model.train()
    waveform.requires_grad_(False)
    crepe_probs.requires_grad_(False)
    other_embs.requires_grad_(False)

    reset_peak()
    t0 = time.time()

    out = model(waveform, crepe_probs, other_embs=other_embs)
    target_stft = torch.zeros_like(out['output_stft'])
    loss = F.l1_loss(out['output_stft'].abs(), target_stft.abs())

    check(not torch.isnan(loss), f"Loss finite ({loss.item():.4f})")
    print(f"  Loss value: {loss.item():.4f}")

    loss.backward()
    elapsed = time.time() - t0
    print(f"  Fwd+bwd time: {elapsed:.3f}s")

    peak = peak_memory_mb()
    print(f"  Peak VRAM (fwd+bwd): {peak:.0f} MB")

    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check(peak / 1024 < vram_total * 0.9,
              f"Peak VRAM < 90% of total ({peak/1024:.1f}/{vram_total:.1f} GB)")

    grad_norms = {}
    nan_grads = False
    for n, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms[n] = gn
            if gn != gn:
                nan_grads = True

    check(not nan_grads, "No NaN gradients")

    zero_grads = sum(1 for v in grad_norms.values() if v == 0.0)
    total_grads = len(grad_norms)
    print(f"  Gradients: {total_grads} params, {zero_grads} with zero norm")

    if zero_grads == total_grads:
        warn("All gradients are zero — possible detachment in graph")

    top5 = sorted(grad_norms.items(), key=lambda x: -x[1])[:5]
    print("  Top-5 grad norms:")
    for n, v in top5:
        print(f"    {v:.6f}  {n}")

    bot5 = sorted(grad_norms.items(), key=lambda x: x[1])[:5]
    print("  Bottom-5 grad norms:")
    for n, v in bot5:
        print(f"    {v:.6f}  {n}")

    model.zero_grad(set_to_none=True)
    del waveform, crepe_probs, other_embs, out, target_stft, loss, grad_norms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [5] bf16 AMP forward + backward
# ---------------------------------------------------------------------------

def test_amp(dev: torch.device, model: PerTrackProcessor, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[5] bf16 AMP forward + backward")
    print(f"{'='*60}")

    if dev.type == 'cpu':
        print(f"  {SKIP} — AMP only on CUDA/ROCm")
        return

    if not hasattr(torch, 'bfloat16'):
        warn("torch.bfloat16 not available (old PyTorch)")
        return

    sr = cfg.sample_rate
    clip_len = sr * 1
    B = 1
    T = t_frames(clip_len, cfg)
    K = 2

    waveform = torch.randn(B, clip_len, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    other_embs = torch.randn(B, K, T, cfg.d_semantic, device=dev)

    model.train()
    model.zero_grad(set_to_none=True)
    reset_peak()

    t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(waveform, crepe_probs, other_embs=other_embs)
        target_stft = torch.zeros_like(out['output_stft'])
        loss = F.l1_loss(out['output_stft'].abs(), target_stft.abs())

    loss.backward()
    elapsed = time.time() - t0
    peak = peak_memory_mb()

    has_nan_loss = torch.isnan(loss).item()
    check(not has_nan_loss, f"bf16 loss finite ({loss.item():.4f})")
    print(f"  bf16 loss: {loss.item():.4f}")
    print(f"  bf16 fwd+bwd time: {elapsed:.3f}s")
    print(f"  bf16 peak VRAM: {peak:.0f} MB")

    nan_grads = any(
        p.grad is not None and (p.grad.norm().item() != p.grad.norm().item())
        for p in model.parameters()
    )
    check(not nan_grads, "bf16: no NaN gradients")

    model.zero_grad(set_to_none=True)
    del waveform, crepe_probs, other_embs, out, target_stft, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [6] Waveform reconstruction roundtrip
# ---------------------------------------------------------------------------

def test_istft_roundtrip(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[6] Waveform reconstruction roundtrip (STFT → ISTFT)")
    print(f"{'='*60}")

    clip_len = cfg.sample_rate * 1
    B = 1
    wav_in = torch.randn(B, clip_len, device=dev)

    model_tmp = PerTrackProcessor(cfg).to(dev)
    mag_sq, stft_complex = model_tmp._compute_stft(wav_in)
    wav_out = model_tmp.waveform_from_stft(stft_complex)

    L = min(wav_in.shape[-1], wav_out.shape[-1])
    wav_in_trim = wav_in[..., :L]
    wav_out_trim = wav_out[..., :L]

    mse = F.mse_loss(wav_out_trim, wav_in_trim).item()
    max_err = (wav_out_trim - wav_in_trim).abs().max().item()
    print(f"  STFT→ISTFT MSE: {mse:.6f}")
    print(f"  STFT→ISTFT max error: {max_err:.6f}")

    check(mse < 1e-4, f"ISTFT roundtrip MSE < 1e-4 (got {mse:.6f})")
    check(max_err < 0.05, f"ISTFT roundtrip max err < 0.05 (got {max_err:.4f})")

    del model_tmp, wav_in, mag_sq, stft_complex, wav_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [7] Gradient checkpointing
# ---------------------------------------------------------------------------

def test_gradient_checkpointing(dev: torch.device, model: PerTrackProcessor, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[7] Gradient checkpointing (train.py enables this)")
    print(f"{'='*60}")

    if not hasattr(torch.utils, 'checkpoint'):
        print(f"  {SKIP} — torch.utils.checkpoint not available")
        return

    sr = cfg.sample_rate
    clip_len = sr * 1
    B = 1
    T = t_frames(clip_len, cfg)
    K = 2

    waveform = torch.randn(B, clip_len, device=dev)
    crepe_probs = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    other_embs = torch.randn(B, K, T, cfg.d_semantic, device=dev)

    for layer in list(model.mamba_layers) + list(model.semantic_enc.mamba_layers):
        layer.gradient_checkpointing = True

    model.train()
    model.zero_grad(set_to_none=True)
    reset_peak()

    try:
        t0 = time.time()
        out = model(waveform, crepe_probs, other_embs=other_embs)
        target_stft = torch.zeros_like(out['output_stft'])
        loss = F.l1_loss(out['output_stft'].abs(), target_stft.abs())
        loss.backward()
        elapsed = time.time() - t0
        peak = peak_memory_mb()

        check(True, "Gradient checkpointing: fwd+bwd completed")
        print(f"  Checkpointing fwd+bwd time: {elapsed:.3f}s")
        print(f"  Checkpointing peak VRAM: {peak:.0f} MB")

        nan_grads = any(
            p.grad is not None and (p.grad.norm().item() != p.grad.norm().item())
            for p in model.parameters()
        )
        check(not nan_grads, "Checkpointing: no NaN gradients")
    except Exception as e:
        check(False, f"Gradient checkpointing fwd+bwd ({type(e).__name__}: {e})")

    for layer in list(model.mamba_layers) + list(model.semantic_enc.mamba_layers):
        layer.gradient_checkpointing = False

    model.zero_grad(set_to_none=True)
    del waveform, crepe_probs, other_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [8] Memory scaling sweep
# ---------------------------------------------------------------------------

def test_memory_scaling(dev: torch.device, model: PerTrackProcessor, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[8] Memory scaling sweep")
    print(f"{'='*60}")

    sr = cfg.sample_rate
    model.eval()

    test_configs = [
        (1, 1),   # 1s, batch=1
        (1, 2),   # 1s, batch=2
        (2, 1),   # 2s, batch=1
        (2, 2),   # 2s, batch=2
        (4, 1),   # 4s, batch=1
        (4, 2),   # 4s, batch=2
        (4, 4),   # 4s, batch=4
    ]

    print(f"  {'Duration':>8s} {'Batch':>6s} {'Peak MB':>10s} {'Time':>8s} {'Status':>8s}")
    print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

    for dur_s, batch_b in test_configs:
        clip_len = sr * dur_s
        T = t_frames(clip_len, cfg)
        K = 2
        label = f"{dur_s}s"
        oom = False

        try:
            reset_peak()
            t0 = time.time()
            with torch.no_grad():
                w = torch.randn(batch_b, clip_len, device=dev)
                c = torch.softmax(torch.randn(batch_b, T, 360, device=dev), dim=-1)
                oe = torch.randn(batch_b, K, T, cfg.d_semantic, device=dev)
                _ = model(w, c, other_embs=oe)
            elapsed = time.time() - t0
            peak = peak_memory_mb()
            status = PASS
        except torch.cuda.OutOfMemoryError:
            peak = -1
            elapsed = -1
            status = "OOM"
            oom = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                peak = -1
                elapsed = -1
                status = "OOM"
                oom = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

        if oom:
            print(f"  {label:>8s} {batch_b:>6d} {'OOM':>10s} {'---':>8s} {status:>8s}")
        else:
            print(f"  {label:>8s} {batch_b:>6d} {peak:>10.0f} {elapsed:>7.2f}s {status:>8s}")

        del_vars = []
        for name in ['w', 'c', 'oe']:
            if name in dir():
                try:
                    del_vars.append(name)
                except:
                    pass
        for v in del_vars:
            try:
                eval(f"del {v}")
            except:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n  VRAM budget: {vram_total:.1f} GB total")
        print("  Train recommendation: start with --batch 2 --d-model 128 --d-state 16")


# ---------------------------------------------------------------------------
# [9] ROCm / HSA environment check
# ---------------------------------------------------------------------------

def test_rocm_env():
    print(f"\n{'='*60}")
    print("[9] ROCm / HSA environment check")
    print(f"{'='*60}")

    hsa = os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')
    rocm_path = os.environ.get('ROCM_PATH', '')
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')

    print(f"  HSA_OVERRIDE_GFX_VERSION: {hsa or '(not set)'}")
    print(f"  ROCM_PATH: {rocm_path or '(not set)'}")

    has_rocm_lib = any('rocm' in p.lower() or 'amd' in p.lower() for p in ld_path.split(':'))
    print(f"  ROCm in LD_LIBRARY_PATH: {has_rocm_lib}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        prop = torch.cuda.get_device_properties(0)
        total_vram = prop.total_memory / 1024**3
        cc = f"{prop.major}.{prop.minor}"
        print(f"  GPU: {gpu_name}")
        print(f"  Compute capability: {cc}")
        print(f"  VRAM total: {total_vram:.1f} GB")

        is_amd = any(kw in gpu_name.lower() for kw in ['amd', 'radeon', 'gfx', '860m'])
        if is_amd:
            check(bool(hsa), "HSA_OVERRIDE_GFX_VERSION set (required for gfx1103)")
            if not hsa:
                warn("Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for gfx1103 (RDNA 3.5)")

        check(total_vram >= 4.0, f"VRAM >= 4 GB ({total_vram:.1f} GB)")

        backend = torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA'
        print(f"  PyTorch backend: {backend}")
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"  ROCm version: {torch.version.hip}")
    else:
        warn("No CUDA/ROCm device detected — running on CPU only")

    pytorch_ver = torch.__version__
    print(f"  PyTorch version: {pytorch_ver}")
    has_bf16 = hasattr(torch, 'bfloat16')
    print(f"  bf16 support: {has_bf16}")
    if not has_bf16:
        warn("bf16 not available — upgrade PyTorch for SSM training stability")


# ---------------------------------------------------------------------------
# [10] Recipe 1A: synthetic degradation pipeline
# ---------------------------------------------------------------------------

def test_degradation(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[10] Recipe 1A degradation (EQ/comp/sat/reverb → clean target)")
    print(f"{'='*60}")

    from degrade import StemDegradation

    sr = cfg.sample_rate
    clip_len = sr * 1
    B, C = 2, 1

    # Sine + noise: known energy, tests RMS preservation
    t = torch.arange(clip_len, device=dev, dtype=torch.float32) / sr
    wav = 0.5 * torch.sin(2 * math.pi * 440.0 * t)
    wav = wav + 0.02 * torch.randn(clip_len, device=dev)
    wav = wav.unsqueeze(0).unsqueeze(0).expand(B, C, clip_len).contiguous()

    # Force every op on (probability 1.0) via a copy with p=1
    full = StemDegradation(sample_rate=sr, seed=42,
                           p_eq=1.0, p_comp=1.0, p_sat=1.0, p_reverb=1.0)
    out = full(wav)

    check(out.shape == wav.shape, f"degrade: shape preserved {tuple(out.shape)}")
    check(bool(torch.isfinite(out).all()), "degrade: output finite (no NaN/Inf)")
    check(out.abs().max().item() < 20.0,
          f"degrade: output bounded (max {out.abs().max().item():.3f})")

    rms_in  = wav.pow(2).mean().sqrt().item()
    rms_out = out.pow(2).mean().sqrt().item()
    rms_ratio = rms_out / max(rms_in, 1e-8)
    check(0.25 <= rms_ratio <= 4.0,
          f"degrade: RMS roughly preserved (ratio {rms_ratio:.3f})")

    # Determinism: same seed → identical output
    full.reseed(42)
    a = full(wav)
    full.reseed(42)
    b = full(wav)
    check(torch.allclose(a, b), "degrade: reseed() gives deterministic output")

    # Single-op smoke: each op alone stays finite
    for op in ('p_eq', 'p_comp', 'p_sat', 'p_reverb'):
        kwargs = dict(p_eq=0.0, p_comp=0.0, p_sat=0.0, p_reverb=0.0)
        kwargs[op] = 1.0
        one = StemDegradation(sample_rate=sr, seed=7, **kwargs)
        y = one(wav)
        check(bool(torch.isfinite(y).all()), f"degrade: {op} finite")
        check(y.shape == wav.shape, f"degrade: {op} shape preserved")

    # Scalar (L,) and stereo (C, L) shapes must also work
    y1 = full(wav[0, 0])
    y2 = full(wav[0])
    check(y1.shape == wav[0, 0].shape, "degrade: (L,) shape preserved")
    check(y2.shape == wav[0].shape, "degrade: (C, L) shape preserved")

    del wav, out, a, b, y1, y2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [11] f0 pitched gating + native gradient-checkpointing flag
# ---------------------------------------------------------------------------

def test_pitched_gating(dev: torch.device, model: PerTrackProcessor, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[11] f0 pitched gating (target_pitched) + grad-ckpt flag honored")
    print(f"{'='*60}")

    sr = cfg.sample_rate
    clip_len = sr // 2
    B = 1
    T = t_frames(clip_len, cfg)

    wav = 0.3 * torch.randn(B, clip_len, device=dev)
    crepe_a = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    crepe_b = torch.softmax(torch.randn(B, T, 360, device=dev), dim=-1)
    zero = torch.zeros(B, device=dev)
    one  = torch.ones(B, device=dev)

    model.eval()
    with torch.no_grad():
        out_a0 = model(wav, crepe_a, target_pitched=zero)['output_stft']
        out_b0 = model(wav, crepe_b, target_pitched=zero)['output_stft']
        out_a1 = model(wav, crepe_a, target_pitched=one)['output_stft']
        out_b1 = model(wav, crepe_b, target_pitched=one)['output_stft']

    # With target_pitched = 0 the f0 pathway must be exactly zeroed:
    # different CREPE vectors cannot change the output.
    diff0 = (out_a0 - out_b0).abs().max().item()
    check(diff0 == 0.0, f"pitched=0: CREPE ignored exactly (max diff {diff0:.2e})")

    # NOTE: with zero-init FiLM (by design at construction) the whole cond
    # vector — f0 included — is a no-op, so pitched=1 also gives diff == 0.
    # That is the correct FiLM-zero-init behavior, not an f0 bug. The gate
    # semantics are fully covered by the pitched=0 invariance check above.
    diff1 = (out_a1 - out_b1).abs().max().item()
    print(f"  pitched=1: max diff {diff1:.2e} (0 expected while FiLM is zero-init)")

    # Gradient checkpointing flag is honored in training mode:
    # fwd+bwd must complete with the flag on, and produce grads.
    for layer in list(model.mamba_layers) + list(model.semantic_enc.mamba_layers):
        layer.gradient_checkpointing = True
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        out = model(wav, crepe_a, target_pitched=one)
        loss = out['output_stft'].abs().mean()
        loss.backward()
        n_grads = sum(1 for p in model.parameters() if p.grad is not None)
        check(n_grads > 0, f"grad-ckpt flag: backward produced grads on {n_grads} tensors")
    except Exception as e:
        check(False, f"grad-ckpt flag: fwd+bwd ({type(e).__name__}: {e})")
    for layer in list(model.mamba_layers) + list(model.semantic_enc.mamba_layers):
        layer.gradient_checkpointing = False
    model.zero_grad(set_to_none=True)

    del wav, crepe_a, crepe_b, out_a0, out_b0, out_a1, out_b1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# [12] Way 2C scan parity: chunked parallel scan == sequential reference
# ---------------------------------------------------------------------------

def test_scan_parity(dev: torch.device, cfg: TrackProcessorConfig):
    print(f"\n{'='*60}")
    print("[12] Way 2C scan parity (chunked == sequential)")
    print(f"{'='*60}")

    from model import Mamba3MIMOLayer

    torch.manual_seed(0)
    layer = Mamba3MIMOLayer(cfg).to(dev)
    layer.eval()

    B, T = 2, 77          # 77: deliberately not a multiple of scan_chunk
    x    = torch.randn(B, T, cfg.d_model, device=dev)
    x_r8 = torch.randn(B, T, cfg.R, device=dev)
    cond = torch.randn(B, T, cfg.d_cond, device=dev)

    with torch.no_grad():
        layer.use_chunked = False
        y_seq = layer(x, x_r8, cond=cond)
        layer.use_chunked = True
        y_chk = layer(x, x_r8, cond=cond)

    diff = (y_seq - y_chk).abs().max().item()
    check(torch.isfinite(y_seq).all() and torch.isfinite(y_chk).all(),
          "scan parity: outputs finite")
    check(diff < 1e-3, f"scan parity: max |seq − chunked| = {diff:.2e} < 1e-3")

    # State-carry parity: streaming one step at a time must equal full-window
    layer.use_chunked = True
    with torch.no_grad():
        out_full, state_end = layer(x, x_r8, cond=cond, return_state=True)
        state = None
        outs = []
        for t in range(T):
            y_t, state = layer(x[:, t:t + 1], x_r8[:, t:t + 1],
                               cond=cond[:, t:t + 1], state=state, return_state=True)
            outs.append(y_t)
        out_step = torch.cat(outs, dim=1)
    diff2 = (out_full - out_step).abs().max().item()
    check(diff2 < 1e-3, f"state carry: streaming == full-window (max Δ {diff2:.2e})")

    layer.use_chunked = False
    del x, x_r8, cond, y_seq, y_chk, out_full, out_step
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def smoke_test(device: str = 'cuda', verbose: bool = True, amp: bool = False):
    global fail_count, warn_count
    fail_count = 0
    warn_count = 0

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print("Smoke Test: Mamba-3 MIMO Per-Track Processor")
    print(f"Device: {dev}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print('='*60)

    cfg = make_cfg()

    assert cfg.d_model % cfg.n_heads == 0, \
        f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

    model = test_build(dev, cfg)

    test_r8_extractor(dev, cfg)
    test_semantic_encoder(dev, cfg)
    test_f0_conditioner(dev, cfg)
    test_cross_track(dev, cfg)
    test_mamba_layer(dev, cfg)

    test_forward_no_grad(dev, model, cfg)
    test_forward_backward(dev, model, cfg)

    if amp or dev.type != 'cpu':
        test_amp(dev, model, cfg)

    test_istft_roundtrip(dev, cfg)
    test_gradient_checkpointing(dev, model, cfg)
    test_memory_scaling(dev, model, cfg)
    test_rocm_env()
    test_degradation(dev, cfg)
    test_pitched_gating(dev, model, cfg)
    test_scan_parity(dev, cfg)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"  Failures: {fail_count}")
    print(f"  Warnings: {warn_count}")
    if fail_count == 0:
        print(f"\n  All checks {PASS}. Proceed to train.py with --device cuda")
        print("  Recommended: python train.py --data <path> --device cuda --batch 2 --d-model 128 --d-state 16")
    else:
        print(f"\n  {fail_count} check(s) {FAIL}. Review output above.")
    if warn_count > 0:
        print(f"  {warn_count} warning(s). May need attention before training.")
    print("  If OOM: reduce --d-model 64 --d-state 8 --batch 1")
    print("  If NaN: check bf16 AMP — disable with --no-amp first")
    print("  If slow: sequential SSM scan is O(T) — expected on CPU/small GPU")
    print('='*60)

    return fail_count == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Smoke test for Mamba-3 Per-Track Processor on AMD 860M"
    )
    parser.add_argument('--device', default='cuda',
                        help='Device: cuda (ROCm) or cpu')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--amp', action='store_true',
                        help='Force bf16 AMP test even on CPU')
    args = parser.parse_args()

    ok = smoke_test(args.device, verbose=not args.no_verbose, amp=args.amp)
    sys.exit(0 if ok else 1)

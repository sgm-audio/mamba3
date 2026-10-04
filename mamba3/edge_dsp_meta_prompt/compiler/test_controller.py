"""
Regression checks for the Mamba-3 edge controller (compiler/train_controller.py).

Runnable two ways (no test framework is required by this repo):
    python test_controller.py          # plain assertions, exit code 1 on failure
    pytest test_controller.py          # if pytest is installed

Covers the two defects found in the 2026-10 review:
  1. `Mamba3StepLayer.step` broadcast bug — `dt.unsqueeze(-1) * a_cont`
     produced a (B, H, H) tensor and crashed `forward_step` with a size
     mismatch for every batch size > 1 (i.e. the default `--batch 16`).
  2. Hidden-state packing mismatch — `forward_step` packed the per-layer
     state as [re_h0 | im_h0 | re_h1 | im_h1] while `_unflat_hidden` read it
     back as [re_h0 | re_h1 | im_h0 | im_h1], so the recurrent state was
     scrambled on every tick that crossed the pack/unpack boundary.
"""

from __future__ import annotations

import sys

import torch

from train_controller import (
    ControllerConfig,
    GoldenPolicy,
    Mamba3Controller,
    PARAM_COUNT,
    synthetic_batch,
)


def test_forward_step_batched() -> None:
    """forward_step must work for B > 1 (regression for the broadcast crash)."""
    torch.manual_seed(0)
    model = Mamba3Controller().eval()
    c = model.cfg
    for B in (1, 2, 16):
        obs = torch.randn(B, 1, c.obs_dim)
        sess = torch.zeros(B, c.sess_dim)
        h = torch.zeros(B, c.n_layers, c.hidden_dim)
        params, h_out = model.forward_step(obs, sess, h)
        assert params.shape == (B, PARAM_COUNT), params.shape
        assert h_out.shape == (B, c.n_layers, c.hidden_dim), h_out.shape
        assert torch.isfinite(params).all() and torch.isfinite(h_out).all()
        assert (params.abs() <= 1.0).all(), "param head must be tanh-bounded"


def test_hidden_state_roundtrip() -> None:
    """_flat_hidden ∘ _unflat_hidden must be the identity (both directions)."""
    torch.manual_seed(0)
    model = Mamba3Controller()
    c = model.cfg
    B = 3
    h = torch.complex(torch.randn(B, c.n_layers, c.n_heads, c.d_state),
                      torch.randn(B, c.n_layers, c.n_heads, c.d_state))
    flat = model._flat_hidden(h)
    assert flat.shape == (B, c.n_layers, c.hidden_dim)
    back = model._unflat_hidden(flat)
    assert torch.equal(back, h), "flat→unflat is not the identity"

    flat2 = torch.randn(B, c.n_layers, c.hidden_dim)
    assert torch.equal(model._flat_hidden(model._unflat_hidden(flat2)), flat2)


def test_step_state_matches_layout() -> None:
    """The state emitted by forward_step must decode with _unflat_hidden to the
    exact complex state the layers computed (regression for the layout bug)."""
    torch.manual_seed(0)
    model = Mamba3Controller().eval()
    c = model.cfg
    B = 2
    obs = torch.randn(B, 1, c.obs_dim)
    sess = torch.zeros(B, c.sess_dim)
    h0 = model._flat_hidden(model.init_hidden(B))

    # Reference: run the layers by hand and keep the complex states.
    u = torch.nn.functional.silu(model.in_proj(obs[:, 0, :]) + model.sess_bias(sess))
    h_states = model._unflat_hidden(h0)
    ref = []
    for l, layer in enumerate(model.layers):
        u, h_next = layer.step(u, sess, h_states[:, l])
        ref.append(h_next)
    ref = torch.stack(ref, dim=1)                           # (B, L, H, N) complex

    _, h_out = model.forward_step(obs, sess, h0)
    assert torch.allclose(model._unflat_hidden(h_out), ref, atol=1e-6)


def test_sequence_equals_stepwise() -> None:
    """forward() over T steps == T chained forward_step() calls."""
    torch.manual_seed(0)
    model = Mamba3Controller().eval()
    B, T = 2, 7
    obs, sess = synthetic_batch(B, T, torch.device('cpu'))
    with torch.no_grad():
        seq_out, seq_h = model(obs, sess)
        h = model._flat_hidden(model.init_hidden(B))
        outs = []
        for t in range(T):
            p, h = model.forward_step(obs[:, t:t + 1, :], sess, h)
            outs.append(p)
        step_out = torch.stack(outs, dim=1)
    assert torch.allclose(seq_out, step_out, atol=1e-6)
    assert torch.allclose(seq_h, h, atol=1e-6)


def test_recurrence_carries_state() -> None:
    """A non-zero carried state must change the next output (state is live)."""
    torch.manual_seed(0)
    model = Mamba3Controller().eval()
    c = model.cfg
    obs = torch.randn(1, 1, c.obs_dim)
    sess = torch.zeros(1, c.sess_dim)
    with torch.no_grad():
        p0, h1 = model.forward_step(obs, sess, torch.zeros(1, c.n_layers, c.hidden_dim))
        p1, _ = model.forward_step(obs, sess, h1)
    assert not torch.allclose(p0, p1), "carried hidden state had no effect"


def test_step_real_matches_complex_reference() -> None:
    """step_real (export-friendly) must equal the textbook complex bilinear
    recurrence  h' = (1+½dtA)/(1-½dtA)·h + dt/(1-½dtA)·b  it replaced."""
    torch.manual_seed(0)
    layer = Mamba3Controller().layers[0].eval()
    c = layer.cfg
    B = 4
    u = torch.randn(B, c.d_model)
    sess = torch.randn(B, c.sess_dim) * 0.1
    h = torch.complex(torch.randn(B, c.n_heads, c.d_state),
                      torch.randn(B, c.n_heads, c.d_state))
    with torch.no_grad():
        # un-zero the FiLM so the dt path is exercised too
        layer.dt_film.weight.normal_(0, 0.1)
        layer.film.weight.normal_(0, 0.1)
        out, h_next = layer.step(u, sess, h)

        # complex reference (the pre-review formulation, bug-free broadcast)
        gamma, beta = layer.film(sess).chunk(2, dim=-1)
        uu = u * (1.0 + gamma) + beta
        dg, db = layer.dt_film(sess).chunk(2, dim=-1)
        dt = torch.nn.functional.softplus(layer.dt_proj(uu) * (1.0 + dg) + db + layer.dt_bias)
        dt = dt.clamp(c.dt_min, c.dt_max)
        a = torch.complex(-torch.exp(layer.log_a_real), layer.a_imag)
        half = dt * a / 2.0
        A_d = (1.0 + half) / (1.0 - half)
        B_d = dt.to(torch.complex64) / (1.0 - half)
        br, bi = layer.B_proj(uu).chunk(2, dim=-1)
        b = torch.complex(br, bi).reshape(B, c.n_heads, c.d_state)
        h_ref = A_d.unsqueeze(-1) * h + B_d.unsqueeze(-1) * b
        y = layer.C_proj(torch.cat([h_ref.real.reshape(B, -1), h_ref.imag.reshape(B, -1)], -1))
        out_ref = layer.norm(layer.out_proj(y) + u)
    assert torch.allclose(h_next, h_ref, atol=1e-5), (h_next - h_ref).abs().max()
    assert torch.allclose(out, out_ref, atol=1e-5)


def test_golden_policy_shapes() -> None:
    obs, sess = synthetic_batch(4, 5, torch.device('cpu'))
    tgt = GoldenPolicy()(obs, sess)
    assert tgt.shape == (4, 5, PARAM_COUNT)
    assert (tgt.abs() <= 1.0 + 1e-6).all()


def test_onnx_export_roundtrip(tmp_path=None) -> None:
    """export_onnx → onnxruntime must reproduce forward_step over a recurrent
    rollout (skipped when onnx/onnxruntime are not installed)."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("    (skipped: onnx/onnxruntime not installed)")
        return
    import tempfile
    from pathlib import Path
    from train_controller import export_onnx

    torch.manual_seed(0)
    model = Mamba3Controller().eval()
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / 'controller.onnx')
        export_onnx(model, path)
        assert not Path(path + '.data').exists(), \
            "export must be a single self-contained file (plugin include_bytes!)"
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        assert [i.name for i in sess.get_inputs()] == \
            ['observer_features', 'session_state', 'hidden_state_in']
        assert [o.name for o in sess.get_outputs()] == ['dsp_parameters', 'hidden_state_out']

        obs, se = synthetic_batch(1, 12, torch.device('cpu'))
        h_np = np.zeros((1, 3, 64), np.float32)
        h_t = torch.zeros(1, 3, 64)
        with torch.no_grad():
            for t in range(obs.shape[1]):
                p_t, h_t = model.forward_step(obs[:, t:t + 1], se, h_t)
                p_np, h_np = sess.run(None, {
                    'observer_features': obs[:, t:t + 1].numpy(),
                    'session_state': se.numpy(),
                    'hidden_state_in': h_np,
                })
                assert np.allclose(p_np, p_t.numpy(), atol=1e-5)
                assert np.allclose(h_np, h_t.numpy(), atol=1e-5)


def test_config_defaults_consistent() -> None:
    c = ControllerConfig()
    assert c.hidden_dim == c.n_heads * c.d_state * 2


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:  # noqa: BLE001 — report every failure, keep going
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

"""
INT8 dynamic quantization + signature gate for the edge controller ONNX (Way 5A).

Reproduces the artifact pipeline that produced `mamba3_controller_quant.onnx`
(onnxruntime.quantization `quantize_dynamic` → DynamicQuantizeLinear /
MatMulInteger) but adds a verification gate the original pipeline lacked:

    The exported graph MUST expose exactly the documented 3 inputs / 2 outputs.
    The legacy export leaked `onnx::Mul_3`, `onnx::Mul_4`, `onnx::Mul_3168`,
    `onnx::Mul_4689` (per-layer hidden states from a tuple-unpacking bug);
    this module refuses such graphs so the Rust worker's ort::Session can
    never again fail at run() with a missing-input error.

CLI:
    python quantize.py controller.onnx mamba3_controller_quant.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_INPUTS = ['observer_features', 'session_state', 'hidden_state_in']
EXPECTED_OUTPUTS = ['dsp_parameters', 'hidden_state_out']


def quantize_dynamic(in_path: str, out_path: str, verify: bool = True) -> str:
    try:
        from onnxruntime.quantization import quantize_dynamic as _q, QuantType
    except ImportError as e:
        raise SystemExit(
            "onnxruntime is required for quantization.\n"
            "  pip install onnx onnxruntime\n"
            f"({e})"
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _q(
        model_input=in_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    print(f"Quantized (INT8 dynamic) → {out_path}")

    if verify:
        verify_signature(out_path)
        _parity_check(in_path, out_path)
    return out_path


def verify_signature(path: str) -> None:
    """Assert the 3-input / 2-output contract documented in docs/edge-controller-reference.md."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit(f"onnxruntime is required for verification ({e})")

    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]

    ok = True
    if ins != EXPECTED_INPUTS:
        ok = False
        print(f"  ✗ INPUT signature mismatch: got {ins}, want {EXPECTED_INPUTS}")
        if any(n.startswith('onnx::') for n in ins):
            print("    → legacy tuple-unpacking export bug (leaked hidden states).")
            print("      Re-export with compiler/train_controller.py (Way 5A).")
    if outs != EXPECTED_OUTPUTS:
        ok = False
        print(f"  ✗ OUTPUT signature mismatch: got {outs}, want {EXPECTED_OUTPUTS}")
    if not ok:
        raise SystemExit(f"Signature gate FAILED for {path}")

    shapes = {i.name: [d for d in i.shape] for i in sess.get_inputs()}
    print(f"  ✓ signature OK: inputs={ins} outputs={outs}")
    print(f"    shapes: {shapes}")


def _parity_check(fp32_path: str, int8_path: str, tol: float = 0.25) -> None:
    """Run both graphs on the same zero/one inputs; INT8 drift must be bounded."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        return

    s32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    s8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])

    feed = {
        'observer_features': np.zeros((1, 1, 12), dtype=np.float32),
        'session_state': np.zeros((1, 16), dtype=np.float32),
        'hidden_state_in': np.zeros((1, 3, 64), dtype=np.float32),
    }
    # Only feed inputs the sessions actually declare
    feed32 = {k: v for k, v in feed.items() if any(i.name == k for i in s32.get_inputs())}
    feed8 = {k: v for k, v in feed.items() if any(i.name == k for i in s8.get_inputs())}
    if set(feed32) != set(EXPECTED_INPUTS) or set(feed8) != set(EXPECTED_INPUTS):
        print("  (parity skipped: unexpected input names)")
        return

    y32 = s32.run(['dsp_parameters'], feed32)[0]
    y8 = s8.run(['dsp_parameters'], feed8)[0]
    drift = float(abs(y32 - y8).max())
    status = "✓" if drift <= tol else "✗"
    print(f"  {status} INT8 parity: max |fp32 − int8| = {drift:.4f} (tol {tol})")
    if drift > tol:
        raise SystemExit("Quantization drift beyond tolerance — inspect the model.")


def main():
    ap = argparse.ArgumentParser(description="INT8 dynamic quantization + signature gate")
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()
    quantize_dynamic(args.input, args.output, verify=not args.no_verify)


if __name__ == '__main__':
    main()

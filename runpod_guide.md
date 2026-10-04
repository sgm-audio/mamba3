# RunPod Training Guide

## 1. SSH into your pod

Copy the SSH command from the RunPod pod details page:
```bash
ssh root@<RUNPOD_SSH_HOST> -p <RUNPOD_SSH_PORT> -i ~/.ssh/<your_key>
```

## 2. Upload code (if not git-cloned)

```bash
# Option A: git clone
git clone https://github.com/sgm-audio/mamba3.git /workspace/mamba3_repo
ln -s /workspace/mamba3_repo/mamba3 /workspace/mamba3   # train.py lives in the mamba3/ sub-dir; keep later steps' paths valid
cd /workspace/mamba3

# Option B: scp from local machine (run on YOUR machine, not the pod)
scp -P <PORT> -i <key> -r C:\Dev\mamba3\mamba3 root@<host>:/workspace/mamba3
```

## 3. Install dependencies

```bash
apt update && apt install -y tmux && pip install torch torchaudio auraloss torchcrepe soundfile numpy
```

## 4. Upload dataset (if not already on pod)

```bash
# From YOUR machine (run locally, not on the pod)
scp -P <PORT> -i <key> -r C:\Dev\mamba3\datasets\musdb18hq root@<host>:/workspace/datasets/musdb18hq
```

## 5. Start training

```bash
cd /workspace/mamba3
tmux new -s train
```

### H100 SXM (80GB) — recommended
```bash
python train.py --device cuda --batch 128
```

### H100 SXM — max speed (no gradient checkpointing)
```bash
python train.py --device cuda --batch 128 --no-grad-ckpt
```

### H100 SXM — larger effective batch
```bash
python train.py --device cuda --batch 256 --no-grad-ckpt
```

### RTX PRO 6000 (96GB)
```bash
python train.py --device cuda --batch 128
```

### L4 (24GB)
```bash
python train.py --device cuda --batch 32
```

Detach tmux: `Ctrl+B`, then `D`

## 6. Resume after SSH disconnect or stop

```bash
# Reconnect
ssh root@<RUNPOD_SSH_HOST> -p <RUNPOD_SSH_PORT> -i ~/.ssh/<your_key>

# Reattach training session
tmux attach -t train
```

## 7. Resume after Ctrl+C or crash

```bash
tmux new -s train
cd /workspace/mamba3
python train.py --device cuda --batch 128 --resume checkpoints/interrupt_XXXXXX.pt
```

Check available checkpoints:
```bash
ls -la checkpoints/
```

## 8. Graceful shutdown

```bash
# Ctrl+C in the training tmux session
# Saves interrupt checkpoint, then exits cleanly
# Resume with --resume as shown above
```

## Notes

- Checkpoints save every 1000 steps automatically
- `--resume` restores model, optimizer, scheduler, and step counter
- Dataset expected at `/workspace/datasets/musdb18hq/train/`
- Checkpoints save to `./checkpoints/` in the working directory

---

## Recipe 1A note (training objective)

`train.py` now trains a **restorer**: each batch feeds a synthetically degraded
stem (random EQ resonances/cuts, over-compression, saturation, room reverb —
see `mamba3/degrade.py`) into the model and scores it against the CLEAN stem.
Validation uses a seeded deterministic degradation for stable early stopping.
To ablate the old behavior pass `--no-degrade`.

## Edge controller: distill → quantize → embed (Way 5A)

After (or alongside) PerTrackProcessor training, the real-time plugin's
controller can be produced entirely on the pod (CPU is enough — the controller
is tiny):

```bash
# 1. Distill the GoldenPolicy teacher into the Mamba-3 controller and export
#    the clean single-step ONNX (3 inputs / 2 outputs — no unrolled window):
python mamba3/edge_dsp_meta_prompt/compiler/train_controller.py \
    --steps 2000 --device cuda \
    --export controller.onnx --quantized mamba3_controller_quant.onnx

# 2. (Per customer) fit the 8-D tone profile — Way 6C, weights frozen:
python mamba3/edge_dsp_meta_prompt/compiler/fit_profile.py \
    --data client_features/ --profile-out profile.npy

# 3. Copy the quantized model back into the plugin and rebuild:
cp mamba3_controller_quant.onnx mamba3/edge_dsp_meta_prompt/mamba3_controller_quant.onnx
# on a machine with Rust — NOTE: the workspace has no `xtask` crate yet, so
# `cargo xtask bundle` is not available; `cargo build --release -p edge_dsp_plugin`
# builds the cdylib. <!-- STATUS: research --> (CLAP bundling step still to be set up)
```

`compiler/quantize.py` enforces the 3-input/2-output signature gate and an
fp32↔int8 parity check, so a bad export can never reach the plugin worker.

The export needs `onnx`, `onnxruntime` **and** `onnxscript` (torch ≥ 2.9 imports
it even for the TorchScript exporter): `pip install -r mamba3/edge_dsp_meta_prompt/compiler/requirements.txt`.
Run `python mamba3/edge_dsp_meta_prompt/compiler/test_controller.py` first — it
checks the recurrence, the packed-state layout and an ONNX round-trip in a few seconds.

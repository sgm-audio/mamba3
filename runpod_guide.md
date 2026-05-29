# RunPod Training Guide

## 1. SSH into your pod

Copy the SSH command from the RunPod pod details page:
```bash
ssh root@<RUNPOD_SSH_HOST> -p <RUNPOD_SSH_PORT> -i ~/.ssh/<your_key>
```

## 2. Upload code (if not git-cloned)

```bash
# Option A: git clone
git clone https://github.com/you/mamba3.git /workspace/mamba3 && cd /workspace/mamba3

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

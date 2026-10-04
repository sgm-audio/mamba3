# Mamba-3 Native Ubuntu Training Guide

> **Scripts:** `setup_env.sh` (venv + PyTorch; `TORCH_INDEX=` selects ROCm/CUDA/CPU wheels),
> `run_smoke_test.sh`, `extract_f0.sh` (`DATASET_ROOT=` override) and `run_train.sh`
> (`DATASET_DIR=` override, or pass `--data`). The defaults below point at one specific machine's
> mounts — set the env vars for yours.

---

## 1. Native Workspace & Dataset Paths

All native helper scripts are configured to use your mounted drives directly:
* **Active Project Workspace**: `/home/sgm/Desktop/mamba3`
* **Original Bazzite Backup**: `/media/sgm/bazzite/home/SGM/mamba3`
* **Raw MUSDB18-HQ Dataset**: `/media/sgm/bazzite/home/SGM/datasets/musdb18hq`

---

## 2. Step-by-Step Training Workflow

We have automated the environment setup, testing, offline extraction, and training loop. Follow these four simple commands:

### Step 2.1: Native Environment Setup
Run this once to create the virtual environment (`.venv`) and install ROCm-compatible PyTorch, torchaudio, and secondary training libraries natively:
```bash
chmod +x setup_env.sh run_smoke_test.sh extract_f0.sh run_train.sh
./setup_env.sh
```

### Step 2.2: Pre-extract CREPE Pitch Tracks (CRITICAL)
Phases 2 and 3 of the Mamba-3 separator depend on pre-extracted frame-level F0 frequency projections. Run this step to process the `train/` and `test/` splits under `$DATASET_ROOT` natively on your GPU (`--model full`; pass `--model tiny` for a faster, less accurate pass):
```bash
./extract_f0.sh
```
*Note: This script automatically extracts pitch contours for the `train` and `test` sub-directories and generates `.crepe.npy` + `.pitched.npy` arrays directly next to the audio stems. It supports resuming if interrupted.*

### Step 2.3: Native GPU Sanity Check
Before launching multi-hour training runs, execute the full model verification suite on your GPU to assert that the mathematical layers, complex pole calculations, and gradient checkpointing execute without errors:
```bash
./run_smoke_test.sh
```
*Expected Output: `All checks OK. Proceed to train.py`.*

### Step 2.4: Launch or Resume Training
Start training natively. The runner automatically detects if checkpoints (standard or interrupt) exist in `checkpoints/` and resumes from the latest step:
```bash
./run_train.sh
```
*Tip: To override training settings (e.g. increase batch size, specify a specific checkpoint), pass arguments directly:*
```bash
./run_train.sh --batch 4 --resume checkpoints/checkpoint_005000.pt
```

---

## 3. Real-Time Resource & Temperature Monitoring

Open separate terminal windows on your native Ubuntu desktop to monitor the training footprint without interrupting execution:

### 3.1 Live GPU Stats (Utilization, VRAM, and Power)
Since ROCm libraries are natively visible on the host, monitor live GPU temperature, power draw, and VRAM occupancy:
```bash
watch -n 2 "cat /sys/class/drm/card0/device/hwmon/hwmon*/temp1_input 2>/dev/null | awk '{print \$1/1000 \"°C\"}' && cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_average 2>/dev/null | awk '{print \$1/1000000 \"W\"}'"
```
Or graphical GPU shader load via:
```bash
sudo apt install -y radeontop
radeontop
```

### 3.2 Live CPU & RAM Footprint
Verify host CPU core allocation, clock speed, and RAM usage:
```bash
htop
```

### 3.3 Tail the Active Training Logs
```bash
tail -f train.log
```

---

## 4. Thermal Thresholds (16GB UMA Hardware)

Sustained training on laptops demands vigilance regarding temperatures to prevent silicon throttling or emergency power shutdowns:

| Temperature | Core Status | Recommended Action |
| :--- | :--- | :--- |
| **< 80°C** | Healthy | Safe to proceed at current batch sizes. |
| **80°C – 90°C** | Warm | Verify laptop base is elevated for increased airflow. |
| **90°C – 95°C** | Throttling | Thermal throttling may occur. Reduce batch size or limit frame length. |
| **> 95°C** | Emergency | Stop training immediately (`Ctrl+C` once for graceful checkpoint). |

---

## 5. Troubleshooting (Native ROCm / Ubuntu)

### "HIP error: invalid device function"
* **Reason**: ROCm environment variables were lost or not loaded.
* **Solution**: Always launch using our `./run_train.sh` or `./run_smoke_test.sh` scripts, which force-export `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `PYTORCH_ROCM_ARCH=gfx1100`, and `HSA_ENABLE_SDMA=0` before running Python (these are the values the scripts actually set; they match `TRAINING.md` §1.3).

### iGPU Not Visible to PyTorch (`CUDA/ROCm Available: False` / running CPU-only)
* **Reason**: On AMD APUs (Ryzen laptops with integrated Radeon 860M graphics), the KFD (Kernel Fusion Driver) interface must be enabled, display memory pinning conflicts resolved, IOMMU isolated, and system user groups verified.
* **Solution**:
  1. **User Group Access**: Ensure you are in the `video` and `render` groups by running `groups`. If not, add your user and restart your Ubuntu session:
     ```bash
     sudo usermod -aG video,render $USER
     ```
  2. **AMDGPU Configuration**: Resolve display memory pinning conflicts (which prevent KFD from mapping iGPU VRAM pages) by adding the following option:
     Create or edit `/etc/modprobe.d/amdgpu.conf` and append:
     ```text
     options amdgpu sg_display=0
     ```
  3. **IOMMU Passthrough**: Integrated GPUs require IOMMU to be set to passthrough mode on Linux to allow KFD address translation:
     Edit `/etc/default/grub` and add `iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT`:
     ```text
     GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.sg_display=0 iommu=pt"
     ```
     Then rebuild your initramfs, update grub, and reboot your laptop:
     ```bash
     sudo update-initramfs -u
     sudo update-grub
     sudo reboot
     ```
  4. **BIOS UMA Buffer**: Confirm that the iGPU UMA frame buffer is set to the desired dedicated size (16GB) in your laptop's BIOS settings to give PyTorch ample room.

### `radeontop` Doesn't See the GPU
* **Reason**: On laptops with hybrid graphics (integrated AMD APU + CPU, or discrete cards), `radeontop` requires direct DRM read privileges and needs to be targeted directly to the AMD GPU bus ID.
* **Solution**:
  1. **Target the Bus ID**: Run `lspci | grep -i "vga\|3d"` to find your AMD GPU bus ID (e.g., `03:00.0`, which corresponds to bus `3`).
  2. **Elevate Privileges**: Run `radeontop` with root privileges and specify the targeted bus (replace `3` with your actual bus ID):
     ```bash
     sudo radeontop -b 3
     ```
     Alternatively, let `radeontop` auto-detect the active card:
     ```bash
     sudo radeontop -d -
     ```

### Out of Memory (OOM) on 16GB UMA
* **Reason**: Batch size or model layer width is too large for the current UMA buffer.
* **Solution**: The default batch size in `run_train.sh` is optimized at `2`. If you still experience OOMs, reduce it to `1` by executing:
  ```bash
  ./run_train.sh --batch 1
  ```
  Alternatively, reduce sequence duration inside `train.py`'s `StemDataset` constructor (edit `clip_seconds=4.0` to `2.0`).

### NaN/Inf Loss or Gradients
* **Reason**: Numerical instability in half-precision (bf16) pole calculations.
* **Solution**: Stop training and relaunch with `AMP` disabled:
  ```bash
  ./run_train.sh --no-amp
  ```
  This forces full `float32` calculation across all layers, ensuring maximum precision and numerical stability.

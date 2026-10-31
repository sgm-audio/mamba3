# Mamba-3 Training Workflow — Windows 11 Edition

> **Project:** `C:\Users\<you>\mamba3\` • **Venv:** `C:\Users\<you>\mamba3\.venv\`
> **GPU:** AMD Radeon 860M (gfx1152, Strix Point) • **ROCm 7.13.0 Preview on Windows 11**
> **PyTorch 2.11.0 + ROCm 7.13.0**

---

## ⚠ Critical Pre-Reading — Known Risks

### Risk 1: Active GPU hang bug on your chip
ROCm GitHub issue #5844 (open as of Jan 2026) reports `MES:0x82` GPU hangs on the **exact** Ryzen AI 7 350 / Radeon 860M / gfx1152 hardware. May or may not hit your run. **Mitigation:** save checkpoints often (`--ckpt-every 500`). If you see hangs, your fallback is dual-boot back to Linux/Bazzite.

### Risk 2: Smart App Control must be OFF
If you see "Application Control Policy blocking DLL loading", Smart App Control is interfering. **Disabling it is a one-way trip.** Once off, you cannot re-enable Smart App Control without reinstalling Windows. Check via Settings → Privacy & Security → Windows Security → App & Browser Control → Smart App Control.

### Risk 3: Windows ROCm is PyTorch-only preview
The full ROCm stack (rocBLAS tooling, rocprofiler, ROCm-aware Triton, etc.) is Linux-only. You get PyTorch with ROCm acceleration and that's it. Our code doesn't need anything else, but if you hit something exotic, that's why.

### Risk 4: XDNA NPU is NOT in play for training
The Ryzen AI NPU runs only ONNX/DirectML inference paths. Mamba-3 with custom `autograd.Function` scan **cannot train on the NPU.** Training uses the **iGPU (Radeon 860M)** via ROCm. NPU is post-training inference deployment concern.

---

## 1. Sleep Prevention

### Disable sleep via PowerShell (run as Administrator):
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /change disk-timeout-ac 0
powercfg /change disk-timeout-dc 0
powercfg /change monitor-timeout-ac 0
```

### Re-enable sleep after training (run as Administrator):
```powershell
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 15
powercfg /change hibernate-timeout-ac 60
powercfg /change hibernate-timeout-dc 30
powercfg /change disk-timeout-ac 20
powercfg /change disk-timeout-dc 10
powercfg /change monitor-timeout-ac 10
```

### Additional belt-and-suspenders (one-time GUI clicks):
- Settings → System → Power & battery → Screen and sleep → set all timeouts to "Never"
- Settings → System → Power & battery → Lid, power and sleep button controls → "When I close the lid" → "Do nothing" (on AC at minimum)
- Settings → Personalization → Lock screen → Screen saver settings → "(None)"

---

## 2. One-Time Environment Setup

Skip after first run.

### 2.1 Install AMD Adrenalin Driver 26.x (or newer)
Download from [amd.com/en/support](https://www.amd.com/en/support). This is the GPU driver that ROCm-on-Windows piggybacks on. Must be installed and rebooted before next step.

### 2.2 Install Python 3.12 (3.11 or 3.13 also work)
Download from [python.org/downloads](https://www.python.org/downloads/). During install:
- ✅ Check "Add Python to PATH"
- ✅ Check "Install for all users" (recommended if running PowerShell as admin)
- ❌ Do NOT use Microsoft Store Python — it has sandboxing issues with native CUDA-like APIs

Verify:
```powershell
py -3.12 --version
```

### 2.3 Install Visual C++ Redistributables
Per ROCm Windows docs: "If you encounter errors related to missing .dll libraries, install Visual C++ 2015-2022 Redistributables."
Download from Microsoft: [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### 2.4 Verify Smart App Control is OFF
Settings → Privacy & Security → Windows Security → App & Browser Control → Smart App Control → set to **Off**.
**Once off, cannot be re-enabled without Windows reinstall.** Make peace with that before proceeding.

---

## 3. Create the Project & Venv

Open **Windows Terminal** (or PowerShell) and run:

```powershell
mkdir $env:USERPROFILE\mamba3
cd $env:USERPROFILE\mamba3

# Create venv with Python 3.12
py -3.12 -m venv .venv

# Activate (this you'll do every session)
.\.venv\Scripts\Activate.ps1
```

If activation fails with "execution of scripts is disabled":
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then try `.\.venv\Scripts\Activate.ps1` again.

---

## 4. Install PyTorch ROCm 7.13.0 for gfx1152

This is the **native gfx1152 build** — no spoofing needed:

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx1152/ "torch==2.11.0+rocm7.13.0" "torchvision==0.26.0+rocm7.13.0" "torchaudio==2.11.0+rocm7.13.0"
```

### Verify GPU is detected:
```powershell
python -c "import torch; print('CUDA-API available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```
**Expected:** `CUDA-API available: True`, `Device: AMD Radeon 860M Graphics` (or similar). On Windows ROCm, the device shows up under the `torch.cuda` namespace just like NVIDIA — that's intentional API compatibility.

If `False` or device "NONE":
- Adrenalin driver not installed or version too old (need 25.10+ minimum)
- Smart App Control blocking
- Need to reboot after driver install

---

## 5. Install Supporting Dependencies

```powershell
python -m pip install soundfile numpy auraloss torchcrepe
```

**Note:** `auraloss` is needed for multi-resolution STFT loss. If install complains, it's pure Python and shouldn't need any compilation.

`torchcrepe` ships only `tiny` and `full` model capacities. Both download on first use to `%LOCALAPPDATA%\torchcrepe\assets\`.

---

## 6. Drop in the Python Files

Place these four files in `C:\Users\<you>\mamba3\`:
- `model.py`
- `train.py`
- `test_smoke.py`
- `extract_crepe.py`

These are the same files from the Linux setup — they're platform-portable. Only `train.py` has a small Windows compatibility tweak (SIGTERM registration is skipped on Windows; SIGINT still works for graceful Ctrl+C).

---

## 7. Run Smoke Test (Recommended Before Training)

```powershell
cd $env:USERPROFILE\mamba3
python test_smoke.py --device cuda
```

Should complete 10/10 sections in ~60-90 seconds. If any section fails, do NOT proceed to training — fix the failure first.

---

## 8. Dataset Recovery / Re-Download

If your MUSDB18-HQ files were lost:

```powershell
mkdir $env:USERPROFILE\datasets
cd $env:USERPROFILE\datasets

# Download (~21 GB, takes 1-2+ hours)
curl.exe -L -o musdb18hq.zip https://zenodo.org/records/3338373/files/musdb18hq.zip

# Extract
Expand-Archive -Path musdb18hq.zip -DestinationPath musdb18hq
```

Verify structure:
```powershell
Get-ChildItem $env:USERPROFILE\datasets\musdb18hq\train | Select-Object -First 3
```
Should show track directories like `Skelpolu - Together Alone`.

---

## 9. CREPE Pre-Extraction

Required before Phase 2 (step 20,000). torchcrepe ships only `tiny` and `full`:

```powershell
# Train set (~30-60 min on the 860M)
python extract_crepe.py --data $env:USERPROFILE\datasets\musdb18hq\train --device cuda --model full

# Test set (~15-30 min)
python extract_crepe.py --data $env:USERPROFILE\datasets\musdb18hq\test --device cuda --model full
```

**Resumable:** Re-running skips files that already have `.crepe.npy`. Safe to interrupt with Ctrl+C and resume.

---

## 10. Training

### 10.1 Find latest checkpoint:
```powershell
Get-ChildItem $env:USERPROFILE\mamba3\checkpoints\checkpoint_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### 10.2 Resume from a checkpoint:
Replace `XXXXXX` with the step number (e.g., `005000`):

```powershell
python train.py `
  --data $env:USERPROFILE\datasets\musdb18hq\train `
  --resume $env:USERPROFILE\mamba3\checkpoints\checkpoint_XXXXXX.pt `
  --device cuda `
  --batch 4 `
  --ckpt-every 500 `
  --val-every 2000 `
  2>&1 | Tee-Object -FilePath $env:USERPROFILE\mamba3\train.log -Append
```

> **Note the backticks** (`` ` ``) for line continuation — PowerShell's equivalent of bash's `\`.
> **Note `--ckpt-every 500`** (more frequent than Linux 1000) — extra insurance against the gfx1152 GPU hang bug.

### 10.3 Start fresh (no checkpoint available):
```powershell
python train.py `
  --data $env:USERPROFILE\datasets\musdb18hq\train `
  --device cuda `
  --batch 4 `
  --ckpt-every 500 `
  --val-every 2000 `
  2>&1 | Tee-Object -FilePath $env:USERPROFILE\mamba3\train.log
```

### 10.4 Bigger model variant (24 GB shared RAM — fresh start only):
```powershell
python train.py `
  --data $env:USERPROFILE\datasets\musdb18hq\train `
  --device cuda `
  --batch 4 `
  --d-model 384 `
  --d-state 48 `
  --n-layers 8 `
  --ckpt-every 500 `
  --val-every 2000 `
  2>&1 | Tee-Object -FilePath $env:USERPROFILE\mamba3\train_v2.log
```

---

## 11. Monitoring (separate Windows Terminal tabs)

Windows Terminal supports tabs (`Ctrl+T`) and splits (`Alt+Shift+−` horizontal, `Alt+Shift++` vertical). Open new tabs for each monitor.

### GPU usage, temperature, VRAM (live):
PowerShell doesn't ship `rocm-smi` on Windows. Use Task Manager → Performance → GPU, or these alternatives:

```powershell
# Detailed GPU info via WMI
Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor

# Live update every 5s (Ctrl+C to stop)
while ($true) { Clear-Host; Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Where-Object {$_.CookedValue -gt 0} | Format-Table InstanceName, CookedValue -AutoSize; Start-Sleep 5 }
```

### Recommended: install HWiNFO64 for detailed live thermals
Free at [hwinfo.com/download](https://www.hwinfo.com/download/). Shows iGPU temp, package temp, NPU temp, VRM, fan speed, every clock — all in one window.

### CPU + RAM + process tree:
- Task Manager (`Ctrl+Shift+Esc`) → Performance + Details tabs
- Or PowerShell:
  ```powershell
  while ($true) { Clear-Host; Get-Process python | Format-Table Id, ProcessName, CPU, WS, VirtualMemorySize64 -AutoSize; Start-Sleep 3 }
  ```

### Training log live:
```powershell
Get-Content $env:USERPROFILE\mamba3\train.log -Wait -Tail 50
```
This is the PowerShell equivalent of `tail -f`. Ctrl+C to stop.

---

## 12. Graceful Shutdown

### Soft stop (saves checkpoint, exits cleanly):
Press `Ctrl+C` **once** in the training window. The training loop's SIGINT handler writes `interrupt_NNNNNN.pt` then exits.

### Resume from interrupt checkpoint:
```powershell
python train.py --data $env:USERPROFILE\datasets\musdb18hq\train --resume $env:USERPROFILE\mamba3\checkpoints\interrupt_NNNNNN.pt --device cuda 2>&1 | Tee-Object -FilePath $env:USERPROFILE\mamba3\train.log -Append
```

### Hard stop (no checkpoint):
`Ctrl+C` twice in quick succession, or close the terminal window.

---

## 13. Expected Wall-Clock & Phases

| Phase | Steps | Duration (860M Windows) | What's training |
|---|---|---|---|
| 1 | 0 → 20,000 | ~6-14 hr (Windows often slower than Linux on ROCm) | Contrastive semantic encoder (NT-Xent) |
| 2 | 20,000 → 50,000 | ~14-22 hr | Single-track adaptive filter (MR-STFT + Mel-L1) |
| 3 | 50,000 → 100,000 | ~28-50 hr | Cross-track conditioned full model |
| **Total** | | **~2-4+ days** | |

> Add 10-30% to Linux estimates — Windows ROCm preview has higher overhead than mature Linux ROCm.

### Decision checkpoints:
- **Step 5,000** — Phase 1 loss trending down (~0.6 → ~0.2)
- **Step 20,000** — Phase 1 complete; validation contrastive < 1.0
- **Step 50,000** — Phase 2 complete; SI-SDR > 5 dB on validation
- **Step 100,000** — Phase 3 complete

Auto-stops if validation plateaus for 10 val checks (~20,000 steps).

---

## 14. Thermal Thresholds

| Temp | Status | Action |
|---|---|---|
| **<85°C** | Healthy | Continue |
| **85-95°C** | Warm but safe | Verify airflow |
| **>95°C** | Throttling | Pause, improve cooling |
| **>100°C** | Emergency | Stop |

Laptop chassis on hard surface, raised on something to let intake airflow. A cooler pad helps significantly on Strix Point. Sustained >90°C on iGPU under training load is a thermal management failure of the chassis, not the workload.

---

## 15. Troubleshooting

### `torch.cuda.is_available()` returns False
- Driver outdated → install latest Adrenalin
- Reboot after driver install (mandatory)
- Smart App Control still on → disable
- Wrong wheel index URL → re-check the gfx1152-specific URL in §4

### `Could not load this library` / DLL errors
- Install VC++ Redistributables (§2.3)
- Check Smart App Control is off
- Reboot

### `HIP error: invalid device function` or similar runtime errors
- ROCm 7.13.0 is **preview**. Some ops may not work on your specific chip.
- Reduce batch size or model size as workaround
- Worst case: dual-boot to Linux Bazzite (you have working scripts for that)

### GPU hang / system freeze / display driver crash
- This is **issue #5844** on your exact hardware.
- Save your checkpoint state often (`--ckpt-every 500` recommended).
- Reduce `--batch` to 2 or `--d-model` to 192 → smaller workload may avoid the trigger.
- If hangs are frequent, dual-boot to Linux is the only known stable path right now.

### `unrecognized arguments: --resume`
You're running the old `train.py` from before the resume patch. The patched version is ~970 lines and supports `--resume`. Verify:
```powershell
(Get-Content $env:USERPROFILE\mamba3\train.py).Count    # Should be ~970
(Select-String -Path $env:USERPROFILE\mamba3\train.py -Pattern "resume").Count  # Should be 8+
```

### Every step is NaN/Inf
Disable bf16:
```powershell
python train.py --data $env:USERPROFILE\datasets\musdb18hq\train --resume <path> --device cuda --no-amp 2>&1 | Tee-Object -FilePath $env:USERPROFILE\mamba3\train.log -Append
```

### OOM (out of memory)
Reduce batch/model:
```powershell
python train.py --data $env:USERPROFILE\datasets\musdb18hq\train --device cuda --batch 2 --d-model 192
```

### Path errors with backslashes
PowerShell uses `\`, Python prefers `/` or raw strings. The `$env:USERPROFILE\...` form in PowerShell expands to a string that Python can consume via argparse without issue. If you see weird path errors, try forward slashes:
```powershell
python train.py --data $env:USERPROFILE/datasets/musdb18hq/train ...
```

---

## 16. File Locations

```
C:\Users\<you>\mamba3\
├── .venv\                    # Python virtual environment
├── model.py                  # Mamba-3 MIMO architecture
├── train.py                  # Training loop (resume/val/early-stop)
├── test_smoke.py             # Reduced pre-training smoke test (10 sections)
├── smoke_test.py             # Full pre-training smoke test (12 sections; what run_smoke_test.ps1 runs)
├── extract_crepe.py          # Offline CREPE f0 extraction
├── train.log                 # Training output (appended on resume)
└── checkpoints\
    ├── checkpoint_000500.pt
    ├── checkpoint_001000.pt
    ├── ...
    ├── best.pt               # Best validation loss
    ├── final.pt              # Written at training completion
    └── interrupt_NNNNNN.pt   # Written on Ctrl+C

C:\Users\<you>\datasets\musdb18hq\
├── train\                    # 86 tracks
│   └── <Artist - Title>\
│       ├── bass.wav
│       ├── drums.wav
│       ├── vocals.wav
│       ├── other.wav
│       ├── mixture.wav
│       ├── bass.crepe.npy    # ← from extract_crepe.py
│       ├── bass.pitched.npy
│       └── ... (per stem)
└── test\                     # 50 held-out tracks
```

---

## 17. Pre-Walk-Away Checklist

After kicking off training, **wait 5 minutes**, then verify:

1. **Log is being written:**
   ```powershell
   Get-Content $env:USERPROFILE\mamba3\train.log -Tail 5
   ```
2. **GPU is busy** (Task Manager → Performance → GPU should show high % on Compute)
3. **Temperature is OK** — HWiNFO64 or Task Manager → Performance → GPU → Temperature, under 90°C
4. **Loss is finite** — scan log for `NaN/Inf`, should be rare or absent
5. **Steps/sec reasonable** — log line shows `X.XX step/s`, should be > 0.05 (Windows may be slower than Linux)

All five check → close the terminal window IS NOT safe on Windows since closing the terminal kills the process. Either:
- Leave the terminal window open and minimize it, or
- Run via `Start-Process` with `-NoNewWindow` and detach (advanced)

Windows lacks a clean `tmux`-equivalent. The "least bad" option is leaving the Windows Terminal window open on a workspace you're not using.

---

## 18. One-Shot Resume Workflow (Copy-Paste Reference)

```powershell
# ─── On Bazzite host: NOT APPLICABLE on Windows ───
# Sleep prevention is via powercfg (§1) — do that ONCE as admin

# ─── Open Windows Terminal, navigate to project ───
cd $env:USERPROFILE\mamba3

# ─── Activate venv ───
.\.venv\Scripts\Activate.ps1

# ─── Verify environment ───
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# ─── Find latest checkpoint ───
Get-ChildItem checkpoints\checkpoint_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# ─── Resume training (REPLACE XXXXXX with step number) ───
python train.py `
  --data $env:USERPROFILE\datasets\musdb18hq\train `
  --resume checkpoints\checkpoint_XXXXXX.pt `
  --device cuda `
  --batch 4 `
  --ckpt-every 500 `
  --val-every 2000 `
  2>&1 | Tee-Object -FilePath train.log -Append

# Press Ctrl+C ONCE for graceful checkpoint+exit
# Press Ctrl+C TWICE for hard stop

# ─── WHEN TRAINING FINISHES: restore power settings (admin PowerShell) ───
# powercfg /change standby-timeout-ac 30
# powercfg /change hibernate-timeout-ac 60
# (see §1 for full list)
```

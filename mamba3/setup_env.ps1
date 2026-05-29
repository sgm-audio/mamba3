#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Mamba-3 Windows Environment Setup
.DESCRIPTION
    Creates Python virtual environment and installs dependencies.
    Supports ROCm (AMD GPU), DirectML (NPU/any GPU), or CPU-only.
.PARAMETER Backend
    Backend to install: rocm, dml, or cpu (default: auto-detect)
#>

param(
    [ValidateSet("auto", "rocm", "dml", "cpu")]
    [string]$Backend = "auto",
    [switch]$Help = $false
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Definition -Full
    exit 0
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $ScriptDir ".venv"

Write-Host "="x70
Write-Host " Mamba-3 Windows Environment Setup"
Write-Host "="x70

# ────────────────────────────────────────────────────────────────
# 1. Check Python
# ────────────────────────────────────────────────────────────────
$pythonVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from python.org"
    exit 1
}
Write-Host "--> $pythonVer"

# ────────────────────────────────────────────────────────────────
# 2. Create virtual environment
# ────────────────────────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Write-Host "--> Creating virtual environment at $VenvDir ..."
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment"
        exit 1
    }
} else {
    Write-Host "--> Virtual environment exists at $VenvDir"
}

# ────────────────────────────────────────────────────────────────
# 3. Activate
# ────────────────────────────────────────────────────────────────
. (Join-Path $VenvDir "Scripts" "Activate.ps1")
Write-Host "--> Virtual environment activated"

# ────────────────────────────────────────────────────────────────
# 4. Upgrade pip
# ────────────────────────────────────────────────────────────────
Write-Host "--> Upgrading pip..."
python -m pip install --upgrade pip

# ────────────────────────────────────────────────────────────────
# 5. Auto-detect backend if set to auto
# ────────────────────────────────────────────────────────────────
if ($Backend -eq "auto") {
    Write-Host "--> Auto-detecting hardware..."

    # Check for AMD HIP SDK (ROCm)
    $hipExists = $null -ne (Get-Command "hipconfig" -ErrorAction SilentlyContinue)
    $hasAmdGpu = $null -ne (Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like "*AMD*" -or $_.Name -like "*Radeon*" -or $_.Name -like "*Radeon*" })

    # Check for NVIDIA GPU
    $hasNvidiaGpu = $null -ne (Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" })

    # Check for NPU
    $hasNpu = $null -ne (Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -like "*NPU*" -or $_.Name -like "*AI*" })

    if ($hipExists -or $hasAmdGpu) {
        $Backend = "rocm"
        Write-Host "  -> AMD GPU detected with HIP SDK → using ROCm"
    } elseif ($hasNvidiaGpu -or $hasNpu) {
        $Backend = "dml"
        Write-Host "  -> NVIDIA GPU or NPU detected → using DirectML"
    } else {
        $Backend = "cpu"
        Write-Host "  -> No supported GPU/NPU detected → CPU only"
    }
}

# ────────────────────────────────────────────────────────────────
# 6. Install PyTorch + dependencies
# ────────────────────────────────────────────────────────────────
$commonDeps = @(
    "soundfile",
    "auraloss",
    "tqdm",
    "tensorboard"
)

switch ($Backend) {
    "rocm" {
        Write-Host "--> Installing ROCm PyTorch (AMD GPU)..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    }
    "dml" {
        Write-Host "--> Installing DirectML PyTorch (NPU/GPU)..."
        pip install torch torchaudio
        pip install torch-directml
    }
    "cpu" {
        Write-Host "--> Installing CPU-only PyTorch..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch install failed. Check your internet connection and backend compatibility."
    exit 1
}

# Install common dependencies
Write-Host "--> Installing common dependencies..."
foreach ($dep in $commonDeps) {
    pip install $dep
}

# ────────────────────────────────────────────────────────────────
# 7. Verify
# ────────────────────────────────────────────────────────────────
Write-Host "--> Verifying installation..."
$verifyCode = @"
import torch
import torchaudio
import sys

print(f'PyTorch: {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    backend = torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA'
    print(f'GPU: {gpu}')
    print(f'Backend: {backend}')
else:
    print('No CUDA/ROCm device')

try:
    import torch_directml
    print(f'DirectML device: {torch_directml.device()}')
except ImportError:
    print('DirectML: not installed')

try:
    import soundfile
    print(f'soundfile: {soundfile.__version__}')
except ImportError:
    print('soundfile: not installed')
"@
python -c $verifyCode

Write-Host "="x70
Write-Host " Setup complete!"
Write-Host " Backend: $Backend"
Write-Host ""
Write-Host " Next steps:"
Write-Host "   1. Run smoke test:  .\run_smoke_test.ps1"
Write-Host "   2. Start training:  .\run_train.ps1 -Data D:\path\to\musdb18hq\train"
Write-Host "="x70

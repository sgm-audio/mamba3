# ────────────────────────────────────────────────────────────────
# Mamba-3 Windows Install Script — Python 3.14 Edition
# ────────────────────────────────────────────────────────────────
# Target: Windows 11 + Ryzen AI 7 350 + Radeon 860M (gfx1152)
# Python: 3.14 preferred, falls back to 3.13 then 3.12 if needed
# ROCm:   7.13.0 Preview wheels for gfx1152
# ────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Mamba-3 Windows Install (Python 3.14)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Gate 0: Determine project root (default = script directory)
# ────────────────────────────────────────────────────────────────
Write-Host "[0/9] Determining project root..." -ForegroundColor Yellow
$script:PROJECT_ROOT = $PSScriptRoot
$script:DATASETS_ROOT = "$env:USERPROFILE\datasets"
Write-Host "  Project root: $script:PROJECT_ROOT" -ForegroundColor Green
Write-Host "  Datasets:     $script:DATASETS_ROOT" -ForegroundColor Green
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Gate 1: Find Python — try 3.14 first, fall back to 3.13, then 3.12
# ────────────────────────────────────────────────────────────────
Write-Host "[1/9] Locating Python (preferring 3.14)..." -ForegroundColor Yellow
$pythonCmd = $null
$pythonVer = $null
foreach ($ver in @('3.13', '3.14', '3.12', '3.11')) {
    $check = & py -$ver --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "py -$ver"
        $pythonVer = $ver
        Write-Host "  Found: $check via 'py -$ver'" -ForegroundColor Green
        break
    }
}

if ($null -eq $pythonCmd) {
    Write-Host "  ERROR: No supported Python found" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Quick fix options:" -ForegroundColor Yellow
    Write-Host "    A) If you have Python Install Manager (py.exe):" -ForegroundColor Yellow
    Write-Host "         py install 3.14" -ForegroundColor White
    Write-Host "       (Re-run this script after install completes)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    B) Direct installer (bypasses Install Manager):" -ForegroundColor Yellow
    Write-Host "       https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe" -ForegroundColor White
    Write-Host "       During install:" -ForegroundColor Yellow
    Write-Host "         - Check 'Add Python to PATH'" -ForegroundColor White
    Write-Host "         - Click 'Customize installation'" -ForegroundColor White
    Write-Host "         - Check 'Install for all users'" -ForegroundColor White
    Write-Host "         - Check 'Add Python to environment variables'" -ForegroundColor White
    exit 1
}

# Reject Microsoft Store Python
$pyExePath = & py -$pythonVer -c "import sys; print(sys.executable)" 2>$null
if ($pyExePath -match 'WindowsApps') {
    Write-Host "  ERROR: Microsoft Store Python detected at:" -ForegroundColor Red
    Write-Host "    $pyExePath" -ForegroundColor Red
    Write-Host "  Sandbox restrictions break PyTorch ROCm. Uninstall and reinstall from python.org." -ForegroundColor Red
    exit 1
}
Write-Host "  Python: $pyExePath" -ForegroundColor Green

# If we got 3.14, warn that some wheels may not exist yet
if ($pythonVer -eq '3.14') {
    Write-Host "  NOTE: Python 3.14 is new (Oct 2025). If any pip install fails with" -ForegroundColor Yellow
    Write-Host "  'no matching distribution' for torchcrepe or similar, you may need to" -ForegroundColor Yellow
    Write-Host "  drop to 3.13 or 3.12. Script will tell you if/when this happens." -ForegroundColor Yellow
}
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Gate 2: AMD GPU + driver check
# ────────────────────────────────────────────────────────────────
Write-Host "[2/9] Checking GPU driver..." -ForegroundColor Yellow
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -First 1
if ($null -eq $gpu) {
    Write-Host "  ERROR: No AMD GPU detected via WMI" -ForegroundColor Red
    Write-Host "  Install Adrenalin: https://www.amd.com/en/support (NO login needed)" -ForegroundColor Red
    Write-Host "  Search: Ryzen AI 7 350" -ForegroundColor Red
    exit 1
}
Write-Host "  GPU: $($gpu.Name)" -ForegroundColor Green
Write-Host "  Driver: $($gpu.DriverVersion)" -ForegroundColor Green

$driverMajor = [int]($gpu.DriverVersion.Split('.')[0])
if ($driverMajor -lt 31) {
    Write-Host "  WARNING: Driver may be too old for ROCm 7.13.0" -ForegroundColor Yellow
    Write-Host "  Need Adrenalin 25.10+ (driver version 32.x+ in WMI)" -ForegroundColor Yellow
    Write-Host "  Download: https://www.amd.com/en/support (NO LOGIN)" -ForegroundColor Yellow
    $cont = Read-Host "  Continue anyway? (y/N)"
    if ($cont -ne 'y' -and $cont -ne 'Y') { exit 1 }
}
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Gate 3: VC++ Redistributable check
# ────────────────────────────────────────────────────────────────
Write-Host "[3/9] Checking VC++ Redistributable..." -ForegroundColor Yellow
$vcRuntime = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
if ($null -eq $vcRuntime) {
    Write-Host "  WARNING: VC++ 2015-2022 Redistributable not detected" -ForegroundColor Yellow
    Write-Host "  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
    $cont = Read-Host "  Continue anyway? (y/N)"
    if ($cont -ne 'y' -and $cont -ne 'Y') { exit 1 }
} else {
    Write-Host "  VC++ Runtime: $($vcRuntime.Version)" -ForegroundColor Green
}
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Gate 4: Smart App Control check
# ────────────────────────────────────────────────────────────────
Write-Host "[4/9] Checking Smart App Control..." -ForegroundColor Yellow
$sac = Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -Name "VerifiedAndReputablePolicyState" -ErrorAction SilentlyContinue
if ($sac -and $sac.VerifiedAndReputablePolicyState -eq 1) {
    Write-Host "  ERROR: Smart App Control is ON" -ForegroundColor Red
    Write-Host "  Disable: Settings -> Privacy & Security -> Windows Security ->" -ForegroundColor Red
    Write-Host "           App & Browser Control -> Smart App Control -> Off" -ForegroundColor Red
    Write-Host "  ONE-WAY DISABLE: can't be re-enabled without Windows reinstall" -ForegroundColor Red
    Read-Host "  Press Enter once disabled, or Ctrl+C to abort"
} else {
    Write-Host "  Smart App Control: off or absent — proceeding" -ForegroundColor Green
}
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Step 5: Execution policy
# ────────────────────────────────────────────────────────────────
Write-Host "[5/9] Setting PowerShell execution policy..." -ForegroundColor Yellow
try {
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
    Write-Host "  Set to RemoteSigned (current user)" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Could not set execution policy: $_" -ForegroundColor Yellow
}
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Step 6: Project + venv
# ────────────────────────────────────────────────────────────────
Write-Host "[6/9] Creating project + venv..." -ForegroundColor Yellow

if (-not (Test-Path $script:DATASETS_ROOT)) {
    New-Item -ItemType Directory -Path $script:DATASETS_ROOT | Out-Null
}

$venvPath = "$script:PROJECT_ROOT\.venv"
if (Test-Path $venvPath) {
    # Check for stale Linux venv (home = /usr/bin indicates non-Windows)
    $cfgPath = Join-Path $venvPath "pyvenv.cfg"
    $isLinuxVenv = $false
    if (Test-Path $cfgPath) {
        $cfgContent = Get-Content $cfgPath -Raw
        if ($cfgContent -match 'home\s*=\s*/') {
            $isLinuxVenv = $true
        }
    }
    $hasActivateScript = Test-Path (Join-Path $venvPath "Scripts\Activate.ps1")

    if ($isLinuxVenv) {
        Write-Host "  WARNING: Detected Linux-built venv (will not work on Windows)" -ForegroundColor Yellow
        $del = Read-Host "  Delete and recreate? (Y/n)"
        if ($del -ne 'n' -and $del -ne 'N') {
            Remove-Item -Recurse -Force $venvPath
            Write-Host "  Deleted stale venv" -ForegroundColor Yellow
        } else {
            Write-Host "  ERROR: Cannot continue with Linux venv on Windows" -ForegroundColor Red
            exit 1
        }
    } elseif (-not $hasActivateScript) {
        Write-Host "  WARNING: Venv missing Scripts\Activate.ps1 — likely corrupted or cross-platform" -ForegroundColor Yellow
        $del = Read-Host "  Delete and recreate? (Y/n)"
        if ($del -ne 'n' -and $del -ne 'N') {
            Remove-Item -Recurse -Force $venvPath
            Write-Host "  Deleted corrupted venv" -ForegroundColor Yellow
        } else {
            Write-Host "  ERROR: Cannot continue without working venv" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  Windows venv detected (skipping creation)" -ForegroundColor Green
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Host "  Creating venv with Python $pythonVer..." -ForegroundColor Cyan
    & py "-$pythonVer" -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: venv creation failed" -ForegroundColor Red
        exit 1
    }
}

. "$script:PROJECT_ROOT\.venv\Scripts\Activate.ps1"
Write-Host "  Venv activated" -ForegroundColor Green
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Step 7: PyTorch ROCm 7.13.0 for gfx1152
# ────────────────────────────────────────────────────────────────
Write-Host "[7/9] Installing PyTorch + ROCm 7.13.0 for gfx1152..." -ForegroundColor Yellow
Write-Host "  ~3-5 GB download. 5-20 min." -ForegroundColor Cyan

python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR: pip upgrade failed" -ForegroundColor Red; exit 1 }

python -m pip install `
    --index-url https://repo.amd.com/rocm/whl/gfx1152/ `
    "torch==2.11.0+rocm7.13.0" `
    "torchvision==0.26.0+rocm7.13.0" `
    "torchaudio==2.11.0+rocm7.13.0"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  ERROR: PyTorch install failed" -ForegroundColor Red
    if ($pythonVer -eq '3.14') {
        Write-Host "  This is likely because no cp314 wheel exists yet at repo.amd.com" -ForegroundColor Yellow
        Write-Host "  Recovery: install Python 3.13 and re-run script" -ForegroundColor Yellow
        Write-Host "    py install 3.13" -ForegroundColor White
        Write-Host "  Then delete the venv and re-run:" -ForegroundColor Yellow
        Write-Host "    Remove-Item -Recurse -Force $script:PROJECT_ROOT\.venv" -ForegroundColor White
        Write-Host "    .\env_314.ps1" -ForegroundColor White
    } else {
        Write-Host "  Check: 1) Smart App Control off  2) Driver up to date  3) Internet" -ForegroundColor Red
    }
    exit 1
}
Write-Host "  PyTorch + ROCm installed" -ForegroundColor Green
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Step 8: Training dependencies
# ────────────────────────────────────────────────────────────────
Write-Host "[8/9] Installing training dependencies..." -ForegroundColor Yellow

python -m pip install numpy soundfile auraloss

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Core dependencies failed" -ForegroundColor Red
    exit 1
}

# torchcrepe separately — most likely to lack cp314 wheel
Write-Host "  Installing torchcrepe (most fragile on 3.14)..." -ForegroundColor Cyan
python -m pip install torchcrepe
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Binary wheel failed — trying source build..." -ForegroundColor Yellow
    python -m pip install torchcrepe --no-binary :all:
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  WARNING: torchcrepe install failed (binary + source both tried)" -ForegroundColor Yellow
    if ($pythonVer -eq '3.14') {
        Write-Host "  This is the documented 3.14 risk — no cp314 wheel, source may need Cython." -ForegroundColor Yellow
        Write-Host "  Options:" -ForegroundColor Yellow
        Write-Host "    1. Drop to Python 3.13 (cleaner):" -ForegroundColor White
        Write-Host "         py -3.13 --version  (if available)" -ForegroundColor White
        Write-Host "         Remove-Item -Recurse -Force $script:PROJECT_ROOT\.venv" -ForegroundColor White
        Write-Host "         .\env_314.ps1" -ForegroundColor White
    } else {
        Write-Host "  Check: 1) Internet  2) VC++ Redist  3) Cython installed" -ForegroundColor Red
    }
    $cont = Read-Host "  Continue without torchcrepe? (y/N)"
    if ($cont -ne 'y' -and $cont -ne 'Y') { exit 1 }
    Write-Host "  Skipped torchcrepe — you'll need it before CREPE pre-extraction" -ForegroundColor Yellow
} else {
    Write-Host "  torchcrepe installed" -ForegroundColor Green
}
Write-Host ""

# ────────────────────────────────────────────────────────────────
# Gate 9: Verification
# ────────────────────────────────────────────────────────────────
Write-Host "[9/9] Verifying installation..." -ForegroundColor Yellow

$verifyScript = @'
import sys
print(f'Python:    {sys.version.split()[0]}')
print(f'Exec:      {sys.executable}')

try:
    import torch
    print(f'torch:     {torch.__version__}')
    print(f'CUDA-API:  {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU:       {torch.cuda.get_device_name(0)}')
        a = torch.randn(64, 64, device='cuda')
        b = torch.randn(64, 64, device='cuda')
        c = a @ b
        torch.cuda.synchronize()
        print(f'GPU smoke: PASS (matmul mean = {c.mean().item():.4f})')
    else:
        print('GPU:       NOT DETECTED')
        sys.exit(1)
except ImportError as e:
    print(f'torch FAILED: {e}')
    sys.exit(1)

ok = True
for pkg in ['torchaudio', 'torchvision', 'soundfile', 'numpy', 'auraloss']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'unknown')
        print(f'{pkg}: {v}')
    except ImportError as e:
        print(f'{pkg}: MISSING — {e}')
        ok = False

try:
    import torchcrepe
    print(f'torchcrepe: {getattr(torchcrepe, "__version__", "installed")}')
except ImportError:
    print('torchcrepe: not installed (needed for CREPE pre-extraction)')

if not ok:
    sys.exit(1)
print()
print('ALL CHECKS PASSED')
'@


$verifyScript | python -

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  ERROR: Verification failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " INSTALL COMPLETE — Python $pythonVer" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  (All commands below run from the current directory: $script:PROJECT_ROOT)"
Write-Host ""
Write-Host "  1. Smoke test:"
Write-Host "       .\.venv\Scripts\Activate.ps1"
Write-Host "       python smoke_test.py --device cuda"
Write-Host ""
Write-Host "  2. Disable sleep (admin PowerShell):"
Write-Host "       powercfg /change standby-timeout-ac 0"
Write-Host "       powercfg /change monitor-timeout-ac 0"
Write-Host "       powercfg /change hibernate-timeout-ac 0"
Write-Host ""
Write-Host "  3. CREPE pre-extraction (after data is in place):"
Write-Host "       python extract_crepe.py --data $script:DATASETS_ROOT\musdb18hq\train --device cuda --model full"
Write-Host ""
Write-Host "  4. Train:"
Write-Host "       .\run_train.ps1 -Batch 4 -CkptEvery 500"
Write-Host "    Or directly:"
Write-Host "       python train.py --data $script:DATASETS_ROOT\musdb18hq\train --device cuda --batch 4 --ckpt-every 500"
Write-Host ""


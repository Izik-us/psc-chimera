param(
    [string]$PythonLauncher = 'py -3.11',
    [switch]$CpuOnly,
    [switch]$WithDev
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path '.venv')) {
    Invoke-Expression "$PythonLauncher -m venv .venv"
}
$Python = Join-Path $Root '.venv\Scripts\python.exe'
& $Python -m pip install --upgrade pip wheel setuptools

if ($CpuOnly) {
    & $Python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

# Install from setup.py so the fast path cannot drift from the repository's
# declared runtime dependencies.  The preinstalled CPU torch satisfies the
# torch requirement when -CpuOnly is selected.
& $Python -m pip install -e .

if ($WithDev) {
    & $Python -m pip install -r requirements-dev.txt
}

& $Python scripts\check_install.py

Write-Host ''
Write-Host 'PSC-CHIMERA fast environment ready.' -ForegroundColor Green
Write-Host "Python: $Python"
Write-Host "Run: & '$Python' scripts\run_design.py --help"
Write-Host "Check: & '$Python' scripts\check_install.py"
Write-Host 'Weights: .\scripts\download_weights.ps1 .\weights'

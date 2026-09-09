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
} else {
    & $Python -m pip install torch
}
& $Python -m pip install -e . --no-deps
& $Python -m pip install transformers fair-esm einops biopython faiss-cpu numpy scipy pandas h5py tqdm rich typer

if ($WithDev) {
    & $Python -m pip install pytest 'black>=23.0.0'
}

Write-Host ''
Write-Host 'PSC-CHIMERA fast environment ready.' -ForegroundColor Green
Write-Host "Python: $Python"
Write-Host "Run: & '$Python' scripts\run_design.py --help"
Write-Host "Check: & '$Python' scripts\check_install.py"
Write-Host 'Weights: .\scripts\download_weights.ps1 .\weights'

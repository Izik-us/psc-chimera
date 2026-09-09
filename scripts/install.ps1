param(
    [string]$PythonLauncher = 'py -3.11'
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path '.venv')) {
    Invoke-Expression "$PythonLauncher -m venv .venv"
}

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python was not created at $Python"
}

& $Python -m pip install --upgrade pip wheel setuptools
& $Python -m pip install -e .

Write-Host ''
Write-Host 'PSC-CHIMERA environment ready.' -ForegroundColor Green
Write-Host "Python: $Python"
Write-Host 'Run:'
Write-Host "  & '$Python' scripts\run_design.py --help"
Write-Host "  & '$Python' scripts\check_install.py"
Write-Host ''
Write-Host 'For development/test dependencies:'
Write-Host "  & '$Python' -m pip install -e '.[dev]'"
Write-Host ''
Write-Host 'For optional molecular-dynamics validation:'
Write-Host "  & '$Python' -m pip install -e '.[md]'"
Write-Host ''
Write-Host 'To download the supported native checkpoints:'
Write-Host "  & '$Python' -c \"from pathlib import Path; import urllib.request; Path('weights').mkdir(exist_ok=True); print('Use scripts\\download_weights.ps1 .\\weights for checkpoint downloads.')\""
Write-Host '  .\scripts\download_weights.ps1 .\weights'

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path '.venv')) {
    py -3.11 -m venv .venv
}

$Python = Join-Path $Root '.venv\Scripts\python.exe'
& $Python -m pip install --upgrade pip wheel setuptools
& $Python -m pip install -e .

Write-Host ''
Write-Host 'PSC-CHIMERA environment ready.' -ForegroundColor Green
Write-Host "Python: $Python"
Write-Host 'Run:'
Write-Host "  & '$Python' scripts\run_design.py --help"
Write-Host ''
Write-Host 'For development/test dependencies:'
Write-Host "  & '$Python' -m pip install -e '.[dev]'"
Write-Host ''
Write-Host 'For optional molecular-dynamics validation:'
Write-Host "  & '$Python' -m pip install -e '.[md]'"
Write-Host ''
Write-Host 'To download the supported native checkpoints:'
Write-Host '  bash scripts/download_weights.sh .\weights'

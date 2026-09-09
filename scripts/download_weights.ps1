param(
    [string]$WeightsDir = '.\weights'
)

$ErrorActionPreference = 'Stop'

New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null

function Download-Artifact {
    param(
        [string]$Url,
        [string]$Output,
        [string]$Label
    )

    $part = "$Output.part"
    if ((Test-Path $Output) -and ((Get-Item $Output).Length -gt 0)) {
        Write-Host "  [OK] $Label already present: $Output"
        return
    }

    Write-Host "  [DOWN] $Label"
    try {
        Start-BitsTransfer -Source $Url -Destination $part -DisplayName "PSC-CHIMERA: $Label" -Description $Url
    }
    catch {
        Write-Warning "BITS failed; falling back to Invoke-WebRequest"
        Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing
    }

    if (-not (Test-Path $part) -or ((Get-Item $part).Length -eq 0)) {
        Remove-Item -Force -ErrorAction SilentlyContinue $part
        throw "Download failed or produced an empty partial file: $part"
    }

    Move-Item -Force $part $Output
    Write-Host "  [OK] $Label ready"
}

Write-Host '========================================================'
Write-Host ' PSC-CHIMERA Windows Weight Downloader'
Write-Host " Destination: $WeightsDir"
Write-Host '========================================================'

Download-Artifact `
    'https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt' `
    (Join-Path $WeightsDir 'proteinmpnn_v48_020.pt') `
    'ProteinMPNN v_48_020'

Download-Artifact `
    'https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Base_ckpt.pt' `
    (Join-Path $WeightsDir 'rfdiffusion_base.pt') `
    'RFdiffusion Base_ckpt'

Write-Host ''
Write-Host 'OpenFold/AlphaFold and PoET remain optional native-backend artifacts.'
Write-Host 'The local approximation classes intentionally do not load incompatible upstream checkpoints.'
Write-Host ''
Write-Host 'Next: .\.venv\Scripts\python.exe scripts\run_design.py --help'

param(
    [string]$WeightsDir = '.\weights'
)

$ErrorActionPreference = 'Stop'
$chimera = Get-Command chimera -ErrorAction SilentlyContinue
if ($null -eq $chimera) {
    throw 'Install PSC-CHIMERA first so the chimera models command is available.'
}

foreach ($asset in @('proteinmpnn_v48_020', 'rfdiffusion_base')) {
    & $chimera.Source models fetch $asset --cache-dir $WeightsDir
    if ($LASTEXITCODE -ne 0) {
        throw "Model acquisition failed for $asset (exit code $LASTEXITCODE)"
    }
}

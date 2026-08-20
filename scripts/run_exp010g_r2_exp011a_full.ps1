# EXP-010G-R2 + EXP-011A full overnight
# Salve este arquivo em scripts\run_exp010g_r2_exp011a_full.ps1
# Execute na raiz do projeto:
#   powershell -ExecutionPolicy Bypass -File scripts\run_exp010g_r2_exp011a_full.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$Exp010Dir = Join-Path $ProjectRoot "resultados\experimentos\EXP-010G-R2"
$Exp011Dir = Join-Path $ProjectRoot "resultados\experimentos\EXP-011A_R2_FULL"

New-Item -ItemType Directory -Force -Path $Exp010Dir | Out-Null
New-Item -ItemType Directory -Force -Path $Exp011Dir | Out-Null

$Log010 = Join-Path $Exp010Dir "run_$Timestamp.log"
$Log011 = Join-Path $Exp011Dir "run_$Timestamp.log"

Write-Host "================================================================================"
Write-Host "EXP-010G-R2 - Enriquecimento model-ready"
Write-Host "Log: $Log010"
Write-Host "================================================================================"

python scripts\exp_010g_r2_enrich_dataset_v2_model_ready.py 2>&1 | Tee-Object -FilePath $Log010
if ($LASTEXITCODE -ne 0) {
    throw "EXP-010G-R2 falhou com exit code $LASTEXITCODE"
}

Write-Host "================================================================================"
Write-Host "EXP-011A - Replay FULL no dataset enriquecido"
Write-Host "Log: $Log011"
Write-Host "================================================================================"

python scripts\exp_011a_replay_baseline_dataset_v2.py `
  --full `
  --workers 1 `
  --input dados\hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv `
  --output-dir resultados\experimentos\EXP-011A_R2_FULL `
  2>&1 | Tee-Object -FilePath $Log011

if ($LASTEXITCODE -ne 0) {
    throw "EXP-011A full falhou com exit code $LASTEXITCODE"
}

Write-Host "================================================================================"
Write-Host "PROCESSO CONCLUIDO COM SUCESSO"
Write-Host "EXP-010G-R2: $Exp010Dir"
Write-Host "EXP-011A FULL: $Exp011Dir"
Write-Host "================================================================================"

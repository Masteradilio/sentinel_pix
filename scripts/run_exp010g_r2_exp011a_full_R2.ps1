# EXP-010G-R2 + EXP-011A full overnight - R2
# Versao corrigida para Windows PowerShell:
# - Nao usa pipe 2>&1 | Tee-Object em comandos Python.
# - Evita que logs INFO do Python, que saem em STDERR por padrao, sejam tratados como NativeCommandError.
#
# Execute na raiz do projeto:
#   powershell -ExecutionPolicy Bypass -File scripts\run_exp010g_r2_exp011a_full_R2.ps1

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

function Run-PythonCommand {
    param(
        [Parameter(Mandatory=$true)][string]$Title,
        [Parameter(Mandatory=$true)][string]$CommandLine,
        [Parameter(Mandatory=$true)][string]$LogPath
    )

    Write-Host "================================================================================"
    Write-Host $Title
    Write-Host "Log: $LogPath"
    Write-Host "================================================================================"

    # Executa via cmd.exe para redirecionar stdout/stderr sem transformar STDERR em erro do PowerShell.
    $cmd = "$CommandLine > `"$LogPath`" 2>&1"
    cmd.exe /c $cmd

    $exitCode = $LASTEXITCODE

    Write-Host ""
    Write-Host "Ultimas linhas do log:"
    Write-Host "--------------------------------------------------------------------------------"
    if (Test-Path $LogPath) {
        Get-Content $LogPath -Tail 80
    } else {
        Write-Host "Log nao encontrado: $LogPath"
    }
    Write-Host "--------------------------------------------------------------------------------"

    if ($exitCode -ne 0) {
        throw "$Title falhou com exit code $exitCode. Veja o log completo em: $LogPath"
    }
}

Run-PythonCommand `
    -Title "EXP-010G-R2 - Enriquecimento model-ready" `
    -CommandLine "python scripts\exp_010g_r2_enrich_dataset_v2_model_ready.py" `
    -LogPath $Log010

Run-PythonCommand `
    -Title "EXP-011A - Replay FULL no dataset enriquecido" `
    -CommandLine "python scripts\exp_011a_replay_baseline_dataset_v2.py --full --workers 1 --input dados\hmo_ml_tb_pix_dataset_v2_180d_v1_enriched.csv --output-dir resultados\experimentos\EXP-011A_R2_FULL" `
    -LogPath $Log011

Write-Host "================================================================================"
Write-Host "PROCESSO CONCLUIDO COM SUCESSO"
Write-Host "EXP-010G-R2: $Exp010Dir"
Write-Host "EXP-011A FULL: $Exp011Dir"
Write-Host "================================================================================"

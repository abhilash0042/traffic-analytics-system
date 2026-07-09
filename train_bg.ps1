# train_bg.ps1 — Run a training task in a NEW window so your main terminal stays free.
#
# Usage:
#   .\train_bg.ps1 plates          # train plate detector
#   .\train_bg.ps1 helmets         # train helmet detector
#   .\train_bg.ps1 vehicles        # train vehicle detector
#   .\train_bg.ps1 plates -Resume  # resume a paused run
#   .\train_bg.ps1 all             # train all models in order
#
# Logs are saved to: logs\train_<task>_<timestamp>.log

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("plates","helmets","vehicles","all","status")]
    [string]$Task,

    [switch]$Resume,
    [switch]$Force,
    [switch]$DryRun
)

$root = $PSScriptRoot
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "train_${Task}_${timestamp}.log"

# Build argument string
$trainArgs = "--$Task"
if ($Task -eq "all")    { $trainArgs = "--all" }
if ($Task -eq "status") { $trainArgs = "--status" }
if ($Resume)            { $trainArgs += " --resume" }
if ($Force)             { $trainArgs += " --force" }
if ($DryRun)            { $trainArgs += " --dry-run" }

Write-Host ""
Write-Host "Launching training in a new window..." -ForegroundColor Cyan
Write-Host "  Task : $Task  Args: $trainArgs" -ForegroundColor White
Write-Host "  Log  : $logFile" -ForegroundColor White
Write-Host ""
Write-Host "This terminal stays FREE — keep chatting with the AI agent here." -ForegroundColor Green
Write-Host "Watch live progress with:" -ForegroundColor Yellow
Write-Host "  Get-Content '$logFile' -Wait" -ForegroundColor Yellow
Write-Host ""

$innerScript = @"
Set-Location '$root'
if (Test-Path '.\venv\Scripts\Activate.ps1') { & '.\venv\Scripts\Activate.ps1' }
Write-Host 'Starting: python train_models.py $trainArgs' -ForegroundColor Cyan
Write-Host 'Log: $logFile' -ForegroundColor Yellow
Write-Host '---'
python train_models.py $trainArgs 2>&1 | Tee-Object -FilePath '$logFile'
Write-Host ''
Write-Host 'Training finished. Press Enter to close.' -ForegroundColor Green
Read-Host
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $innerScript

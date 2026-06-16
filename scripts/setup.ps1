param(
	[string]$VenvPath = ".venv",
	[switch]$Recreate,
	[switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonCmd = "python"

if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
	throw "Python was not found on PATH. Install Python 3.12+ and retry."
}

if ($Recreate -and (Test-Path -LiteralPath $VenvPath)) {
	Write-Host "Removing existing virtual environment at $VenvPath" -ForegroundColor Yellow
	Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPath)) {
	Write-Host "Creating virtual environment at $VenvPath" -ForegroundColor Cyan
	& $pythonCmd -m venv $VenvPath
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
	throw "Virtual environment Python not found at $venvPython"
}

Write-Host "Upgrading pip" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

if (-not $SkipInstall) {
	if (-not (Test-Path -LiteralPath "requirements.txt")) {
		throw "requirements.txt not found in repository root"
	}
	Write-Host "Installing dependencies from requirements.txt" -ForegroundColor Cyan
	& $venvPython -m pip install -r requirements.txt
}

$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "  $activateScript" -ForegroundColor Green

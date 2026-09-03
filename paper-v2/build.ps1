# One command to rebuild the paper from whatever data exists right now.
#
#     powershell -ExecutionPolicy Bypass -File paper-v2\build.ps1
#
# Safe to run mid-queue: it reads whatever runs/ currently holds and reports
# what is still missing. Run it every time a model finishes.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== 1. collecting numbers from runs/ ==" -ForegroundColor Cyan
py paper-v2\collect.py
if (-not $?) { throw "collect.py failed" }

Write-Host "`n== 2. regenerating figures ==" -ForegroundColor Cyan
py paper-v2\figures.py
if (-not $?) { throw "figures.py failed" }

Write-Host "`n== 3. compiling main.tex ==" -ForegroundColor Cyan
if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    Set-Location "$root\paper-v2"
    pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    Set-Location $root
    Write-Host "wrote paper-v2\main.pdf"
    $stale = Select-String -Path "paper-v2\main.tex" -Pattern '\?\?' -SimpleMatch
    if ($stale) { Write-Host "WARNING: ?? placeholders remain" -ForegroundColor Yellow }
} else {
    Write-Host "pdflatex not installed - skipping compile." -ForegroundColor Yellow
    Write-Host "Upload paper-v2\ to Overleaf and compile there instead."
}

Write-Host "`nDone. Check the printout above for untestable categories." -ForegroundColor Green

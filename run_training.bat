@echo off
:: Run this from the project root with the .venv activated,
:: OR double-click this file (it activates the venv automatically).

cd /d "%~dp0"

:: Activate the venv if not already active
if "%VIRTUAL_ENV%"=="" (
    call .venv\Scripts\activate.bat
)

echo Starting Phase 4A: U-Net training for bottle category...
echo Outputs will be saved to ml/outputs/bottle/unet/
echo Press Ctrl+C to cancel.
echo.

python ml\train\bottle_unet.py %*

echo.
echo Done! Check ml/outputs/bottle/unet/ for results.
pause

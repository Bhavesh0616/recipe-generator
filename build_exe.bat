@echo off
setlocal

REM ---- CONFIG ----
set "APP_NAME=FoodChat"
set "ENTRY=main.py"

REM Use the venv you created with Python 3.10.11
for %%I in ("%~dp0") do set "PROJ=%%~fI"
set "VENV=%PROJ%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

echo Using Python: %PY%
if not exist "%PY%" (
  echo [!] Could not find venv at %VENV%
  echo Create it first:  python -m venv .venv  ^&^&  .\.venv\Scripts\pip install --upgrade pip
  exit /b 1
)

REM ---- Prevent optional heavy imports during analysis ----
set "TRANSFORMERS_NO_TF=1"
set "TRANSFORMERS_NO_TORCHVISION=1"

REM ---- Install/update deps ----
echo.
echo Installing requirements...
"%PIP%" install --upgrade pip
"%PIP%" install -r "%PROJ%\requirements.txt" || (echo [x] Failed to install requirements.& exit /b 1)

REM ---- Clean old builds ----
rmdir /s /q "%PROJ%\build" 2>nul
rmdir /s /q "%PROJ%\dist"  2>nul
del /q "%PROJ%\%APP_NAME%.spec" 2>nul

REM ---- Optional icon (won't fail if missing) ----
set "ICON_OPT="
if exist "%PROJ%\app_icon.ico" (
  set "ICON_OPT=--icon ""%PROJ%\app_icon.ico"""
) else (
  set "ICON_OPT=--icon NONE"
)

REM ---- Build EXE ----
echo.
echo Building standalone .exe (this may take a while)...
"%PY%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name "%APP_NAME%" ^
  --paths "%PROJ%" ^
  %ICON_OPT% ^
  ^
  --add-data "%PROJ%\assets;assets" ^
  --add-data "%PROJ%\data;data" ^
  --add-data "%PROJ%\frontend_food_chat.html;." ^
  --add-data "%PROJ%\mic.html;." ^
  ^
  REM ***** CRITICAL FIX: bundle the ENTIRE indic_transliteration package *****
  --collect-all indic_transliteration ^
  ^
  REM Other packages that need data/submodules
  --collect-data pythainlp ^
  --collect-data jieba ^
  --collect-data pypinyin ^
  --collect-data thai2rom ^
  --collect-data whisper ^
  --collect-submodules transformers ^
  --collect-submodules pythainlp ^
  --collect-submodules jieba ^
  --collect-submodules pypinyin ^
  --collect-submodules thai2rom ^
  --collect-submodules whisper ^
  ^
  --hidden-import torch ^
  --exclude-module "transformers.kernels.falcon_mamba" ^
  --exclude-module "torch.utils.tensorboard" ^
  "%PROJ%\%ENTRY%"

if errorlevel 1 (
  echo [x] PyInstaller failed.
  exit /b 1
)

echo.
echo ✓ Build done. EXE at: %PROJ%\dist\%APP_NAME%\%APP_NAME%.exe
echo   Data folder:          %PROJ%\dist\%APP_NAME%\data
endlocal

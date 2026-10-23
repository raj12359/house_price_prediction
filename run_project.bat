@echo off
title EstateIQ - House Price Prediction
echo ========================================================
echo        ESTATEIQ: HOUSE PRICE PREDICTION SYSTEM
echo ========================================================
echo.
cd /d "%~dp0"

echo [1/2] Checking Python environment...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH! Please install Python.
    pause
    exit /b
)

echo [2/2] Launching Flask Web Application...
echo Open your web browser at: http://127.0.0.1:5000
echo.
echo Press CTRL+C in this terminal window to stop the server.
echo ========================================================
echo.

python app.py
pause

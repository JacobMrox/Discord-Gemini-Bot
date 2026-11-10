@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from python.org and tick "Add to PATH".
    pause
    exit /b 1
)

python -c "import discord, aiohttp" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency install failed. Fix the error above and run this file again.
        pause
        exit /b 1
    )
)

python Discord_bot_google_gtts.py
pause

@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from python.org and tick "Add to PATH".
    pause
    exit /b 1
)

rem Check every module the bot imports. A machine that already had discord.py used
rem to pass a narrower check, skip the install and silently lose edge-tts.
python -c "import discord, aiohttp, edge_tts, gtts, bs4" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt...
    python -m pip install --disable-pip-version-check -r requirements.txt
    python -c "import discord, aiohttp, edge_tts, gtts, bs4" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo Some dependencies are still missing. Run: python -m pip install -r requirements.txt
        echo The bot will still start, but the natural voice may be unavailable.
        echo.
    )
)

python Discord_bot_google_gtts.py
pause

@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from python.org and tick "Add to PATH".
    pause
    exit /b 1
)

rem discord.py 2.7 or newer is required. Discord refuses voice connections that do
rem not implement the DAVE protocol, closing them with code 4017.
python -c "import sys, discord; sys.exit(0 if tuple(int(p) for p in discord.__version__.split('.')[:2]) >= (2, 7) else 1)" >nul 2>nul
if errorlevel 1 goto install

python -c "import aiohttp, edge_tts, gtts, bs4, davey" >nul 2>nul
if errorlevel 1 goto install
goto run

:install
echo Updating dependencies from requirements.txt...
python -m pip install --upgrade --disable-pip-version-check -r requirements.txt
python -c "import aiohttp, edge_tts, gtts, bs4, davey" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Some dependencies are still missing. Run: python -m pip install -r requirements.txt
    echo The bot will start, but voice may not work.
    echo.
)
python -c "import sys, discord; sys.exit(0 if tuple(int(p) for p in discord.__version__.split('.')[:2]) >= (2, 7) else 1)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: discord.py is still older than 2.7, so Discord will reject voice with 4017.
    echo Try: python -m pip install --upgrade "discord.py[voice]"
    echo.
)

:run
python Discord_bot_google_gtts.py
pause

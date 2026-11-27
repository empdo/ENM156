@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sounddevice" >nul 2>&1
if errorlevel 1 (
    echo Installing the microphone library. First launch needs internet.
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" robot_audio_sampler.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo Setup or launch failed. Install Python 3.10 or newer and see README.txt.
pause
exit /b 1

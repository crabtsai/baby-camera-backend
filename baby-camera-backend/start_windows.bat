@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo Project Python environment was not found:
    echo %PROJECT_PYTHON%
    echo.
    echo Please create the environment and install requirements first.
    pause
    exit /b 1
)

"%PROJECT_PYTHON%" -m scripts.open_monitor
if errorlevel 1 (
    pause
    exit /b 1
)
exit /b 0

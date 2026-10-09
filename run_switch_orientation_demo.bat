@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run: python -m venv .venv
  echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m switch_orientation_demo --replay --show-direction %*
if errorlevel 1 pause

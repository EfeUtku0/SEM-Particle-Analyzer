@echo off
REM Double-click to launch the SEM Particle Analyzer (development mode).
REM Windows counterpart of run.command.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No virtual environment found.
  echo Run these once, from this folder:
  echo     python -m venv .venv
  echo     .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\pythonw.exe" app\gui.py

@echo off
setlocal
if not exist "%~dp0dist\ArrayLab-0.26.0.dev2\ArrayLab.exe" (
  echo ArrayLab.exe was not found. Please build the release first.
  pause
  exit /b 1
)
start "" "%~dp0dist\ArrayLab-0.26.0.dev2\ArrayLab.exe"
endlocal

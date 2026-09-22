@echo off
rem MYVOCAL Studio launcher.
rem All Korean messages are printed by Python so that the console
rem code page does not garble them.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 goto NOPYTHON
    set "PY=python"
)

%PY% bootstrap.py
if errorlevel 1 goto FAILED

%PY% main.py %*
if errorlevel 1 goto FAILED
exit /b 0

:NOPYTHON
echo.
echo Python is not installed, or not on PATH.
echo Download it from https://www.python.org/downloads/
echo and check "Add Python to PATH" during setup.
echo.
echo ---
echo.
echo Python 이 설치돼 있지 않거나 PATH 에 없습니다.
echo https://www.python.org/downloads/ 에서 받아 설치해 주세요.
echo 설치할 때 "Add Python to PATH" 를 반드시 체크하세요.
echo.
pause
exit /b 1

:FAILED
echo.
pause
exit /b 1

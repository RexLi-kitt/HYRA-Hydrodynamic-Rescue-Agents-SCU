@echo off
setlocal
cd /d "%~dp0"
whoami /groups | findstr /i "Mandatory Label"
set "GRADLE_USER_HOME=%~dp0freeterraforged-reproduction\.runtime\gradle"
echo Gradle cache: %GRADLE_USER_HOME%
python terrain-lab\check_java_cache.py
if errorlevel 1 echo [warn] Java cache check failed; starting the workbench anyway. See terrain-lab\.runtime\windows-cache-check.log
echo Open http://127.0.0.1:8768/ after the server starts.
echo Keep this window open. First generation downloads build dependencies.
python terrain-lab\server.py --port 8768
pause
endlocal
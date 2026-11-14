@echo off
setlocal
cmake -P "%~dp0install-ispc.cmake"
exit /b %errorlevel%


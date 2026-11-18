@echo off
>&2 echo Windows engine builds use the Visual Studio 2026 preset. Redirecting to scripts\build-vs.bat.
call "%~dp0build-vs.bat" %*
exit /b %errorlevel%


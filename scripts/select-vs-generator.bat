@echo off
setlocal
set "CMAKE_GENERATOR="
set "GEFS_VS_GENERATOR="
for /f "tokens=2,3,4,5" %%a in ('cmake --help ^| findstr /r /c:"^\* Visual Studio "') do set "GEFS_VS_GENERATOR=%%a %%b %%c %%d"
if not defined GEFS_VS_GENERATOR (
    >&2 echo CMake could not find a supported Visual Studio installation.
    exit /b 1
)
endlocal & set "GEFS_VS_GENERATOR=%GEFS_VS_GENERATOR%"

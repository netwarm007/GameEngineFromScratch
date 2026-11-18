@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=Debug"

if /I "%CONFIG%"=="Debug" (
    set "BUILD_PRESET=windows-vs2026-debug"
) else if /I "%CONFIG%"=="Release" (
    set "BUILD_PRESET=windows-vs2026-release"
) else (
    >&2 echo Unsupported configuration "%CONFIG%". Use Debug or Release.
    exit /b 2
)

pushd "%ROOT%"
cmake --preset windows-vs2026
if errorlevel 1 (popd & exit /b 1)
cmake --build --preset %BUILD_PRESET%
if errorlevel 1 (popd & exit /b 1)
popd


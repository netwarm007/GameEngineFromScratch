@echo off
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\External\build\zlib-windows"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=Debug"

call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git -C "%ROOT%" submodule update --init -- External/src/zlib || exit /b 1
cmake --fresh -S "%ROOT%\External\src\zlib" -B "%BUILD_DIR%" ^
    "-DCMAKE_INSTALL_PREFIX=%ROOT%\External" -G "%GEFS_VS_GENERATOR%" ^
    -A x64 -DBUILD_SHARED_LIBS=OFF -Thost=x64 || exit /b 1
cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target install || exit /b 1


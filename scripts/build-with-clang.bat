@echo off
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\build\clang-cl"
set "PATH=%PATH%;%ROOT%\External\Windows\bin"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=Debug"

cmake --fresh -S "%ROOT%" -B "%BUILD_DIR%" -G "NMake Makefiles" ^
    "-DCMAKE_TOOLCHAIN_FILE=%ROOT%\cmake\clang-cl.cmake" || exit /b 1
cmake --build "%BUILD_DIR%" --config "%CONFIG%" || exit /b 1

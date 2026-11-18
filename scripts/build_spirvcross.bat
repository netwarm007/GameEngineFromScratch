@echo off
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\External\build\spirv-cross-windows"
set "INSTALL_DIR=%ROOT%\External\Windows"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=Debug"

git -C "%ROOT%" submodule update --init -- External/src/spirv-cross || exit /b 1
cmake --fresh -S "%ROOT%\External\src\spirv-cross" -B "%BUILD_DIR%" ^
    "-DCMAKE_INSTALL_PREFIX=%INSTALL_DIR%" "-DCMAKE_INSTALL_RPATH=%INSTALL_DIR%" ^
    -DCMAKE_BUILD_TYPE=RELEASE || exit /b 1
cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target install || exit /b 1
echo "Completed build of SPIRV-Cross"

@echo off
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\External\build\zlib-emscripten-windows"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=Debug"

git -C "%ROOT%" submodule update --init -- External/src/zlib || exit /b 1
cmake --fresh -S "%ROOT%\External\src\zlib" -B "%BUILD_DIR%" ^
    -DCMAKE_CROSSCOMPILING_EMULATOR=node.exe "-DCMAKE_TOOLCHAIN_FILE=%ROOT%\cmake\Emscripten.cmake" ^
    "-DCMAKE_INSTALL_PREFIX=%ROOT%\External" -G "NMake Makefiles" -DBUILD_SHARED_LIBS=off || exit /b 1
cmake --build "%BUILD_DIR%" --config "%CONFIG%" --target install || exit /b 1

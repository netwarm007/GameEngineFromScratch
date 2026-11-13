@echo off
setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External\src\ISPCTextureCompressor
if errorlevel 1 exit /b 1
if not exist External\build\ISPCTextureCompressor mkdir External\build\ISPCTextureCompressor
pushd External\build\ISPCTextureCompressor || exit /b 1
cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "%GEFS_VS_GENERATOR%" -A "x64" -Thost=x64 ..\..\src\ISPCTextureCompressor
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd


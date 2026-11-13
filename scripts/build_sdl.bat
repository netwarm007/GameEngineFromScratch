@echo off
setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External/src/SDL
mkdir External\build\SDL
pushd External\build\SDL
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -DBUILD_SHARED_LIBS=OFF -G "%GEFS_VS_GENERATOR%" -A "x64" -Thost=x64 ../../src/SDL
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd

@echo off
git submodule update --init External\src\openal
if errorlevel 1 exit /b 1
if not exist External\build\openal mkdir External\build\openal
pushd External\build\openal || exit /b 1
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ..\..\src\openal
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd

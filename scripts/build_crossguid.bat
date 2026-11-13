@echo off
git submodule update --init External/src/crossguid
if errorlevel 1 exit /b 1
if not exist External\build\crossguid mkdir External\build\crossguid
pushd External\build\crossguid || exit /b 1
cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ..\..\src\crossguid
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd


@echo off
setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External\src\freealut
mkdir External\build\freealut
pushd External\build\freealut
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -G "%GEFS_VS_GENERATOR%" -A "x64" -Thost=x64 -DBUILD_TESTS=OFF -DBUILD_STATIC=ON -DBUILD_EXAMPLES=OFF ..\..\src\freealut
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd

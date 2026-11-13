@echo off
setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External/src/glfw
mkdir External\build\glfw
pushd External\build\glfw
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF -DGLFW_BUILD_EXAMPLES=OFF -G "%GEFS_VS_GENERATOR%" -A "x64" ../../src/glfw
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd

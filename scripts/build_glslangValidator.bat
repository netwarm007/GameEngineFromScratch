@echo off
git submodule update --init External\src\glslang
if errorlevel 1 exit /b 1
if not exist External\build\glslang mkdir External\build\glslang
pushd External\build\glslang || exit /b 1
cmake -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DBUILD_EXTERNAL=NO ../../src/glslang
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd
echo "Completed build of glslangValidator"

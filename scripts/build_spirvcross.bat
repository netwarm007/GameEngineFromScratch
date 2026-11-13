@echo off
git submodule update --init External\src\spirv-cross
if errorlevel 1 exit /b 1
if not exist External\build\spirv-cross mkdir External\build\spirv-cross
pushd External\build\spirv-cross || exit /b 1
rm -rf *
if errorlevel 1 (popd & exit /b 1)
cmake -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/spirv-cross
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd
echo "Completed build of SPIRV-Cross"

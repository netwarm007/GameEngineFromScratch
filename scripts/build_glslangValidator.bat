@echo off
git submodule update --init External\src\glslang
mkdir External\build\glslang
pushd External\build\glslang
cmake -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DBUILD_EXTERNAL=NO ../../src/glslang
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd
echo "Completed build of glslangValidator"

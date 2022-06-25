@echo off
git submodule update --init External\src\spirv-cross
mkdir External\build\spirv-cross
pushd External\build\spirv-cross
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/spirv-cross
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd
echo "Completed build of SPIRV-Cross"

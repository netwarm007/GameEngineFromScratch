@echo off
git submodule update --init External\src\spirv-cross
mkdir External\build\spirv-cross
pushd External\build\spirv-cross
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/spirv-cross
cmake --build . --config Release --target install
popd
echo "Completed build of SPIRV-Cross"

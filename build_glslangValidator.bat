@echo off
git submodule update --init External\src\glslang
mkdir External\build\glslang
pushd External\build\glslang
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DBUILD_EXTERNAL=NO ../../src/glslang
cmake --build . --config Release --target install
popd
echo "Completed build of glslangValidator"

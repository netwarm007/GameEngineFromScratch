@echo off
git submodule update --init External\src\glslangValidator
mkdir External\build\glslangValidator
cd External\build\glslangValidator
rm -rf *
cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=../../Windows/ -DCMAKE_INSTALL_RPATH=../../Windows/ -DCMAKE_BUILD_TYPE=RELEASE ../../src/glslangValidator
cmake --build . --config release --target install
echo "Completed build of glslangValidator"

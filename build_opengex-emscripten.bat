@echo off
git submodule update --init External\src\opengex
mkdir External\build\opengex
pushd External\build\opengex
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=../../../cmake/Emscripten.cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "NMake Makefiles" ..\..\src\opengex
cmake --build . --config Release --target install
popd


@echo off
git submodule update --init External/src/SDL
mkdir External\build\SDL
pushd External\build\SDL
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=../../../cmake/Emscripten.cmake -DCMAKE_INSTALL_PREFIX=../../Emscripten/ -DCMAKE_INSTALL_RPATH=../../Emscripten/ -G "NMake Makefiles" ../../src/SDL
cmake --build . --config Release --target install
popd

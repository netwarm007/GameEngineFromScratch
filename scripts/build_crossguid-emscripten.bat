@echo off
git submodule update --init External/src/crossguid
mkdir External\build\crossguid
pushd External\build\crossguid
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=../../../cmake/Emscripten.cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "NMake Makefiles" ..\..\src\crossguid
cmake --build . --config Release --target install
popd


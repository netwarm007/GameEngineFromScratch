mkdir build
pushd build
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=./cmake/Emscripten.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G "Ninja" ..
cmake --build . --config debug
popd


mkdir build
pushd build
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=./cmake/Emscripten.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G "Ninja" ..
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd


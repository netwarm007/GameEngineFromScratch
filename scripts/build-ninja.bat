mkdir build
pushd build
cmake -G "Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build . --config Debug
popd


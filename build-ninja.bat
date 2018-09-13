mkdir build
pushd build
rm -rf *
cmake -G "Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build . --config debug
popd


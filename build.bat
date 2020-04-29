mkdir build
pushd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G "Visual Studio 16 2019" -A "x64" -Thost=x64 ..
cmake --build . --config Debug
popd


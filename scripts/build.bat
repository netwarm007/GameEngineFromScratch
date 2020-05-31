mkdir build
pushd build
cmake -G "Visual Studio 16 2019" -A "x64" -Thost=x64 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOPENGL_RHI_DEBUG=ON ..
cmake --build . --config Debug
popd


mkdir -p build
pushd build
cmake -G "Visual Studio 15 2017 Win64" ..
cmake --build . --config debug
popd


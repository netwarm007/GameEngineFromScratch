mkdir build
pushd build
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Debug -Thost=x64 ..
cmake --build . --config debug
popd


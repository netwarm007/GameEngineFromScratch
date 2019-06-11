mkdir build
pushd build
cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Debug -Thost=x64 ..
cmake --build . --config debug
popd


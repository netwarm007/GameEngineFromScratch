mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=Debug -Thost=x64 ..
cmake --build . --config debug
popd


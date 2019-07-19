mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=Debug -A Win64 -Thost=x64 ..
cmake --build . --config debug
popd


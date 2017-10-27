mkdir -p build
pushd build
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=..\cmake\orbis-clang.cmake -G "Visual Studio 15 2017 Win64" -Thost=x64 ..
cmake --build . --config debug
popd


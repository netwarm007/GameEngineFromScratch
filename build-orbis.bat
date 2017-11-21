mkdir -p build
pushd build
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=..\cmake\orbis-clang.cmake -G "NMake Makefiles" ..
cmake --build . --config debug
popd


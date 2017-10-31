mkdir -p build
pushd build
rm -rf *
cmake -G "NMake Makefiles" -DCMAKE_TOOLCHAIN_FILE=..\cmake\clang-cl.cmake ..
cmake --build . --config debug
popd


mkdir build
pushd build
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=..\cmake\psp2snc.cmake -G "NMake Makefiles" ..
cmake --build . --config debug
popd


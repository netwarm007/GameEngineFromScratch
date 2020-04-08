mkdir build
pushd build
rm -rf *
set PATH=%PATH%;..\External\Windows\bin
cmake -G "NMake Makefiles" -DCMAKE_TOOLCHAIN_FILE=..\cmake\clang-cl.cmake ..
cmake --build . --config Debug
popd


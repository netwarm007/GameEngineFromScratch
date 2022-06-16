mkdir build
pushd build
rm -rf *
set PATH=%PATH%;..\External\Windows\bin
cmake -G "NMake Makefiles" -DCMAKE_TOOLCHAIN_FILE=..\cmake\clang-cl.cmake ..
if "%1" == "" (cmake --build . --config Debug) else (cmake --build . --config %1)
popd


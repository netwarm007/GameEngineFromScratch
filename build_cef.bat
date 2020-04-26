@echo off
git submodule update --init External\src\cef
mkdir External\build\cef
pushd External\build\cef
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -DCMAKE_INSTALL_RPATH=../../ -DCMAKE_BUILD_TYPE=DEBUG -DCEF_RUNTIME_LIBRARY_FLAG="/MD" -G "Visual Studio 16 2019" -A "x64" ../../src/cef
cmake --build . --config Debug --target install
popd
echo "Completed build of libcef_dll_wrapper"

@echo off
git submodule update --init External\src\cef
mkdir External\build\cef
pushd External\build\cef
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -DCMAKE_INSTALL_RPATH=../../ -DCMAKE_BUILD_TYPE=DEBUG -DCEF_RUNTIME_LIBRARY_FLAG="/MD" -G "Visual Studio 15 Win64" ../../src/cef
cmake --build . --target install
popd
echo "Completed build of libcef_dll_wrapper"

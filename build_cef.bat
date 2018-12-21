@echo off
git submodule update --init External\src\cef
mkdir External\build\cef
pushd External\build\cef
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -DCMAKE_INSTALL_RPATH=../../ -DCMAKE_BUILD_TYPE=RELEASE ../../src/cef
cmake --build . --config release --target install
popd
echo "Completed build of libcef_dll_wrapper"

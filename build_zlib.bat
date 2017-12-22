@echo off
git submodule update --init External/src/zlib
mkdir External\build\zlib
pushd External\build\zlib
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -G "Visual Studio 15 2017 Win64" -Thost=x64 ../../src/zlib
cmake --build . --config release --target install
popd



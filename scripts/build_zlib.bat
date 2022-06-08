@echo off
git submodule update --init External/src/zlib
mkdir External\build\zlib
pushd External\build\zlib
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -G "Visual Studio 17 2022" -A "x64" -DBUILD_SHARED_LIBS=OFF -Thost=x64 ../../src/zlib
cmake --build . --config Release --target install
popd


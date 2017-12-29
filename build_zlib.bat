@echo off
git submodule update --init External/src/zlib
mkdir -p External\build\zlib
pushd External\build\zlib
cmake -DCMAKE_INSTALL_PREFIX=../Windows ../../../External/src/zlib
cmake --build . --config release --target install
popd


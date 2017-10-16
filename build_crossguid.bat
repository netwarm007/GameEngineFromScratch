@echo off
git submodule update --init External/src/crossguid
mkdir -p External\build\crossguid
pushd External\build\crossguid
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "NMake Makefiles" ../../src/crossguid
cmake --build . --config release --target install
popd


@echo off
git submodule update --init External/src/crossguid
mkdir External\build\crossguid
pushd External\build\crossguid
rm -rf *
cmake -DCMAKE_TOOLCHAIN_FILE=../../../cmake/psp2snc.cmake -DCMAKE_INSTALL_PREFIX=../../ -G "NMake Makefiles" ../../src/crossguid
cmake --build . --config relwithdebuginfo --target install
popd


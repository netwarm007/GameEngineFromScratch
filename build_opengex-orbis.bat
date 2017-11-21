@echo off
git submodule update --init External\src\opengex
mkdir External\build\opengex
pushd External\build\opengex
rm -rf *
cmake  -DCMAKE_TOOLCHAIN_FILE=..\..\..\cmake\orbis-clang.cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "NMake Makefiles" ..\..\src\opengex
cmake --build . --config debug --target install
popd


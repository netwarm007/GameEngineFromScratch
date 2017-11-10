@echo off
git submodule update --init External\src\opengex
mkdir External\build\opengex
pushd External\build\opengex
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows -G "Visual Studio 15 2017 Win64" -Thost=x64 ..\..\src\opengex
cmake --build . --config debug --target install
popd


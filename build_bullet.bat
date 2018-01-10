@echo off
git submodule update --init External/src/bullet
mkdir External\build\bullet
pushd External\build\bullet
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -G "Visual Studio 15 2017 Win64" -Thost=x64 ../../src/bullet
cmake --build . --config debug --target install
popd



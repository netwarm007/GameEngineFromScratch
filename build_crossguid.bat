@echo off
git submodule update --init External/src/crossguid
mkdir External\build\crossguid
pushd External\build\crossguid
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "Visual Studio 15 2017 Win64" -Thost=x64 ..\..\src\crossguid
cmake --build . --config debug --target install
popd


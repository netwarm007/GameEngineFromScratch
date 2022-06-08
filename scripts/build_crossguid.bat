@echo off
git submodule update --init External/src/crossguid
mkdir External\build\crossguid
pushd External\build\crossguid
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=..\..\ -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ..\..\src\crossguid
cmake --build . --config Debug --target install
popd


@echo off
git submodule update --init External\src\flatbuffers
mkdir External\build\flatbuffers
pushd External\build\flatbuffers
rmdir /s
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -G "Visual Studio 16 2019" -A "x64" -Thost=x64 ..\..\src\flatbuffers
cmake --build . --config Debug --target install
popd

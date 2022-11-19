@echo off
git submodule update --init External\src\freealut
mkdir External\build\freealut
pushd External\build\freealut
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -G "Visual Studio 16 2019" -A "x64" -Thost=x64 -DBUILD_TESTS=OFF -DBUILD_STATIC=ON -DBUILD_EXAMPLES=OFF ..\..\src\freealut
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd

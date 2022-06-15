@echo off
git submodule update --init External\src\bullet
mkdir External\build\bullet
pushd External\build\bullet
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -DCMAKE_INSTALL_RPATH=..\..\Windows\ -DINSTALL_LIBS=ON -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -G "Visual Studio 17 2022" -A "x64" -Thost=x64 ..\..\src\bullet
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd

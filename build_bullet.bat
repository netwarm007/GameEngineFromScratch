@echo off
git submodule update --init External\src\bullet
mkdir External\build\bullet
pushd External\build\bullet
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=..\..\Windows\ -DCMAKE_INSTALL_RPATH=..\..\Windows\ -DINSTALL_LIBS=ON -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -G "Visual Studio 15 2017 Win64" -Thost=x64 ..\..\src\bullet
cmake --build . --config debug --target install
popd



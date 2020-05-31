@echo off
git submodule update --init External\src\bullet
mkdir External\build\bullet
pushd External\build\bullet
cmake -DCMAKE_CROSSCOMPILING_EMULATOR="node.exe" -DCMAKE_TOOLCHAIN_FILE=../../../cmake/Emscripten.cmake -DCMAKE_INSTALL_PREFIX=..\..\Emscripten\ -DCMAKE_INSTALL_RPATH=..\..\Emscripten\ -DINSTALL_LIBS=ON -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_CPU_DEMOS=OFF -DBUILD_PYBULLET=OFF -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXTRAS=OFF -DBUILD_UNIT_TESTS=OFF -G "NMake Makefiles" ..\..\src\bullet
cmake --build . --config Release --target install
popd



@echo off
git submodule update --init External/src/glfw
mkdir External\build\glfw
pushd External\build\glfw
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF -DGLFW_BUILD_EXAMPLES=OFF -G "Visual Studio 17 2022" -A "x64" ../../src/glfw
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
popd

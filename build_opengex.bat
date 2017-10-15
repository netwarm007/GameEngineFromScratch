@echo off
git submodule update --init External/src/opengex
mkdir -p External\build\opengex
cd External\build\opengex
cmake -DCMAKE_INSTALL_PREFIX=../../Windows ../..External//src/opengex
cmake --build . --config release --target install


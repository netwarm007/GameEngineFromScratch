git submodule update --init External/src/ispc
mkdir External\build\ispc
pushd External\build\ispc
cmake -DLLVM_INSTALL_DIR=../../Windows -DLLVM_HOME=../../Windows -DARM_ENABLED=ON -DNVPTX_ENABLED=OFF -DISPC_INCLUDE_EXAMPLES=OFF -DISPC_INCLUDE_TESTS=OFF -DCMAKE_INSTALL_PREFIX=../../Windows  -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 15 2017 Win64" -Thost=x64 ../../src/ispc
cmake --build . --config release --target install
popd


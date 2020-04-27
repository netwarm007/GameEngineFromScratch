git submodule update --init External/src/llvm
mkdir External\build\llvm
pushd External\build\llvm
cmake -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON -DCMAKE_INSTALL_PREFIX=../../Windows  -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;ARM;X86;AArch64;NVPTX" -DLLVM_ENABLE_PROJECTS="clang" -G "Visual Studio 16 2019" -A "x64" -Thost=x64 ../../src/llvm/llvm
cmake --build . --config Release --target install
popd


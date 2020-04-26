git submodule update --init External/src/llvm External/src/clang External/src/libcxx External/src/libcxxabi
mkdir External\build\llvm
pushd External\build\llvm
cmake -DLLVM_ENABLE_DUMP=ON -DCMAKE_INSTALL_PREFIX=../../Windows  -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;ARM;X86;AArch64;NVPTX" -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi" -G "Visual Studio 16 2019" -Thost=x64 ../../src/llvm
cmake --build . --config Release --target install
make install-cxx
make install-cxx-abi
popd


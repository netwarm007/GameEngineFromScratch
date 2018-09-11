git submodule update --init External/src/llvm External/src/clang External/src/libcxx External/src/libcxxabi
mkdir External\build\llvm
cd External\build\llvm
cmake -DCMAKE_INSTALL_PREFIX=../../Windows  -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;ARM;X86" -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi" -DLLVM_PARALLEL_COMPILE_JOBS=8 -G "Visual Studio 15 2017 Win64" -Thost=x64 ../../src/llvm
cmake --build . --config release --target install


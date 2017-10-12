git submodule update --init
mkdir -p build\llvm
cd build\llvm
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -DLLVM_LIBDIR_SUFFIX=64 -DLLVM_ENABLE_PROJECTS=clang -G "Visual Studio 15 2017 Win64" ../../src/llvm
cmake --build . --config release --target install


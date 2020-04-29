git submodule update --init External/src/ispc
mkdir External\build\ispc
pushd External\build\ispc
set INSTALL_ROOT=..\..\Windows
set LLVM_HOME=..\..\src\llvm
set LLVM_VERSION=LLVM_10_0
set ISPC_HOME=..\..\src\ispc
set PATH=%INSTALL_ROOT%\bin;%PATH%
cmake -G "Visual Studio 16 2019" -DLLVM_INSTALL_DIR=%INSTALL_ROOT% -DCMAKE_INSTALL_PREFIX=%INSTALL_ROOT% -DARM_ENABLED=ON -DNVPTX_ENABLED=OFF -DISPC_INCLUDE_EXAMPLES=OFF %ISPC_HOME%
cmake --build . --config Release --target install
popd


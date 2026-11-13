setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
git submodule update --init External/src/ispc
mkdir External\build\ispc
pushd External\build\ispc
set INSTALL_ROOT=..\..\Windows
set LLVM_HOME=..\..\src\llvm
set LLVM_VERSION=LLVM_15_0
set ISPC_HOME=..\..\src\ispc
set PATH=%INSTALL_ROOT%\bin;%PATH%
cmake -G "%GEFS_VS_GENERATOR%" -DLLVM_INSTALL_DIR=%INSTALL_ROOT% -DCMAKE_INSTALL_PREFIX=%INSTALL_ROOT% -DARM_ENABLED=ON -DNVPTX_ENABLED=OFF -DISPC_INCLUDE_EXAMPLES=OFF %ISPC_HOME%
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug --target install) else (cmake --build . --config %1 --target install)
if errorlevel 1 (popd & exit /b 1)
popd


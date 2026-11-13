setlocal
call "%~dp0select-vs-generator.bat"
if errorlevel 1 exit /b 1
mkdir build
pushd build
cmake -G "%GEFS_VS_GENERATOR%" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOPENGL_RHI_DEBUG=ON -DD3D12_RHI_DEBUG=ON ..
if errorlevel 1 (popd & exit /b 1)
if "%1" == "" (cmake --build . --config Debug) else (cmake --build . --config %1)
if errorlevel 1 (popd & exit /b 1)
popd


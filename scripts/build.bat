mkdir build
pushd build
cmake -G "Visual Studio 17 2022" -A "x64" -Thost=x64 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOPENGL_RHI_DEBUG=ON -DD3D12_RHI_DEBUG=ON ..
if "%1" == "" (cmake --build . --config Debug) else (cmake --build . --config %1)
popd


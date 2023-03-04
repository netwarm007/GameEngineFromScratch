mkdir build
pushd build
cmake -G "NMake Makefiles" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOPENGL_RHI_DEBUG=ON -DD3D12_RHI_DEBUG=ON ..
if "%1" == "" (cmake --build . --config Debug) else (cmake --build . --config %1)
popd


mkdir build
pushd build
cmake -G "Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
if "%1" == "" (cmake --build . --config Debug) else (cmake --build . --config %1)
popd


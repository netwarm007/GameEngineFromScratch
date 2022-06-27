mkdir build\Asset\Materials
pushd build\Asset\Materials
if "%1" == "" (..\..\Utility\Debug\MaterialBaker.exe) else (..\..\Utility\%1\MaterialBaker.exe)
popd
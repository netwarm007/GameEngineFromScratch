mkdir build\Asset\Materials
pushd build\Asset\Materials
..\..\Utility\Debug\MaterialBaker.exe
if "%1" == "" (..\..\Utility\Debug\MaterialBaker.exe) else (..\..\Utility\%1\MaterialBaker.exe)
popd
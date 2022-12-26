mkdir build\Asset\Materials
pushd build\Asset\Materials
if "%1" == "" (..\..\Utility\MaterialBaker.exe) else (..\..\Utility\MaterialBaker.exe)
popd
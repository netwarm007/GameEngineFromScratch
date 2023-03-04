mkdir build\Asset\Materials
pushd build\Asset\Materials
if exist ..\..\Utility\Debug\MaterialBaker.exe (
    ..\..\Utility\Debug\MaterialBaker.exe
) else if exist ..\..\Utility\Release\MaterialBaker.exe (
    ..\..\Utility\Release\MaterialBaker.exe
) else (
    ..\..\Utility\MaterialBaker.exe
)
popd
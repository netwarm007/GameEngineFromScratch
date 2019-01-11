mkdir build
pushd build
rm -rf *
emcmake cmake -G "Visual Studio 15 2017 Win64" -Thost=x64 ..
cmake --build . --config debug
popd


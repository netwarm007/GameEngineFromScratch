@echo off
git submodule update --init External/src/crossguid
mkdir External\build\crossguid
pushd External\build\crossguid
<<<<<<< HEAD
del /Q *
cmake -DCMAKE_INSTALL_PREFIX=../../Windows -G "NMake Makefiles" ../../src/crossguid
cmake --build . --config debug --target install
=======
rm -rf *
cmake -DCMAKE_INSTALL_PREFIX=../../ -G "NMake Makefiles" ../../src/crossguid
cmake --build . --config relwithdebuginfo --target install
>>>>>>> article_31
popd


#!/bin/bash
set -e
git submodule update --init External/src/imgui-node-editor
# source only module, no need to build
mkdir -p External/build/imgui-node-editor
cd External/build/imgui-node-editor
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=../../`uname -s` ../../src/imgui-node-editor/examples
if [[ -z $1 ]];
then
    cmake --build . --config debug 
else
    cmake --build . --config $1 
fi
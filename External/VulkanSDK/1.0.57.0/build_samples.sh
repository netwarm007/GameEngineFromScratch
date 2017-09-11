#!/bin/bash

set -ex

# Build sample dependencies.
pushd source/glslang
cmake -H. -Bbuild
make -j`nproc` -C build
mkdir -p ../../x86_64/lib/glslang
cp build/SPIRV/libSPIRV.a ../../x86_64/lib/glslang
cp build/SPIRV/libSPVRemapper.a ../../x86_64/lib/glslang
cp build/glslang/libglslang.a ../../x86_64/lib/glslang
cp build/glslang/OSDependent/Unix/libOSDependent.a ../../x86_64/lib/glslang
cp build/OGLCompilersDLL/libOGLCompiler.a ../../x86_64/lib/glslang
cp build/hlsl/libHLSL.a ../../x86_64/lib/glslang
popd

# Build the samples.
pushd samples
cmake -H. -Bbuild
make -j`nproc` -C build
popd

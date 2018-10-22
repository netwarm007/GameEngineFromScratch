#!/bin/bash
set -e
echo "concat source files"
case $2 in
    vs)
        ext=vert
        ;;
    ps)
        ext=frag
        ;;
    gs)
        ext=geom
        ;;
    cs)
        ext=comp
        ;;
    tesc)
        ext=tesc
        ;;
    tese)
        ext=tese
        ;;
esac
InputFile=Asset/Shaders/$1_$2.glsl
if [ -e $InputFile ]; then 
    cat Asset/Shaders/cbuffer.glsl Asset/Shaders/functions.glsl $InputFile > Asset/Shaders/Vulkan/$1.$ext
    echo "Vulkan GLSL --> SPIR-V"
    External/`uname -s`/bin/glslangValidator -H -o Asset/Shaders/Vulkan/$1_$2.spv Asset/Shaders/Vulkan/$1.$ext
    echo "SPIR-V --> Desktop GLSL"
    External/`uname -s`/bin/SPIRV-Cross --version 400 --remove-unused-variables --no-420pack-extension --output Asset/Shaders/OpenGL/$1_$2.glsl Asset/Shaders/Vulkan/$1_$2.spv
    echo "SPIR-V --> Embeded GLSL"
    External/`uname -s`/bin/SPIRV-Cross --version 310 --es --remove-unused-variables --output Asset/Shaders/OpenGLES/$1_$2.glsl Asset/Shaders/Vulkan/$1_$2.spv
    echo "SPIR-V --> HLSL"
    External/`uname -s`/bin/SPIRV-Cross --hlsl --shader-model 52 --remove-unused-variables --output Asset/Shaders/HLSL/$1_$2.hlsl Asset/Shaders/Vulkan/$1_$2.spv
    echo "SPIR-V --> Metal"
    External/`uname -s`/bin/SPIRV-Cross --msl --msl-version 020101 --remove-unused-variables --rename-entry-point main $1_$2_main $ext --output Asset/Shaders/Metal/$1_$2.metal Asset/Shaders/Vulkan/$1_$2.spv
    if [ $2 = cs ]; then
        External/`uname -s`/bin/SPIRV-Cross --ispc --remove-unused-variables --output Asset/Shaders/ISPC/$1.ispc Asset/Shaders/Vulkan/$1_$2.spv
    fi
fi
echo "Finished"


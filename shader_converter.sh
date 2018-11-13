#!/bin/bash
set -e
InputFile=Asset/Shaders/$1.$2.glsl
if [ -e $InputFile ]; then 
    echo "Vulkan GLSL --> SPIR-V"
    cat Asset/Shaders/{version.h.glsl,const.h.glsl,cbuffer.h.glsl,functions.glsl} $InputFile | \
    External/`uname -s`/bin/glslangValidator -H --stdin -S $2 --amb -o Asset/Shaders/Vulkan/$1.$2.spv  
    echo "SPIR-V --> Desktop GLSL"
    External/`uname -s`/bin/SPIRV-Cross --version 400 --remove-unused-variables --no-420pack-extension --output Asset/Shaders/OpenGL/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> Embeded GLSL"
    External/`uname -s`/bin/SPIRV-Cross --version 310 --es --remove-unused-variables --output Asset/Shaders/OpenGLES/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> Metal"
    External/`uname -s`/bin/SPIRV-Cross --msl --msl-version 020101 --rename-entry-point main $1_$2_main $2 --remove-unused-variables --output Asset/Shaders/Metal/$1.$2.metal Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> HLSL"
    External/`uname -s`/bin/SPIRV-Cross --hlsl --shader-model 52 --remove-unused-variables --output Asset/Shaders/HLSL/$1.$2.hlsl Asset/Shaders/Vulkan/$1.$2.spv
    if [ $2 = comp ]; then
        echo "SPIR-V --> ISPC"
        External/`uname -s`/bin/SPIRV-Cross --ispc --remove-unused-variables --output Asset/Shaders/ISPC/$1.ispc Asset/Shaders/Vulkan/$1.$2.spv
    fi
fi
echo "Finished"


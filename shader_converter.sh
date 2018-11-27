#!/bin/bash
set -e
InputFile=Asset/Shaders/HLSL/$1.$2.hlsl
if [ -e $InputFile ]; then 
    echo "HLSL --> SPIR-V"
    External/`uname -s`/bin/glslangValidator -H -IFramework/Common -o Asset/Shaders/Vulkan/$1.$2.spv -e $1_$2_main $InputFile  
    echo "SPIR-V --> Desktop GLSL"
    External/`uname -s`/bin/SPIRV-Cross --version 420 --remove-unused-variables --output Asset/Shaders/OpenGL/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> Embeded GLSL"
    External/`uname -s`/bin/SPIRV-Cross --version 310 --es --remove-unused-variables --output Asset/Shaders/OpenGLES/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> Metal"
    External/`uname -s`/bin/SPIRV-Cross --msl --msl-version 020101 --remove-unused-variables --output Asset/Shaders/Metal/$1.$2.metal Asset/Shaders/Vulkan/$1.$2.spv
    if [ $2 = comp ]; then
        echo "SPIR-V --> ISPC"
        External/`uname -s`/bin/SPIRV-Cross --ispc --remove-unused-variables --output Asset/Shaders/ISPC/$1.ispc Asset/Shaders/Vulkan/$1.$2.spv
    fi
fi
echo "Finished"


#!/bin/bash
set -e
InputFile=Asset/Shaders/HLSL/$1.$2.hlsl
if [ -e $InputFile ]; then 
    echo "HLSL --> SPIR-V"
    External/`uname -s`/bin/glslangValidator -H -I. -I./Framework/Common -DOS_WEBASSEMBLY -o Asset/Shaders/Vulkan/$1.$2.spv -e $1_$2_main $InputFile  
    echo "SPIR-V --> WebGL2 GLSL"
    External/`uname -s`/bin/spirv-cross --version 300 --es --remove-unused-variables --output Asset/Shaders/WebGL/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv

    echo "HLSL --> SPIR-V"
    External/`uname -s`/bin/glslangValidator -H -I. -I./Framework/Common -o Asset/Shaders/Vulkan/$1.$2.spv -e $1_$2_main $InputFile  
    echo "SPIR-V --> Desktop GLSL"
    External/`uname -s`/bin/spirv-cross --version 420 --remove-unused-variables --output Asset/Shaders/OpenGL/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> Embeded GLSL"
    External/`uname -s`/bin/spirv-cross --version 320 --es --remove-unused-variables --output Asset/Shaders/OpenGLES/$1.$2.glsl Asset/Shaders/Vulkan/$1.$2.spv
    echo "SPIR-V --> Metal"
    External/`uname -s`/bin/spirv-cross --msl --msl-version 020101 --remove-unused-variables --output Asset/Shaders/Metal/$1.$2.metal Asset/Shaders/Vulkan/$1.$2.spv
    if [ $2 = comp ]; then
        echo "SPIR-V --> ISPC"
        External/`uname -s`/bin/spirv-cross --ispc --remove-unused-variables --output Asset/Shaders/ISPC/$1.ispc Asset/Shaders/Vulkan/$1.$2.spv
    fi
fi
echo "Finished"


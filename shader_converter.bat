@ECHO OFF
echo "HLSL --> SPIR-V"
External\Windows\bin\glslangValidator.exe -H -I. -IFramework\Common -DOS_WEBASSEMBLY -o Asset\Shaders\Vulkan\%1.%2.spv -e %1_%2_main Asset\Shaders\HLSL\%1.%2.hlsl
echo "SPIR-V --> WebGL2 GLSL"
External\Windows\bin\SPIRV-Cross.exe --version 300 --es --remove-unused-variables --output Asset\Shaders\WebGL\%1.%2.glsl Asset\Shaders\Vulkan\%1.%2.spv

echo "HLSL --> SPIR-V"
External\Windows\bin\glslangValidator.exe -H -I. -IFramework\Common -o Asset\Shaders\Vulkan\%1.%2.spv -e %1_%2_main Asset\Shaders\HLSL\%1.%2.hlsl
echo "SPIR-V --> Desktop GLSL"
External\Windows\bin\SPIRV-Cross.exe --version 420 --remove-unused-variables --output Asset\Shaders\OpenGL\%1.%2.glsl Asset\Shaders\Vulkan\%1.%2.spv
echo "SPIR-V --> Embeded GLSL"
External\Windows\bin\SPIRV-Cross.exe --version 320 --es --remove-unused-variables --output Asset\Shaders\OpenGLES\%1.%2.glsl Asset\Shaders\Vulkan\%1.%2.spv
echo "SPIR-V --> Metal"
External\Windows\bin\SPIRV-Cross --msl --msl-version 020101 --remove-unused-variables --output Asset\Shaders\Metal\%1.%2.metal Asset\Shaders\Vulkan\%1.%2.spv
if "%2"=="cs" (
echo "SPIR-V --> ISPC"
External\Windows\bin\SPIRV-Cross.exe --ispc --output Asset\Shaders\ISPC\%1.ispc Asset\Shaders\Vulkan\%1.%2.spv
echo "Finished"
ENDLOCAL

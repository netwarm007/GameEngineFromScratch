@ECHO OFF
echo "concat source files"
SETLOCAL
if "%2"=="vs" (set ext=vert)
if "%2"=="ps" (set ext=frag)
if "%2"=="gs" (set ext=geom)
if "%2"=="cs" (set ext=comp)
set InputFile=Asset\Shaders\%1_%2.glsl
if not exist %InputFile% (
echo "cannot find file %InputFile%"
exit /b
)
cat Asset\Shaders\cbuffer.glsl Asset\Shaders\functions.glsl %InputFile% > Asset\Shaders\Vulkan\%1.%ext%
echo "Vulkan GLSL --> SPIR-V"
External\Windows\bin\glslangValidator.exe -H -o Asset\Shaders\Vulkan\%1_%2.spv Asset\Shaders\Vulkan\%1.%ext%
echo "SPIR-V --> Desktop GLSL"
External\Windows\bin\SPIRV-Cross.exe --version 400 --no-420pack-extension --output Asset\Shaders\OpenGL\%1_%2.glsl Asset\Shaders\Vulkan\%1_%2.spv
echo "SPIR-V --> Embeded GLSL"
External\Windows\bin\SPIRV-Cross.exe --version 310 --es --output Asset\Shaders\OpenGLES\%1_%2.glsl Asset\Shaders\Vulkan\%1_%2.spv
if "%2"=="cs" (
echo "SPIR-V --> ISPC"
External\Windows\bin\SPIRV-Cross.exe --ispc --output Asset\Shaders\ISPC\%1.ispc Asset\Shaders\Vulkan\%1_%2.spv
)
echo "Finished"
ENDLOCAL

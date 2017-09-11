# Build Targets
The build target (i.e. program name) for all samples is the base source
filename of the sample without the .cpp suffix.

All sample programs are linked with the Vulkan loader and samples utility
library.

The Vulkan Samples Kit currently supports the following types of build targets:  
  - single file sample, no shaders  
  - single file sample, inline glsl shaders  
  - single file sample, external spirv shaders  
  - single file sample, external glsl shaders (with auto-convert to spirv)  

CMake functions are provided for each of the above types of build targets.
Certain variables must be set before invoking the functions, one of which is
VULKAN_VERSION.  This variable must be set to the version prefix used in both
the file name and data subdirectory name.

## Single File, No Shaders
The `sampleWithSingleFile` function will build all samples identified in the
S_TARGETS variable.

To add a single file sample to the build, include the base name (no prefix or
.cpp) of the file in the `S_TARGETS` list.

```
set(VULKAN_VERSION vk0.10)
set (S_TARGETS instance device)
sampleWithSingleFile()
```

## Single File, External SPIRV Shaders
Samples with dependencies on SPIRV shaders are built with the
`sampleWithSPIRVShaders` function.

To add a sample with SPIRV shader dependencies to the build, identify the list
of SPIRV shader file names (no .spv suffix) to the SAMPLE_SPIRV_SHADERS
variable.

```
set(VULKAN_VERSION vk0.10)
set(SAMPLE_SPIRV_SHADERS spirvshader-vert spirvshader-frag)
sampleWithSPIRVShaders(usespirvshader)
```

## Single File, External GLSL Shaders
Samples with dependencies on external GLSL shaders that must be converted to
SPIRV are built with the `sampleWithGLSLShaders` function.  This function will
convert the GLSL vertex and fragment shaders to the SPIRV equivalents that are
used in the sample program.

To add a sample with GLSL shader dependencies to the build, identify the list
of GLSL shader file names (no suffixes) of the specific type to the
appropriate SAMPLE_GLSL_VERT_SHADERS and SAMPLE_GLSL_FRAG_SHADERS variables.

```
set(VULKAN_VERSION vk0.10)
set(SAMPLE_GLSL_FRAG_SHADERS glslshader)
set(SAMPLE_GLSL_VERT_SHADERS glslshader)
sampleWithGLSLShaders(useglslshader)
```

In the example above, the files glslshader.vert and glslshader.frag reside in
$VULKAN_SAMPLES/data/vk0.10.  These GLSL shaders are converted to SPIRV via
glslangValidator, with the resulting files named glslshader-frag.spv and
glslshader-vert.spv, residing in the same directory as the originals.

## TODO
- samples with more than one source file
- support other variations and types of shaders (TBD)


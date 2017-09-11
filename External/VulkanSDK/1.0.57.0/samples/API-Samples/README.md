# Code and Other Conventions for Vulkan Samples

  - The Vulkan version to which the sample is being written must be the prefix
    in the sample code file name.  For example, vk0.10-SAMPLE.cpp.

  - A sample should highlight a specific Vulkan entry point or capability
    which must be identified in the name of the sample source file.  For
    example, vk0.10-device-info.cpp.

  - Use the data/ (Vulkan-release-independent) and data/VERSION/
    (Vulkan-release-dependent) directories for uniquely-named data, shaders,
    images, etc. used by your sample.  VERSION must match the version prefix
    used in the sample file name.  For example, a shader file for
    $VULKAN_SAMPLES/src/vk0.10-imageformat.cpp could be
    $VULKAN_SAMPLES/data/vk0.10/imgf-texture-2d.spv. 

  - Each sample should include a comment of the following form used for auto-
    extraction of a short one-line description of the sample:

```
/*
VULKAN_SAMPLE_SHORT_DESCRIPTION
short description of sample
*/
```
  - Each sample may include a comment of the following form used for auto-
    extraction of a more detailed, multi-line description of the sample:

```
/*
VULKAN_SAMPLE_DESCRIPTION_START
short description of sample
continued here
and here
VULKAN_SAMPLE_DESCRIPTION_END
*/
```

  - For easier navigation to the relevant section(s) of code in the source 
    file, where applicable, the sections should be identified by the following
    comment header/footer:

```
/* VULKAN_KEY_START */
/* VULKAN_KEY_END */
```


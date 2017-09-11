# Shaders

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `11-init_shaders.cpp`

## Compiling GLSL Shaders into SPIR-V

The low-level shader code representation for Vulkan is SPIR-V.
The sample programs compile shader programs written in GLSL
into SPIR-V with a call to a utility function:

    GLSLtoSPV(VK_SHADER_STAGE_VERTEX_BIT, vertShaderText, vtx_spv);

The shader source code is in the `vertShaderText` variable and the
compiled SPIR-V is returned in `vtx_spv`, which is a vector of
`unsigned int` and is suitable for storing the SPIR-V code.

Look at the sample code to find the shader source for this vertex
shader and notice that the fragment shader source is provided as well,
along with a similar call to compile it.

Also notice that these are simple shaders.
The vertex shader simply passes the color through to its output
and transforms the incoming position with the MVP transform that
we saw in previous sections.
The fragment shader is even simpler and just passes the color through.

In this simple sample, there are only two shader stages: the vertex and fragment
stages, stored in that order in `info.shaderStages`.

## Creating Vulkan Shader Modules

The compiled shader code is given to Vulkan by creating a `VkShaderModule` and
storing it in a `VkPipelineShaderStageCreateInfo` structure that is used in another
sample later as part of creating the overall graphics pipeline.

    VkShaderModuleCreateInfo moduleCreateInfo;
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.pNext = NULL;
    moduleCreateInfo.flags = 0;
    moduleCreateInfo.codeSize = vtx_spv.size() * sizeof(unsigned int);
    moduleCreateInfo.pCode = vtx_spv.data();
    res = vkCreateShaderModule(info.device, &moduleCreateInfo, NULL,
                               &info.shaderStages[0].module);

Note that the code resulting from the GLSL to SPIR-V conversion is used to
create the shader module.
The same procedure is used to create a `vkShaderModule` for the fragment
shader, which is stored in `info.shaderStages[1].module`.

Some additional initialization of the creation info for the pipeline
shader stage is also performed at this time:

    info.shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.shaderStages[0].pNext = NULL;
    info.shaderStages[0].pSpecializationInfo = NULL;
    info.shaderStages[0].flags = 0;
    info.shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    info.shaderStages[0].pName = "main";

At this point, the shaders are ready to go.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%"><a href="10-init_render_pass.html" title="Prev">Render Pass</a></td>
        <td align="center" width="33%"><a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%"><a href="12-init_frame_buffers.html" title="Next">Framebuffers</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>

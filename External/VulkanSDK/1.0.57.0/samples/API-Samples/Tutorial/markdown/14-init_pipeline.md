# Create a Graphics Pipeline

<link href="../css/lg_stylesheet.css" rel="stylesheet"></link>

Code file for this section is `14-init_pipeline.cpp`

You are getting closer to pulling it all together enough to draw a cube!
The next step is to configure the GPU for rendering by setting up a
graphics pipeline.

A graphics pipeline consists of shader stages, a pipeline layout,
a render pass, and fixed-function pipeline stages.
You defined shader stages and pipeline layouts in earlier sections.
Here, you will configure the remaining fixed-function pipeline stages.
This involves filling in some "create info" data structures for
creating the pipeline.
Most of the work performed here configures the per-fragment operations,
just before the fragments are placed in the framebuffer.

Here's a diagram showing the big picture:

![Graphics Pipeline](../images/GraphicsPipeline.png)

The next step is to configure the pipeline state objects, represented by the stack
of grey boxes on the lower right.
The final step is to connect the other objects pointed to from the purple pipeline box
on the upper left in order to complete the definition of your graphics pipeline.

## Dynamic State

A dynamic pipeline state is a state that can be changed by a command buffer command
during the execution of a command buffer.
Advance notification of what states are dynamic during command buffer execution may
be useful for a driver as it sets up the GPU for command buffer execution.

The sample provides a list of states that it intends to change during
command buffer execution.
Here, the code starts out by setting up a list of dynamic states and starts out with
them all disabled.

    VkDynamicState dynamicStateEnables[VK_DYNAMIC_STATE_RANGE_SIZE];
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.pNext = NULL;
    dynamicState.pDynamicStates = dynamicStateEnables;
    dynamicState.dynamicStateCount = 0;

Later, the sample indicates that it intends to change some of the states dynamically
with a command buffer command, so it will later change the `dynamicStateEnables`
array, when it configures the viewport and scissors rectangles.
The code to modify the `dynamicStateEnables` is kept with the viewport and scissors
configuration code below for clarity.

## Pipeline Vertex Input State

You already initialized the vertex input state when you created the vertex buffer because
it was straightforward to do it at that time.
The input state includes the format and arrangement of the vertex data.
You can review the vertexbuffer sample to see how the `vi_binding` and
`vi_attribs` variables are set.

    VkPipelineVertexInputStateCreateInfo vi;
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.pNext = NULL;
    vi.flags = 0;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &info.vi_binding;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions = info.vi_attribs;

## Pipeline Vertex Input Assembly State

The input assembly state is basically where you declare how your vertices form
the geometry you want to draw.
For example, your vertices may be intended to form a triangle strip or
triangle fan.
Here, we are just using a list of triangles, where every three vertices
describe a triangle:

    VkPipelineInputAssemblyStateCreateInfo ia;
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.pNext = NULL;
    ia.flags = 0;
    ia.primitiveRestartEnable = VK_FALSE;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

## Pipeline Rasterization State

The next data structure configures the rasterization operations in the GPU.

    VkPipelineRasterizationStateCreateInfo rs;
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.pNext = NULL;
    rs.flags = 0;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.depthClampEnable = VK_TRUE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.depthBiasEnable = VK_FALSE;
    rs.depthBiasConstantFactor = 0;
    rs.depthBiasClamp = 0;
    rs.depthBiasSlopeFactor = 0;
    rs.lineWidth = 1.0f;

These fields are set with fairly common values that you would expect for
our straightforward cube rendering sample.
You may recognize the correlation between the `frontFace` member and
the GL function `glFrontFace()`.

## Pipeline Color Blend State

Blending is another "end of the fixed pipe" operation that you configure here
to do simple replacement of pixels in the destination:

    VkPipelineColorBlendStateCreateInfo cb;
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.pNext = NULL;
    cb.flags = 0;
    VkPipelineColorBlendAttachmentState att_state[1];
    att_state[0].colorWriteMask = 0xf;
    att_state[0].blendEnable = VK_FALSE;
    att_state[0].alphaBlendOp = VK_BLEND_OP_ADD;
    att_state[0].colorBlendOp = VK_BLEND_OP_ADD;
    att_state[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    att_state[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    att_state[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    att_state[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cb.attachmentCount = 1;
    cb.pAttachments = att_state;
    cb.logicOpEnable = VK_FALSE;
    cb.logicOp = VK_LOGIC_OP_NO_OP;
    cb.blendConstants[0] = 1.0f;
    cb.blendConstants[1] = 1.0f;
    cb.blendConstants[2] = 1.0f;
    cb.blendConstants[3] = 1.0f;

Note that some of the configuration info is provided on a per-attachment basis.
You need to have one `VkPipelineColorBlendAttachmentState` for each color attachment
in your pipeline.
In this case, there is only one color attachment.

The `colorWriteMask` selects which of the R, G, B, and/or A components are enabled for writing.
Here, you enable all 4 components.

You disable `blendEnable` which means that the rest of the settings in `att_state[0]`
related to blending do not matter much.

You also disable the pixel-writing logical operation, since this sample just does
a simple replacement when writing pixels to the framebuffer.

The blend constants are used for some of the "blend factors"
(e.g., `VK_BLEND_FACTOR_CONSTANT_COLOR`)
and are just set to something reasonable, as they are not used in this sample.

## Pipeline Viewport State

The draw_cube sample is going to set the viewport and scissors rectangles using
commands in the command buffer.
This code tells the driver that these viewport and scissors states are dynamic and to
ignore the `pViewPorts` and `pScissors` members.

    VkPipelineViewportStateCreateInfo vp = {};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.pNext = NULL;
    vp.flags = 0;
    vp.viewportCount = 1;
    dynamicStateEnables[dynamicState.dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
    vp.scissorCount = 1;
    dynamicStateEnables[dynamicState.dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;
    vp.pScissors = NULL;
    vp.pViewports = NULL;

## Pipeline Depth Stencil State

Continue with the backend fixed-function initialization by setting up the
depth buffer for the commonly used configuration and disable the stencil operations.

    VkPipelineDepthStencilStateCreateInfo ds;
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.pNext = NULL;
    ds.flags = 0;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.minDepthBounds = 0;
    ds.maxDepthBounds = 0;
    ds.stencilTestEnable = VK_FALSE;
    ds.back.failOp = VK_STENCIL_OP_KEEP;
    ds.back.passOp = VK_STENCIL_OP_KEEP;
    ds.back.compareOp = VK_COMPARE_OP_ALWAYS;
    ds.back.compareMask = 0;
    ds.back.reference = 0;
    ds.back.depthFailOp = VK_STENCIL_OP_KEEP;
    ds.back.writeMask = 0;
    ds.front = ds.back;

Since you do want to so depth buffering, you enable depth buffer writing and testing.
In addition, you set the depth buffer compare operation to the commonly used
`VK_COMPARE_OP_LESS_OR_EQUAL`.
Finally, you disable stencil operations since this sample has no need for it.

## Pipeline Multisample State

You're not going to do any fancy multisampling in this sample,
so finish off the pipeline configuration by setting up for no multisampling.

    VkPipelineMultisampleStateCreateInfo ms;
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.pNext = NULL;
    ms.flags = 0;
    ms.pSampleMask = NULL;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.sampleShadingEnable = VK_FALSE;
    ms.alphaToCoverageEnable = VK_FALSE;
    ms.alphaToOneEnable = VK_FALSE;
    ms.minSampleShading = 0.0;

## Pulling It All Together - Create Graphics Pipeline

Finally, you have all the information needed to create the pipeline:

    VkGraphicsPipelineCreateInfo pipeline;
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = NULL;
    pipeline.layout = info.pipeline_layout;
    pipeline.basePipelineHandle = VK_NULL_HANDLE;
    pipeline.basePipelineIndex = 0;
    pipeline.flags = 0;
    pipeline.pVertexInputState = &vi;
    pipeline.pInputAssemblyState = &ia;
    pipeline.pRasterizationState = &rs;
    pipeline.pColorBlendState = &cb;
    pipeline.pTessellationState = NULL;
    pipeline.pMultisampleState = &ms;
    pipeline.pDynamicState = &dynamicState;
    pipeline.pViewportState = &vp;
    pipeline.pDepthStencilState = &ds;
    pipeline.pStages = info.shaderStages;
    pipeline.stageCount = 2;
    pipeline.renderPass = info.render_pass;
    pipeline.subpass = 0;

    res = vkCreateGraphicsPipelines(info.device, NULL, 1,
                                    &pipeline, NULL, &info.pipeline);

The `info.pipeline_layout`, `info.shaderStages`, and `info.render_pass` members
were initialized in previous sections of this tutorial.
The rest of the members in this structure were set up in this section.

With the pipeline created, you are ready to go on to the next section and draw the cube.

<table border="1" width="100%">
    <tr>
        <td align="center" width="33%"><a href="13-init_vertex_buffer.html" title="Prev">Vertex Buffer</a></td>
        <td align="center" width="33%"><a href="index.html" title="Index">Index</a></td>
        <td align="center" width="33%"><a href="15-draw_cube.html" title="Next">Draw Cube</a></td>
    </tr>
</table>
<footer>&copy; Copyright 2016 LunarG, Inc</footer>

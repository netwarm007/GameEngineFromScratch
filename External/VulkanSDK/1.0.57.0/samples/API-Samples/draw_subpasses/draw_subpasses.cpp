/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
VULKAN_SAMPLE_SHORT_DESCRIPTION
Render two multi-subpass render passes with different framebuffer attachments
*/

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include "cube_data.h"

/* For this sample, we'll start with GLSL so the shader function is plain */
/* and then use the glslang GLSLtoSPV utility to convert it to SPIR-V for */
/* the driver.  We do this for clarity rather than using pre-compiled     */
/* SPIR-V                                                                 */

/* This shader renders the cubes in projected space */
static const char *normalVertShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (std140, binding = 0) uniform bufferVals {\n"
    "    mat4 mvp;\n"
    "} myBufferVals;\n"
    "layout (location = 0) in vec4 pos;\n"
    "layout (location = 1) in vec4 inColor;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = inColor;\n"
    "   gl_Position = myBufferVals.mvp * pos;\n"
    "}\n";

/* This shader renders a simple fullscreen quad using the VS alone */
static const char *fullscreenVertShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = vec4(1.0f, 0.1f, 0.1f, 0.5f);\n"
    "   const vec4 verts[4] = vec4[4](vec4(-1.0, -1.0, 0.5, 1.0),\n"
    "                                 vec4( 1.0, -1.0, 0.5, 1.0),\n"
    "                                 vec4(-1.0,  1.0, 0.5, 1.0),\n"
    "                                 vec4( 1.0,  1.0, 0.5, 1.0));\n"
    "\n"
    "   gl_Position = verts[gl_VertexIndex];\n"
    "}\n";

static const char *fragShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (location = 0) in vec4 color;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = color;\n"
    "}\n";

/**
 *  Sample using multiple render passes per framebuffer (different x,y extents)
 *  and multiple subpasses per renderpass.
 */
int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Multi-pass render passes";
    const bool depthPresent = true;

    process_command_line_args(info, argc, argv);
    init_global_layer_properties(info);
    init_instance_extension_names(info);
    init_device_extension_names(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_window_size(info, 500, 500);
    init_connection(info);
    init_window(info);
    init_swapchain_extension(info);
    init_device(info);
    init_command_pool(info);
    init_command_buffer(info);
    execute_begin_command_buffer(info);
    init_device_queue(info);
    init_swap_chain(info);

    info.depth.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    init_depth_buffer(info);

    init_uniform_buffer(info);
    init_descriptor_and_pipeline_layouts(info, false);
    init_vertex_buffer(info, g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data),
                       sizeof(g_vb_solid_face_colors_Data[0]), false);
    init_descriptor_pool(info, false);
    init_descriptor_set(info, false);
    init_pipeline_cache(info);

    /* VULKAN_KEY_START */

    /**
     *  First renderpass in this sample.
     *  Stenciled rendering: subpass 1 draw to stencil buffer, subpass 2 draw to
     *  color buffer with stencil test
     */
    VkAttachmentDescription attachments[2];
    attachments[0].format = info.format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].flags = 0;

    attachments[1].format = info.depth.format;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].flags = 0;

    VkAttachmentReference color_reference = {};
    color_reference.attachment = 0;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_reference = {};
    depth_reference.attachment = 1;
    depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = 0;
    subpass.pColorAttachments = NULL;
    subpass.pResolveAttachments = NULL;
    subpass.pDepthStencilAttachment = &depth_reference;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    std::vector<VkSubpassDescription> subpasses;

    /* first a depthstencil-only subpass */
    subpasses.push_back(subpass);

    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;

    /* then depthstencil and color */
    subpasses.push_back(subpass);

    /* Set up a dependency between the source and destination subpasses */
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = 0;
    dependency.dstSubpass = 1;
    dependency.dependencyFlags = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.pNext = NULL;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = subpasses.size();
    rp_info.pSubpasses = subpasses.data();
    rp_info.dependencyCount = 1;
    rp_info.pDependencies = &dependency;

    VkRenderPass stencil_render_pass;
    res = vkCreateRenderPass(info.device, &rp_info, NULL, &stencil_render_pass);
    assert(!res);

    /* now that we have the render pass, create framebuffer and pipelines */

    info.render_pass = stencil_render_pass;
    init_framebuffers(info, depthPresent);

    VkDynamicState dynamicStateEnables[VK_DYNAMIC_STATE_RANGE_SIZE];
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.pNext = NULL;
    dynamicState.pDynamicStates = dynamicStateEnables;
    dynamicState.dynamicStateCount = 0;

    VkPipelineVertexInputStateCreateInfo vi;
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.pNext = NULL;
    vi.flags = 0;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &info.vi_binding;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions = info.vi_attribs;

    VkPipelineInputAssemblyStateCreateInfo ia;
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.pNext = NULL;
    ia.flags = 0;
    ia.primitiveRestartEnable = VK_FALSE;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo rs;
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.pNext = NULL;
    rs.flags = 0;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.depthBiasEnable = VK_FALSE;
    rs.depthBiasConstantFactor = 0;
    rs.depthBiasClamp = 0;
    rs.depthBiasSlopeFactor = 0;
    rs.lineWidth = 1.0f;

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

    VkPipelineViewportStateCreateInfo vp = {};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.pNext = NULL;
    vp.viewportCount = NUM_VIEWPORTS;
    dynamicStateEnables[dynamicState.dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
    vp.scissorCount = NUM_SCISSORS;
    dynamicStateEnables[dynamicState.dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;

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

    ds.stencilTestEnable = VK_TRUE;
    ds.back.failOp = VK_STENCIL_OP_REPLACE;
    ds.back.depthFailOp = VK_STENCIL_OP_REPLACE;
    ds.back.passOp = VK_STENCIL_OP_REPLACE;
    ds.back.compareOp = VK_COMPARE_OP_ALWAYS;
    ds.back.compareMask = 0xff;
    ds.back.writeMask = 0xff;
    ds.back.reference = 0x44;
    ds.front = ds.back;

    VkPipelineMultisampleStateCreateInfo ms;
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.pNext = NULL;
    ms.flags = 0;
    ms.pSampleMask = NULL;
    ms.rasterizationSamples = NUM_SAMPLES;
    ms.sampleShadingEnable = VK_FALSE;
    ms.minSampleShading = 0.0;
    ms.alphaToCoverageEnable = VK_FALSE;
    ms.alphaToOneEnable = VK_FALSE;

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
    pipeline.pColorBlendState = NULL;
    pipeline.pTessellationState = NULL;
    pipeline.pMultisampleState = &ms;
    pipeline.pDynamicState = &dynamicState;
    pipeline.pViewportState = &vp;
    pipeline.pDepthStencilState = &ds;
    pipeline.pStages = info.shaderStages;
    pipeline.stageCount = 2;
    pipeline.renderPass = stencil_render_pass;
    pipeline.subpass = 0;

    init_shaders(info, normalVertShaderText, fragShaderText);

    /* The first pipeline will render in subpass 0 to fill the stencil */
    pipeline.subpass = 0;

    VkPipeline stencil_cube_pipe = VK_NULL_HANDLE;
    res = vkCreateGraphicsPipelines(info.device, info.pipelineCache, 1, &pipeline, NULL, &stencil_cube_pipe);
    assert(res == VK_SUCCESS);

    /* destroy the shaders used for the above pipelin eand replace them with
       those for the
       fullscreen fill pass */
    destroy_shaders(info);
    init_shaders(info, fullscreenVertShaderText, fragShaderText);

    /* the second pipeline will stencil test but not write, using the same
     * reference */
    ds.back.failOp = VK_STENCIL_OP_KEEP;
    ds.back.depthFailOp = VK_STENCIL_OP_KEEP;
    ds.back.passOp = VK_STENCIL_OP_KEEP;
    ds.back.compareOp = VK_COMPARE_OP_EQUAL;
    ds.front = ds.back;

    /* don't test depth, only use stencil test */
    ds.depthTestEnable = VK_FALSE;

    /* the second pipeline will be a fullscreen triangle strip, with vertices
       generated purely from the vertex shader - no inputs needed */
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    vi.vertexAttributeDescriptionCount = 0;
    vi.vertexBindingDescriptionCount = 0;

    /* this pipeline will run in the second subpass */
    pipeline.subpass = 1;
    pipeline.pColorBlendState = &cb;

    VkPipeline stencil_fullscreen_pipe = VK_NULL_HANDLE;
    res = vkCreateGraphicsPipelines(info.device, info.pipelineCache, 1, &pipeline, NULL, &stencil_fullscreen_pipe);
    assert(res == VK_SUCCESS);

    destroy_shaders(info);
    info.pipeline = VK_NULL_HANDLE;

    VkClearValue clear_values[2];
    clear_values[0].color.float32[0] = 0.2f;
    clear_values[0].color.float32[1] = 0.2f;
    clear_values[0].color.float32[2] = 0.2f;
    clear_values[0].color.float32[3] = 0.2f;
    clear_values[1].depthStencil.depth = 1.0f;
    clear_values[1].depthStencil.stencil = 0;

    VkSemaphore imageAcquiredSemaphore;
    VkSemaphoreCreateInfo imageAcquiredSemaphoreCreateInfo;
    imageAcquiredSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    imageAcquiredSemaphoreCreateInfo.pNext = NULL;
    imageAcquiredSemaphoreCreateInfo.flags = 0;

    res = vkCreateSemaphore(info.device, &imageAcquiredSemaphoreCreateInfo, NULL, &imageAcquiredSemaphore);
    assert(res == VK_SUCCESS);

    // Get the index of the next available swapchain image:
    res = vkAcquireNextImageKHR(info.device, info.swap_chain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE,
                                &info.current_buffer);
    // TODO: Deal with the VK_SUBOPTIMAL_KHR and VK_ERROR_OUT_OF_DATE_KHR
    // return codes
    assert(res == VK_SUCCESS);

    VkRenderPassBeginInfo rp_begin;
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.pNext = NULL;
    rp_begin.renderPass = stencil_render_pass;
    rp_begin.framebuffer = info.framebuffers[info.current_buffer];
    rp_begin.renderArea.offset.x = 0;
    rp_begin.renderArea.offset.y = 0;
    rp_begin.renderArea.extent.width = info.width / 2;
    rp_begin.renderArea.extent.height = info.height;
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    /* Begin the first render pass. This will render in the left half of the
       screen. Subpass 0 will render a cube, stencil writing but outputting
       no color. Subpass 1 will render a fullscreen pass, stencil testing and
       outputting color only where the cube filled in stencil */
    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stencil_cube_pipe);
    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            info.desc_set.data(), 0, NULL);

    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(info.cmd, 0, 1, &info.vertex_buffer.buf, offsets);

    VkViewport viewport;
    viewport.height = (float)info.height;
    viewport.width = (float)info.width / 2;
    viewport.minDepth = (float)0.0f;
    viewport.maxDepth = (float)1.0f;
    viewport.x = 0;
    viewport.y = 0;
    vkCmdSetViewport(info.cmd, 0, NUM_VIEWPORTS, &viewport);

    VkRect2D scissor;
    scissor.extent.width = info.width / 2;
    scissor.extent.height = info.height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    vkCmdSetScissor(info.cmd, 0, NUM_SCISSORS, &scissor);

    /* Draw the cube into stencil */
    vkCmdDraw(info.cmd, 12 * 3, 1, 0, 0);

    /* Advance to the next subpass */
    vkCmdNextSubpass(info.cmd, VK_SUBPASS_CONTENTS_INLINE);

    /* Bind the fullscreen pass pipeline */
    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stencil_fullscreen_pipe);

    vkCmdSetViewport(info.cmd, 0, NUM_VIEWPORTS, &viewport);
    vkCmdSetScissor(info.cmd, 0, NUM_SCISSORS, &scissor);

    /* Draw the fullscreen pass */
    vkCmdDraw(info.cmd, 4, 1, 0, 0);

    vkCmdEndRenderPass(info.cmd);

    /**
     * Second renderpass in this sample.
     * Blended rendering, each subpass blends continuously onto the color
     */

    /* note that we reuse a lot of the initialisation strutures from the first
       render pass, so this represents a 'delta' from that configuration */

    /* This time, the first subpass will use color */
    subpasses[0].colorAttachmentCount = 1;
    subpasses[0].pColorAttachments = &color_reference;

    /* This time we care about the contents of the color and depth buffers */
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    /* The dependency between the subpasses now includes the color attachment */
    dependency.srcAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    dependency.dstAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    /* Otherwise, the render pass is identical */
    VkRenderPass blend_render_pass;
    res = vkCreateRenderPass(info.device, &rp_info, NULL, &blend_render_pass);
    assert(!res);

    pipeline.renderPass = blend_render_pass;

    /* We must recreate the framebuffers with this renderpass as the two render
       passes are not compatible. Store the current framebuffers for later
       deletion */
    VkFramebuffer *stencil_framebuffers = info.framebuffers;
    info.framebuffers = NULL;

    info.render_pass = blend_render_pass;
    init_framebuffers(info, depthPresent);

    /* Now create the pipelines for the second render pass */

    /* We are rendering the cube again, configure the vertex inputs */
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vi.vertexAttributeDescriptionCount = 2;
    vi.vertexBindingDescriptionCount = 1;

    /* The first pipeline will depth write and depth test */
    ds.depthWriteEnable = VK_TRUE;
    ds.depthTestEnable = VK_TRUE;

    /* We don't want to stencil test */
    ds.stencilTestEnable = VK_FALSE;

    /* This time, both pipelines will blend. the first pipeline uses the blend
     constant
     to determine the blend amount */
    att_state[0].colorWriteMask = 0xf;
    att_state[0].blendEnable = VK_TRUE;
    att_state[0].alphaBlendOp = VK_BLEND_OP_ADD;
    att_state[0].colorBlendOp = VK_BLEND_OP_ADD;
    att_state[0].srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    att_state[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    att_state[0].srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    att_state[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;

    cb.blendConstants[0] = 1.0f;
    cb.blendConstants[1] = 1.0f;
    cb.blendConstants[2] = 1.0f;
    cb.blendConstants[3] = 0.3f;

    init_shaders(info, normalVertShaderText, fragShaderText);

    /* This is the first subpass's pipeline, to blend a cube onto the color
     * image */
    pipeline.subpass = 0;

    VkPipeline blend_cube_pipe = VK_NULL_HANDLE;
    res = vkCreateGraphicsPipelines(info.device, info.pipelineCache, 1, &pipeline, NULL, &blend_cube_pipe);
    assert(res == VK_SUCCESS);

    /* Now we will set up the fullscreen pass to render on top. */
    destroy_shaders(info);
    init_shaders(info, fullscreenVertShaderText, fragShaderText);

    /* the second pipeline will be a fullscreen triangle strip with no inputs */
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    vi.vertexAttributeDescriptionCount = 0;
    vi.vertexBindingDescriptionCount = 0;

    /* We'll use the alpha output from the shader */
    att_state[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    att_state[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    att_state[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    att_state[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;

    /* This renders in the second subpass */
    pipeline.subpass = 1;

    VkPipeline blend_fullscreen_pipe = VK_NULL_HANDLE;
    res = vkCreateGraphicsPipelines(info.device, info.pipelineCache, 1, &pipeline, NULL, &blend_fullscreen_pipe);
    assert(res == VK_SUCCESS);

    destroy_shaders(info);
    info.pipeline = VK_NULL_HANDLE;

    /* Now we are going to render in the right half of the screen */
    viewport.x = (float)info.width / 2;
    scissor.offset.x = info.width / 2;
    rp_begin.renderArea.offset.x = info.width / 2;

    /* Use our framebuffer and render pass */
    rp_begin.framebuffer = info.framebuffers[info.current_buffer];
    rp_begin.renderPass = blend_render_pass;
    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blend_cube_pipe);
    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            info.desc_set.data(), 0, NULL);
    vkCmdBindVertexBuffers(info.cmd, 0, 1, &info.vertex_buffer.buf, offsets);
    vkCmdSetViewport(info.cmd, 0, NUM_VIEWPORTS, &viewport);
    vkCmdSetScissor(info.cmd, 0, NUM_SCISSORS, &scissor);

    /* Draw the cube blending */
    vkCmdDraw(info.cmd, 12 * 3, 1, 0, 0);

    /* Advance to the next subpass */
    vkCmdNextSubpass(info.cmd, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blend_fullscreen_pipe);

    /* Adjust the viewport to be a square in the centre, just overlapping the
     * cube */
    viewport.x += 25.0f;
    viewport.y += 150.0f;
    viewport.width -= 50.0f;
    viewport.height -= 300.0f;

    vkCmdSetViewport(info.cmd, 0, NUM_VIEWPORTS, &viewport);
    vkCmdSetScissor(info.cmd, 0, NUM_SCISSORS, &scissor);

    vkCmdDraw(info.cmd, 4, 1, 0, 0);

    /* The second renderpass is complete */
    vkCmdEndRenderPass(info.cmd);
    /* VULKAN_KEY_END */
    res = vkEndCommandBuffer(info.cmd);
    const VkCommandBuffer cmd_bufs[] = {info.cmd};
    VkFenceCreateInfo fenceInfo;
    VkFence drawFence;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(info.device, &fenceInfo, NULL, &drawFence);

    VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info[1] = {};
    submit_info[0].pNext = NULL;
    submit_info[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info[0].waitSemaphoreCount = 1;
    submit_info[0].pWaitSemaphores = &imageAcquiredSemaphore;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
    submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
    submit_info[0].signalSemaphoreCount = 0;
    submit_info[0].pSignalSemaphores = NULL;

    /* Queue the command buffer for execution */
    res = vkQueueSubmit(info.graphics_queue, 1, submit_info, drawFence);
    assert(res == VK_SUCCESS);

    /* Now present the image in the window */

    VkPresentInfoKHR present;
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.pNext = NULL;
    present.swapchainCount = 1;
    present.pSwapchains = &info.swap_chain;
    present.pImageIndices = &info.current_buffer;
    present.pWaitSemaphores = NULL;
    present.waitSemaphoreCount = 0;
    present.pResults = NULL;

    /* Make sure command buffer is finished before presenting */
    do {
        res = vkWaitForFences(info.device, 1, &drawFence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);
    res = vkQueuePresentKHR(info.present_queue, &present);
    assert(res == VK_SUCCESS);

    wait_seconds(1);
    /* VULKAN_KEY_END */
    if (info.save_images) write_ppm(info, "draw_subpasses");

    for (uint32_t i = 0; i < info.swapchainImageCount; i++) vkDestroyFramebuffer(info.device, stencil_framebuffers[i], NULL);
    free(stencil_framebuffers);

    vkDestroyRenderPass(info.device, stencil_render_pass, NULL);
    vkDestroyRenderPass(info.device, blend_render_pass, NULL);

    vkDestroyPipeline(info.device, blend_cube_pipe, NULL);
    vkDestroyPipeline(info.device, blend_fullscreen_pipe, NULL);

    vkDestroyPipeline(info.device, stencil_cube_pipe, NULL);
    vkDestroyPipeline(info.device, stencil_fullscreen_pipe, NULL);

    vkDestroySemaphore(info.device, imageAcquiredSemaphore, NULL);
    vkDestroyFence(info.device, drawFence, NULL);
    destroy_pipeline_cache(info);
    destroy_descriptor_pool(info);
    destroy_vertex_buffer(info);
    destroy_framebuffers(info);
    destroy_descriptor_and_pipeline_layouts(info);
    destroy_uniform_buffer(info);
    destroy_depth_buffer(info);
    destroy_swap_chain(info);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_device(info);
    destroy_window(info);
    destroy_instance(info);
    return 0;
}

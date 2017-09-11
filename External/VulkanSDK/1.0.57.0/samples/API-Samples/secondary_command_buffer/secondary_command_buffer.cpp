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
Draw several cubes using primary and secondary command buffers
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

const char *vertShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (std140, binding = 0) uniform buf {\n"
    "        mat4 mvp;\n"
    "} ubuf;\n"
    "layout (location = 0) in vec4 pos;\n"
    "layout (location = 1) in vec2 inTexCoords;\n"
    "layout (location = 0) out vec2 texcoord;\n"
    "void main() {\n"
    "   texcoord = inTexCoords;\n"
    "   gl_Position = ubuf.mvp * pos;\n"
    "}\n";

const char *fragShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (binding = 1) uniform sampler2D tex;\n"
    "layout (location = 0) in vec2 texcoord;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = textureLod(tex, texcoord, 0.0);\n"
    "}\n";

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Secondary command buffers";
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
    init_depth_buffer(info);
    init_uniform_buffer(info);
    init_descriptor_and_pipeline_layouts(info, true);
    init_renderpass(info, depthPresent, true, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    init_shaders(info, vertShaderText, fragShaderText);
    init_framebuffers(info, depthPresent);
    init_vertex_buffer(info, g_vb_texture_Data, sizeof(g_vb_texture_Data), sizeof(g_vb_texture_Data[0]), true);
    init_pipeline_cache(info);
    init_pipeline(info, depthPresent);

    // we have to set up a couple of things by hand, but this
    // isn't any different to other examples

    // get two different textures
    init_texture(info, "green.ppm");
    VkDescriptorImageInfo greenTex = info.texture_data.image_info;

    init_texture(info, "lunarg.ppm");
    VkDescriptorImageInfo lunargTex = info.texture_data.image_info;

    // create two identical descriptor sets, each with a different texture but
    // identical UBOa
    VkDescriptorPoolSize pool_size[2];
    pool_size[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size[0].descriptorCount = 2;
    pool_size[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo descriptor_pool = {};
    descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool.pNext = NULL;
    descriptor_pool.flags = 0;
    descriptor_pool.maxSets = 2;
    descriptor_pool.poolSizeCount = 2;
    descriptor_pool.pPoolSizes = pool_size;

    res = vkCreateDescriptorPool(info.device, &descriptor_pool, NULL, &info.desc_pool);
    assert(res == VK_SUCCESS);

    VkDescriptorSetLayout layouts[] = {info.desc_layout[0], info.desc_layout[0]};

    VkDescriptorSetAllocateInfo alloc_info[1];
    alloc_info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info[0].pNext = NULL;
    alloc_info[0].descriptorPool = info.desc_pool;
    alloc_info[0].descriptorSetCount = 2;
    alloc_info[0].pSetLayouts = layouts;

    info.desc_set.resize(2);
    res = vkAllocateDescriptorSets(info.device, alloc_info, info.desc_set.data());
    assert(res == VK_SUCCESS);

    VkWriteDescriptorSet writes[2];

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = NULL;
    writes[0].dstSet = info.desc_set[0];
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &info.uniform_data.buffer_info;
    writes[0].dstArrayElement = 0;
    writes[0].dstBinding = 0;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].pNext = NULL;
    writes[1].dstSet = info.desc_set[0];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &greenTex;
    writes[1].dstArrayElement = 0;

    vkUpdateDescriptorSets(info.device, 2, writes, 0, NULL);

    writes[0].dstSet = writes[1].dstSet = info.desc_set[1];
    writes[1].pImageInfo = &lunargTex;

    vkUpdateDescriptorSets(info.device, 2, writes, 0, NULL);

    /* VULKAN_KEY_START */

    // create four secondary command buffers, for each quadrant of the screen

    VkCommandBufferAllocateInfo cmdalloc = {};
    cmdalloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdalloc.pNext = NULL;
    cmdalloc.commandPool = info.cmd_pool;
    cmdalloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    cmdalloc.commandBufferCount = 4;

    VkCommandBuffer secondary_cmds[4];

    res = vkAllocateCommandBuffers(info.device, &cmdalloc, secondary_cmds);
    assert(res == VK_SUCCESS);

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

    set_image_layout(info, info.buffers[info.current_buffer].image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    const VkDeviceSize offsets[1] = {0};

    VkViewport viewport;
    viewport.height = 200.0f;
    viewport.width = 200.0f;
    viewport.minDepth = (float)0.0f;
    viewport.maxDepth = (float)1.0f;
    viewport.x = 0;
    viewport.y = 0;

    VkRect2D scissor;
    scissor.extent.width = info.width;
    scissor.extent.height = info.height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;

    // now we record four separate command buffers, one for each quadrant of the
    // screen
    VkCommandBufferInheritanceInfo cmd_buf_inheritance_info = {};
    cmd_buf_inheritance_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, cmd_buf_inheritance_info.pNext = NULL;
    cmd_buf_inheritance_info.renderPass = info.render_pass;
    cmd_buf_inheritance_info.subpass = 0;
    cmd_buf_inheritance_info.framebuffer = info.framebuffers[info.current_buffer];
    cmd_buf_inheritance_info.occlusionQueryEnable = VK_FALSE;
    cmd_buf_inheritance_info.queryFlags = 0;
    cmd_buf_inheritance_info.pipelineStatistics = 0;

    VkCommandBufferBeginInfo secondary_begin = {};
    secondary_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    secondary_begin.pNext = NULL;
    secondary_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    secondary_begin.pInheritanceInfo = &cmd_buf_inheritance_info;

    for (int i = 0; i < 4; i++) {
        vkBeginCommandBuffer(secondary_cmds[i], &secondary_begin);

        vkCmdBindPipeline(secondary_cmds[i], VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline);
        vkCmdBindDescriptorSets(secondary_cmds[i], VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, 1,
                                &info.desc_set[i == 0 || i == 3], 0, NULL);

        vkCmdBindVertexBuffers(secondary_cmds[i], 0, 1, &info.vertex_buffer.buf, offsets);

        viewport.x = 25.0f + 250.0f * (i % 2);
        viewport.y = 25.0f + 250.0f * (i / 2);
        vkCmdSetViewport(secondary_cmds[i], 0, NUM_VIEWPORTS, &viewport);

        vkCmdSetScissor(secondary_cmds[i], 0, NUM_SCISSORS, &scissor);

        vkCmdDraw(secondary_cmds[i], 12 * 3, 1, 0, 0);

        vkEndCommandBuffer(secondary_cmds[i]);
    }

    VkRenderPassBeginInfo rp_begin;
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.pNext = NULL;
    rp_begin.renderPass = info.render_pass;
    rp_begin.framebuffer = info.framebuffers[info.current_buffer];
    rp_begin.renderArea.offset.x = 0;
    rp_begin.renderArea.offset.y = 0;
    rp_begin.renderArea.extent.width = info.width;
    rp_begin.renderArea.extent.height = info.height;
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    // specifying VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS means this
    // render pass may
    // ONLY call vkCmdExecuteCommands
    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

    vkCmdExecuteCommands(info.cmd, 4, secondary_cmds);

    vkCmdEndRenderPass(info.cmd);

    VkImageMemoryBarrier prePresentBarrier = {};
    prePresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    prePresentBarrier.pNext = NULL;
    prePresentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    prePresentBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    prePresentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    prePresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    prePresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prePresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prePresentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    prePresentBarrier.subresourceRange.baseMipLevel = 0;
    prePresentBarrier.subresourceRange.levelCount = 1;
    prePresentBarrier.subresourceRange.baseArrayLayer = 0;
    prePresentBarrier.subresourceRange.layerCount = 1;
    prePresentBarrier.image = info.buffers[info.current_buffer].image;
    vkCmdPipelineBarrier(info.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL,
                         0, NULL, 1, &prePresentBarrier);

    res = vkEndCommandBuffer(info.cmd);
    assert(res == VK_SUCCESS);

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
    submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
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
    if (info.save_images) write_ppm(info, "secondary_command_buffer");

    vkFreeCommandBuffers(info.device, info.cmd_pool, 4, secondary_cmds);

    /* VULKAN_KEY_END */

    vkDestroyFence(info.device, drawFence, NULL);
    vkDestroySemaphore(info.device, imageAcquiredSemaphore, NULL);
    destroy_pipeline(info);
    destroy_pipeline_cache(info);
    destroy_textures(info);
    destroy_descriptor_pool(info);
    destroy_vertex_buffer(info);
    destroy_framebuffers(info);
    destroy_shaders(info);
    destroy_renderpass(info);
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

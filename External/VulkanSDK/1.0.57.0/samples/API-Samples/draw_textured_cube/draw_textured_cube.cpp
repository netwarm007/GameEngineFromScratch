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
Draw Textured Cube
*/

/* This is part of the draw cube progression */

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
    char sample_title[] = "Draw Textured Cube";
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
    init_texture(info);
    init_uniform_buffer(info);
    init_descriptor_and_pipeline_layouts(info, true);
    init_renderpass(info, depthPresent);
    init_shaders(info, vertShaderText, fragShaderText);
    init_framebuffers(info, depthPresent);
    init_vertex_buffer(info, g_vb_texture_Data, sizeof(g_vb_texture_Data), sizeof(g_vb_texture_Data[0]), true);
    init_descriptor_pool(info, true);
    init_descriptor_set(info, true);
    init_pipeline_cache(info);
    init_pipeline(info, depthPresent);

    /* VULKAN_KEY_START */

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
    rp_begin.renderPass = info.render_pass;
    rp_begin.framebuffer = info.framebuffers[info.current_buffer];
    rp_begin.renderArea.offset.x = 0;
    rp_begin.renderArea.offset.y = 0;
    rp_begin.renderArea.extent.width = info.width;
    rp_begin.renderArea.extent.height = info.height;
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline);
    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            info.desc_set.data(), 0, NULL);

    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(info.cmd, 0, 1, &info.vertex_buffer.buf, offsets);

    init_viewports(info);
    init_scissors(info);

    vkCmdDraw(info.cmd, 12 * 3, 1, 0, 0);
    vkCmdEndRenderPass(info.cmd);
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
    /* VULKAN_KEY_END */
    if (info.save_images) write_ppm(info, "draw_textured_cube");

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

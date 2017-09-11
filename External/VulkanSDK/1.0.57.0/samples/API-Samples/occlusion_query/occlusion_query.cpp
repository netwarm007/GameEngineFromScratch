/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google, Inc.
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
Use occlusion query to determine if drawing renders any samples.
This could be used to quickly determine if more expensive rendering should be
done. Use vkCreateQueryPool, vkCmdResetQueryPool, and vkDestroyQueryPool to
manage a pool. Use vkCmdBeginQuery and vkCmdEndQuery to enclose rendering.
Use vkCmdCopyQueryPoolResults or vkGetQueryPoolResults to read query results.
This example does one query with no rendering to give a zero result and a second
query with rendering to give a non-zero result.  Note that exact counts are not
guaranteed unless vkGetPhysicalDeviceFeatures sets occlusionQueryPrecise and the
VK_QUERY_CONTROL_PRECISE_BIT is set for vkCmdBeginQuery.

This example uses vkCmdCopyQueryPoolResults followed by vkMapMemory of a buffer.
vkCmdCopyQueryPoolResults could also be used to set uniforms used later by
shaders.
*/

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include "cube_data.h"

#define DEPTH_PRESENT true

/* For this sample, we'll start with GLSL so the shader function is plain */
/* and then use the glslang GLSLtoSPV utility to convert it to SPIR-V for */
/* the driver.  We do this for clarity rather than using pre-compiled     */
/* SPIR-V                                                                 */

static const char *vertShaderText =
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

static const char *fragShaderText =
    "#version 400\n"
    "#extension GL_ARB_separate_shader_objects : enable\n"
    "#extension GL_ARB_shading_language_420pack : enable\n"
    "layout (location = 0) in vec4 color;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = color;\n"
    "}\n";

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    bool U_ASSERT_ONLY pass;
    struct sample_info info = {};
    char sample_title[] = "Draw Cube";

    process_command_line_args(info, argc, argv);
    init_global_layer_properties(info);
    info.instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    info.instance_extension_names.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__ANDROID__)
    info.instance_extension_names.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
    info.instance_extension_names.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
    info.instance_extension_names.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#else
    info.instance_extension_names.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
    info.device_extension_names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
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
    init_descriptor_and_pipeline_layouts(info, false);
    init_renderpass(info, DEPTH_PRESENT);
    init_shaders(info, vertShaderText, fragShaderText);
    init_framebuffers(info, DEPTH_PRESENT);
    init_vertex_buffer(info, g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data),
                       sizeof(g_vb_solid_face_colors_Data[0]), false);
    init_descriptor_pool(info, false);
    init_descriptor_set(info, false);
    init_pipeline_cache(info);
    init_pipeline(info, DEPTH_PRESENT);

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

    /* Allocate a uniform buffer that will take query results. */
    VkBuffer query_result_buf;
    VkDeviceMemory query_result_mem;
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = NULL;
    buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.size = 4 * sizeof(uint64_t);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_info.flags = 0;
    res = vkCreateBuffer(info.device, &buf_info, NULL, &query_result_buf);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(info.device, query_result_buf, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = NULL;
    alloc_info.memoryTypeIndex = 0;
    alloc_info.allocationSize = mem_reqs.size;
    pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex);
    assert(pass && "No mappable, coherent memory");

    res = vkAllocateMemory(info.device, &alloc_info, NULL, &query_result_mem);
    assert(res == VK_SUCCESS);

    res = vkBindBufferMemory(info.device, query_result_buf, query_result_mem, 0);
    assert(res == VK_SUCCESS);

    VkQueryPool query_pool;
    VkQueryPoolCreateInfo query_pool_info;
    query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_pool_info.pNext = NULL;
    query_pool_info.queryType = VK_QUERY_TYPE_OCCLUSION;
    query_pool_info.flags = 0;
    query_pool_info.queryCount = 2;
    query_pool_info.pipelineStatistics = 0;

    res = vkCreateQueryPool(info.device, &query_pool_info, NULL, &query_pool);
    assert(res == VK_SUCCESS);

    vkCmdResetQueryPool(info.cmd, query_pool, 0 /*startQuery*/, 2 /*queryCount*/);

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

    VkViewport viewport;
    viewport.height = (float)info.height;
    viewport.width = (float)info.width;
    viewport.minDepth = (float)0.0f;
    viewport.maxDepth = (float)1.0f;
    viewport.x = 0;
    viewport.y = 0;
    vkCmdSetViewport(info.cmd, 0, NUM_VIEWPORTS, &viewport);

    VkRect2D scissor;
    scissor.extent.width = info.width;
    scissor.extent.height = info.height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    vkCmdSetScissor(info.cmd, 0, NUM_SCISSORS, &scissor);

    vkCmdBeginQuery(info.cmd, query_pool, 0 /*slot*/, 0 /*flags*/);
    vkCmdEndQuery(info.cmd, query_pool, 0 /*slot*/);

    vkCmdBeginQuery(info.cmd, query_pool, 1 /*slot*/, 0 /*flags*/);

    vkCmdDraw(info.cmd, 12 * 3, 1, 0, 0);
    vkCmdEndRenderPass(info.cmd);

    vkCmdEndQuery(info.cmd, query_pool, 1 /*slot*/);

    vkCmdCopyQueryPoolResults(info.cmd, query_pool, 0 /*firstQuery*/, 2 /*queryCount*/, query_result_buf, 0 /*dstOffset*/,
                              sizeof(uint64_t) /*stride*/, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

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
    submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
    submit_info[0].signalSemaphoreCount = 0;
    submit_info[0].pSignalSemaphores = NULL;

    /* Queue the command buffer for execution */
    res = vkQueueSubmit(info.graphics_queue, 1, submit_info, drawFence);
    assert(res == VK_SUCCESS);

    res = vkQueueWaitIdle(info.graphics_queue);
    assert(res == VK_SUCCESS);

    uint64_t samples_passed[4];

    samples_passed[0] = 0;
    samples_passed[1] = 0;
    res = vkGetQueryPoolResults(info.device, query_pool, 0 /*firstQuery*/, 2 /*queryCount*/, sizeof(samples_passed) /*dataSize*/,
                                samples_passed, sizeof(uint64_t) /*stride*/, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    assert(res == VK_SUCCESS);

    std::cout << "vkGetQueryPoolResults data"
              << "\n";
    std::cout << "samples_passed[0] = " << samples_passed[0] << "\n";
    std::cout << "samples_passed[1] = " << samples_passed[1] << "\n";

    /* Read back query result from buffer */
    uint64_t *samples_passed_ptr;
    res = vkMapMemory(info.device, query_result_mem, 0, mem_reqs.size, 0, (void **)&samples_passed_ptr);
    assert(res == VK_SUCCESS);

    std::cout << "vkCmdCopyQueryPoolResults data"
              << "\n";
    std::cout << "samples_passed[0] = " << samples_passed_ptr[0] << "\n";
    std::cout << "samples_passed[1] = " << samples_passed_ptr[1] << "\n";

    vkUnmapMemory(info.device, query_result_mem);

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
    if (info.save_images) write_ppm(info, "occlusion_query");

    vkDestroyBuffer(info.device, query_result_buf, NULL);
    vkFreeMemory(info.device, query_result_mem, NULL);
    vkDestroySemaphore(info.device, imageAcquiredSemaphore, NULL);
    vkDestroyQueryPool(info.device, query_pool, NULL);
    vkDestroyFence(info.device, drawFence, NULL);
    destroy_pipeline(info);
    destroy_pipeline_cache(info);
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

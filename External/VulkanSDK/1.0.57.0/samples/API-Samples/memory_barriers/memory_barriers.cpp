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
Use memory barriers to update texture
*/

/* Set up a vertex buffer and a texture and use them to draw a textured
 * quad.  Use a memory barrier to prepare the texture to be cleared.  Clear the
 * texture to green and use another memory barrier to prepare the image to
 * be used as a texture again. Use the second set of vertices to draw a quad to
 * the right of the first. Do this all in one command buffer.  The image should
 * be a LunarG logo quad on the left and a green quad on the right.
 */

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <cube_data.h>

// Using OpenGL based glm, so Y is upside down between OpenGL and Vulkan

static const VertexUV vb_Data[] = {
    // Textured quad:
    {XYZ1(-2, -0.5, -1), UV(0.f, 0.f)},  // lft-top        / Z
    {XYZ1(-1, -0.5, -1), UV(1.f, 0.f)},  // rgt-top       /
    {XYZ1(-2, 0.5, -1), UV(0.f, 1.f)},   // lft-btm      +------> X
    {XYZ1(-2, 0.5, -1), UV(0.f, 1.f)},   // lft-btm      |
    {XYZ1(-1, -0.5, -1), UV(1.f, 0.f)},  // rgt-top      |
    {XYZ1(-1, 0.5, -1), UV(1.f, 1.f)},   // rgt-btm      v Y
    // Green quad:
    {XYZ1(1, -0.5, -1), UV(0.f, 0.f)},  // lft-top
    {XYZ1(2, -0.5, -1), UV(1.f, 0.f)},  // rgt-top
    {XYZ1(1, 0.5, -1), UV(0.f, 1.f)},   // lft-btm
    {XYZ1(1, 0.5, -1), UV(0.f, 1.f)},   // lft-btm
    {XYZ1(2, -0.5, -1), UV(1.f, 0.f)},  // rgt-top
    {XYZ1(2, 0.5, -1), UV(1.f, 1.f)},   // rgt-btm
};

#define DEPTH_PRESENT false

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

int sample_main(int argc, char **argv) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Memory Barriers";

    process_command_line_args(info, argc, argv);
    init_global_layer_properties(info);
    info.instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    info.instance_extension_names.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif __ANDROID__
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
    init_device(info);
    info.width = info.height = 500;
    init_connection(info);
    init_window(info);
    init_swapchain_extension(info);
    init_command_pool(info);
    init_command_buffer(info);
    execute_begin_command_buffer(info);
    init_device_queue(info);
    init_swap_chain(info, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    // CmdClearColorImage is going to require usage of TRANSFER_DST, but
    // it's not clear which format feature maps to the required TRANSFER_DST usage,
    // BLIT_DST is a reasonable guess and it seems to work
    init_texture(info, nullptr, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FORMAT_FEATURE_BLIT_DST_BIT);
    init_uniform_buffer(info);
    init_descriptor_and_pipeline_layouts(info, true);
    init_renderpass(info, DEPTH_PRESENT, false, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    init_shaders(info, vertShaderText, fragShaderText);
    init_framebuffers(info, DEPTH_PRESENT);
    init_vertex_buffer(info, vb_Data, sizeof(vb_Data), sizeof(vb_Data[0]), true);
    init_descriptor_pool(info, true);
    init_descriptor_set(info, true);
    init_pipeline_cache(info);
    init_pipeline(info, DEPTH_PRESENT);

    /* VULKAN_KEY_START */

    VkImageSubresourceRange srRange = {};
    srRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    srRange.baseMipLevel = 0;
    srRange.levelCount = VK_REMAINING_MIP_LEVELS;
    srRange.baseArrayLayer = 0;
    srRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkClearColorValue clear_color[1];
    clear_color[0].float32[0] = 0.2f;
    clear_color[0].float32[1] = 0.2f;
    clear_color[0].float32[2] = 0.2f;
    clear_color[0].float32[3] = 0.2f;

    VkSemaphoreCreateInfo presentCompleteSemaphoreCreateInfo;
    presentCompleteSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    presentCompleteSemaphoreCreateInfo.pNext = NULL;
    presentCompleteSemaphoreCreateInfo.flags = 0;

    res = vkCreateSemaphore(info.device, &presentCompleteSemaphoreCreateInfo, NULL, &info.imageAcquiredSemaphore);
    assert(res == VK_SUCCESS);

    // Get the index of the next available swapchain image:
    res = vkAcquireNextImageKHR(info.device, info.swap_chain, UINT64_MAX, info.imageAcquiredSemaphore, VK_NULL_HANDLE,
                                &info.current_buffer);
    // TODO: Deal with the VK_SUBOPTIMAL_KHR and VK_ERROR_OUT_OF_DATE_KHR
    // return codes
    assert(res == VK_SUCCESS);

    set_image_layout(info, info.buffers[info.current_buffer].image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // We need to do the clear here instead of using a renderpass load op since
    // we will use the same renderpass multiple times in the frame
    vkCmdClearColorImage(info.cmd, info.buffers[info.current_buffer].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clear_color, 1,
                         &srRange);

    VkRenderPassBeginInfo rp_begin;
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.pNext = NULL;
    rp_begin.renderPass = info.render_pass;
    rp_begin.framebuffer = info.framebuffers[info.current_buffer];
    rp_begin.renderArea.offset.x = 0;
    rp_begin.renderArea.offset.y = 0;
    rp_begin.renderArea.extent.width = info.width;
    rp_begin.renderArea.extent.height = info.height;
    rp_begin.clearValueCount = 0;
    rp_begin.pClearValues = NULL;

    // Draw a textured quad on the left side of the window
    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline);
    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            info.desc_set.data(), 0, NULL);

    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(info.cmd, 0, 1, &info.vertex_buffer.buf, offsets);

    init_viewports(info);
    init_scissors(info);

    vkCmdDraw(info.cmd, 2 * 3, 1, 0, 0);
    // We can't do a clear inside a renderpass, so end this one and start another one
    // for the next draw
    vkCmdEndRenderPass(info.cmd);

    // Send a barrier to change the texture image's layout from SHADER_READ_ONLY
    // to COLOR_ATTACHMENT_GENERAL because we're going to clear it
    VkImageMemoryBarrier textureBarrier = {};
    textureBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    textureBarrier.pNext = NULL;
    textureBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    textureBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    textureBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    textureBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    textureBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    textureBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    textureBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    textureBarrier.subresourceRange.baseMipLevel = 0;
    textureBarrier.subresourceRange.levelCount = 1;
    textureBarrier.subresourceRange.baseArrayLayer = 0;
    textureBarrier.subresourceRange.layerCount = 1;
    textureBarrier.image = info.textures[0].image;
    vkCmdPipelineBarrier(info.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &textureBarrier);

    clear_color[0].float32[0] = 0.0f;
    clear_color[0].float32[1] = 1.0f;
    clear_color[0].float32[2] = 0.0f;
    clear_color[0].float32[3] = 1.0f;
    /* Clear texture to green */
    vkCmdClearColorImage(info.cmd, info.textures[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clear_color, 1, &srRange);

    // Send a barrier to change the texture image's layout back to SHADER_READ_ONLY
    // because we're going to use it as a texture again
    textureBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    textureBarrier.pNext = NULL;
    textureBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    textureBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    textureBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    textureBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    textureBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    textureBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    textureBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    textureBarrier.subresourceRange.baseMipLevel = 0;
    textureBarrier.subresourceRange.levelCount = 1;
    textureBarrier.subresourceRange.baseArrayLayer = 0;
    textureBarrier.subresourceRange.layerCount = 1;
    textureBarrier.image = info.textures[0].image;
    vkCmdPipelineBarrier(info.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &textureBarrier);

    // Draw the second quad to the right using the (now) green texture
    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    // Draw starting with vertex index 6 to draw to the right of the first quad
    vkCmdDraw(info.cmd, 2 * 3, 1, 6, 0);
    vkCmdEndRenderPass(info.cmd);

    // Change the present buffer from COLOR_ATTACHMENT_OPTIMAL to
    // PRESENT_SOURCE_KHR
    // so it can be presented
    execute_pre_present_barrier(info);

    res = vkEndCommandBuffer(info.cmd);
    assert(res == VK_SUCCESS);

    VkSubmitInfo submit_info = {};
    VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    init_submit_info(info, submit_info, pipe_stage_flags);
    assert(res == VK_SUCCESS);

    VkFence drawFence = {};
    init_fence(info, drawFence);

    // Queue the command buffer for execution
    res = vkQueueSubmit(info.graphics_queue, 1, &submit_info, drawFence);
    assert(res == VK_SUCCESS);

    // Now present the image in the window
    VkPresentInfoKHR present{};
    init_present_info(info, present);

    // Make sure command buffer is finished before presenting
    do {
        res = vkWaitForFences(info.device, 1, &drawFence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);
    res = vkQueuePresentKHR(info.present_queue, &present);
    assert(res == VK_SUCCESS);
    /* VULKAN_KEY_END */

    wait_seconds(1);
    if (info.save_images) write_ppm(info, "memory_barriers");

    vkDestroySemaphore(info.device, info.imageAcquiredSemaphore, NULL);
    vkDestroyFence(info.device, drawFence, NULL);
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
    destroy_swap_chain(info);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_window(info);
    destroy_device(info);
    destroy_instance(info);
    return 0;
}

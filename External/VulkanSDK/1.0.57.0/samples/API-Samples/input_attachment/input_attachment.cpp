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
Use an input attachment to draw a yellow triangle
*/

#include <assert.h>
#include <cstdlib>
#include <string.h>
#include <util_init.hpp>

/* For this sample, we'll start with GLSL so the shader function is plain */
/* and then use the glslang GLSLtoSPV utility to convert it to SPIR-V for */
/* the driver.  We do this for clarity rather than using pre-compiled     */
/* SPIR-V                                                                 */

static const char *vertShaderText =
    "#version 450\n"
    "vec2 vertices[3];\n"
    "void main() {\n"
    "    vertices[0] = vec2(-1.0, -1.0);\n"
    "    vertices[1] = vec2( 1.0, -1.0);\n"
    "    vertices[2] = vec2( 0.0,  1.0);\n"
    "    gl_Position = vec4(vertices[gl_VertexIndex % 3], 0.0, 1.0);\n"
    "}\n";

// Use subpassLoad to read from input attachment
static const char *fragShaderText =
    "#version 450\n"
    "layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput "
    "myInputAttachment;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main() {\n"
    "   outColor = subpassLoad(myInputAttachment);\n"
    "}\n";

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    bool U_ASSERT_ONLY pass;
    struct sample_info info = {};
    char sample_title[] = "Input Attachment Sample";
    const bool depthPresent = false;
    const bool vertexPresent = false;

    process_command_line_args(info, argc, argv);
    init_global_layer_properties(info);
    init_instance_extension_names(info);
    init_device_extension_names(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);

    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(info.gpus[0], VK_FORMAT_R8G8B8A8_UNORM, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
        std::cout << "VK_FORMAT_R8G8B8A8_UNORM format unsupported for input "
                     "attachment\n";
        exit(-1);
    }

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

    /* VULKAN_KEY_START */

    // Create a framebuffer with 2 attachments, one the color attachment
    // the shaders render into, and the other an input attachment which
    // will be cleared to yellow, and then used by the shaders to color
    // the drawn triangle. Final result should be a yellow triangle

    // Create the image that will be used as the input attachment
    // The image for the color attachment is the presentable image already
    // created in init_swapchain()
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.pNext = NULL;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = info.format;
    image_create_info.extent.width = info.width;
    image_create_info.extent.height = info.height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = NUM_SAMPLES;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.queueFamilyIndexCount = 0;
    image_create_info.pQueueFamilyIndices = NULL;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.flags = 0;

    VkMemoryAllocateInfo mem_alloc = {};
    mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alloc.pNext = NULL;
    mem_alloc.allocationSize = 0;
    mem_alloc.memoryTypeIndex = 0;

    VkImage input_image;
    VkDeviceMemory input_memory;

    res = vkCreateImage(info.device, &image_create_info, NULL, &input_image);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(info.device, input_image, &mem_reqs);

    mem_alloc.allocationSize = mem_reqs.size;

    pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits, 0, &mem_alloc.memoryTypeIndex);
    assert(pass);

    res = vkAllocateMemory(info.device, &mem_alloc, NULL, &input_memory);
    assert(res == VK_SUCCESS);

    res = vkBindImageMemory(info.device, input_image, input_memory, 0);
    assert(res == VK_SUCCESS);

    // Set the image layout to TRANSFER_DST_OPTIMAL to be ready for clear
    set_image_layout(info, input_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageSubresourceRange srRange = {};
    srRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    srRange.baseMipLevel = 0;
    srRange.levelCount = VK_REMAINING_MIP_LEVELS;
    srRange.baseArrayLayer = 0;
    srRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkClearColorValue clear_color;
    clear_color.float32[0] = 1.0f;
    clear_color.float32[1] = 1.0f;
    clear_color.float32[2] = 0.0f;
    clear_color.float32[3] = 0.0f;
    // Clear the input attachment image to yellow
    vkCmdClearColorImage(info.cmd, input_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &srRange);

    // Set the image layout to SHADER_READONLY_OPTIMAL for use by the shaders
    set_image_layout(info, input_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.pNext = NULL;
    view_info.image = VK_NULL_HANDLE;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = info.format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_R;
    view_info.components.g = VK_COMPONENT_SWIZZLE_G;
    view_info.components.b = VK_COMPONENT_SWIZZLE_B;
    view_info.components.a = VK_COMPONENT_SWIZZLE_A;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView input_attachment_view;
    view_info.image = input_image;
    res = vkCreateImageView(info.device, &view_info, NULL, &input_attachment_view);
    assert(res == VK_SUCCESS);

    VkDescriptorImageInfo input_image_info = {};
    input_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    input_image_info.imageView = input_attachment_view;
    input_image_info.sampler = VK_NULL_HANDLE;

    VkDescriptorSetLayoutBinding layout_bindings[1];
    layout_bindings[0].binding = 0;
    layout_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    layout_bindings[0].descriptorCount = 1;
    layout_bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    layout_bindings[0].pImmutableSamplers = NULL;

    VkDescriptorSetLayoutCreateInfo descriptor_layout = {};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.pNext = NULL;
    descriptor_layout.bindingCount = 1;
    descriptor_layout.pBindings = layout_bindings;

    info.desc_layout.resize(NUM_DESCRIPTOR_SETS);
    res = vkCreateDescriptorSetLayout(info.device, &descriptor_layout, NULL, info.desc_layout.data());
    assert(res == VK_SUCCESS);

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
    pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pPipelineLayoutCreateInfo.pNext = NULL;
    pPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
    pPipelineLayoutCreateInfo.setLayoutCount = NUM_DESCRIPTOR_SETS;
    pPipelineLayoutCreateInfo.pSetLayouts = info.desc_layout.data();

    res = vkCreatePipelineLayout(info.device, &pPipelineLayoutCreateInfo, NULL, &info.pipeline_layout);
    assert(res == VK_SUCCESS);

    // First attachment is the color attachment - clear at the beginning of the
    // renderpass and transition layout to PRESENT_SRC_KHR at the end of
    // renderpass
    VkAttachmentDescription attachments[2];
    attachments[0].format = info.format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments[0].flags = 0;

    // Second attachment is input attachment.  Once cleared it should have
    // width*height yellow pixels.  Doing a subpassLoad in the fragment shader
    // should give the shader the color at the fragments x,y location
    // from the input attachment
    attachments[1].format = info.format;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    attachments[1].flags = 0;

    VkAttachmentReference color_reference = {};
    color_reference.attachment = 0;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference input_reference = {};
    input_reference.attachment = 1;
    input_reference.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 1;
    subpass.pInputAttachments = &input_reference;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pResolveAttachments = NULL;
    subpass.pDepthStencilAttachment = NULL;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.pNext = NULL;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 0;
    rp_info.pDependencies = NULL;

    res = vkCreateRenderPass(info.device, &rp_info, NULL, &info.render_pass);
    assert(!res);

    init_shaders(info, vertShaderText, fragShaderText);

    VkImageView fb_attachments[2];
    fb_attachments[1] = input_attachment_view;

    VkFramebufferCreateInfo fbc_info = {};
    fbc_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbc_info.pNext = NULL;
    fbc_info.renderPass = info.render_pass;
    fbc_info.attachmentCount = 2;
    fbc_info.pAttachments = fb_attachments;
    fbc_info.width = info.width;
    fbc_info.height = info.height;
    fbc_info.layers = 1;

    uint32_t i;

    info.framebuffers = (VkFramebuffer *)malloc(info.swapchainImageCount * sizeof(VkFramebuffer));

    for (i = 0; i < info.swapchainImageCount; i++) {
        fb_attachments[0] = info.buffers[i].view;
        res = vkCreateFramebuffer(info.device, &fbc_info, NULL, &info.framebuffers[i]);
        assert(res == VK_SUCCESS);
    }

    VkDescriptorPoolSize type_count[1];
    type_count[0].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    type_count[0].descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptor_pool = {};
    descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool.pNext = NULL;
    descriptor_pool.maxSets = 1;
    descriptor_pool.poolSizeCount = 1;
    descriptor_pool.pPoolSizes = type_count;

    res = vkCreateDescriptorPool(info.device, &descriptor_pool, NULL, &info.desc_pool);
    assert(res == VK_SUCCESS);

    VkDescriptorSetAllocateInfo desc_alloc_info[1];
    desc_alloc_info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_alloc_info[0].pNext = NULL;
    desc_alloc_info[0].descriptorPool = info.desc_pool;
    desc_alloc_info[0].descriptorSetCount = 1;
    desc_alloc_info[0].pSetLayouts = info.desc_layout.data();

    info.desc_set.resize(1);
    res = vkAllocateDescriptorSets(info.device, desc_alloc_info, info.desc_set.data());
    assert(res == VK_SUCCESS);

    VkWriteDescriptorSet writes[1];

    // Write descriptor set with one write describing input attachment
    writes[0] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = info.desc_set[0];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    writes[0].pImageInfo = &input_image_info;
    writes[0].pBufferInfo = nullptr;
    writes[0].pTexelBufferView = nullptr;
    writes[0].dstArrayElement = 0;

    vkUpdateDescriptorSets(info.device, 1, writes, 0, NULL);

    init_pipeline_cache(info);
    init_pipeline(info, depthPresent, vertexPresent);

    // Color attachment clear to gray
    VkClearValue clear_values;
    clear_values.color.float32[0] = 0.2f;
    clear_values.color.float32[1] = 0.2f;
    clear_values.color.float32[2] = 0.2f;
    clear_values.color.float32[3] = 0.2f;

    VkSemaphoreCreateInfo imageAcquiredSemaphoreCreateInfo;
    imageAcquiredSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    imageAcquiredSemaphoreCreateInfo.pNext = NULL;
    imageAcquiredSemaphoreCreateInfo.flags = 0;

    res = vkCreateSemaphore(info.device, &imageAcquiredSemaphoreCreateInfo, NULL, &info.imageAcquiredSemaphore);
    assert(res == VK_SUCCESS);

    // Get the index of the next available swapchain image:
    res = vkAcquireNextImageKHR(info.device, info.swap_chain, UINT64_MAX, info.imageAcquiredSemaphore, VK_NULL_HANDLE,
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
    rp_begin.clearValueCount = 1;
    rp_begin.pClearValues = &clear_values;

    vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline);

    vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            info.desc_set.data(), 0, NULL);

    init_viewports(info);
    init_scissors(info);

    vkCmdDraw(info.cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(info.cmd);
    res = vkEndCommandBuffer(info.cmd);
    assert(res == VK_SUCCESS);

    /* VULKAN_KEY_END */

    const VkCommandBuffer cmd_bufs[] = {info.cmd};

    VkFenceCreateInfo fenceInfo;
    VkFence drawFence;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(info.device, &fenceInfo, NULL, &drawFence);

    execute_queue_cmdbuf(info, cmd_bufs, drawFence);

    do {
        res = vkWaitForFences(info.device, 1, &drawFence, VK_TRUE, FENCE_TIMEOUT);
    } while (res == VK_TIMEOUT);
    assert(res == VK_SUCCESS);
    vkDestroyFence(info.device, drawFence, NULL);

    execute_present_image(info);

    wait_seconds(1);

    if (info.save_images) write_ppm(info, "input_attachment");

    vkDestroySemaphore(info.device, info.imageAcquiredSemaphore, NULL);
    vkDestroyImageView(info.device, input_attachment_view, NULL);
    vkDestroyImage(info.device, input_image, NULL);
    vkFreeMemory(info.device, input_memory, NULL);
    destroy_pipeline(info);
    destroy_pipeline_cache(info);
    destroy_descriptor_pool(info);
    destroy_framebuffers(info);
    destroy_shaders(info);
    destroy_renderpass(info);
    destroy_descriptor_and_pipeline_layouts(info);
    destroy_swap_chain(info);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_device(info);
    destroy_window(info);
    destroy_instance(info);
    return 0;
}

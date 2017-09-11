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
Initialize Framebuffer
*/

/* This is part of the draw cube progression */

#include <util_init.hpp>
#include <assert.h>
#include <cstdlib>

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Init Framebuffer Sample";
    const bool depthPresent = true;

    init_global_layer_properties(info);
    init_instance_extension_names(info);
    init_device_extension_names(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_connection(info);
    init_window_size(info, 50, 50);
    init_window(info);
    init_swapchain_extension(info);
    init_device(info);
    init_command_pool(info);
    init_command_buffer(info);
    execute_begin_command_buffer(info);
    init_device_queue(info);
    init_swap_chain(info);
    init_depth_buffer(info);
    init_renderpass(info, depthPresent);

    /* VULKAN_KEY_START */
    VkImageView attachments[2];
    attachments[1] = info.depth.view;

    VkFramebufferCreateInfo fb_info = {};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = NULL;
    fb_info.renderPass = info.render_pass;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = attachments;
    fb_info.width = info.width;
    fb_info.height = info.height;
    fb_info.layers = 1;

    uint32_t i;
    info.framebuffers = (VkFramebuffer *)malloc(info.swapchainImageCount * sizeof(VkFramebuffer));
    assert(info.framebuffers);

    for (i = 0; i < info.swapchainImageCount; i++) {
        attachments[0] = info.buffers[i].view;
        res = vkCreateFramebuffer(info.device, &fb_info, NULL, &info.framebuffers[i]);
        assert(res == VK_SUCCESS);
    }
    execute_end_command_buffer(info);
    execute_queue_command_buffer(info);
    /* VULKAN_KEY_END */

    for (i = 0; i < info.swapchainImageCount; i++) {
        vkDestroyFramebuffer(info.device, info.framebuffers[i], NULL);
    }
    free(info.framebuffers);

    destroy_renderpass(info);
    destroy_depth_buffer(info);
    destroy_swap_chain(info);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_device(info);
    destroy_window(info);
    destroy_instance(info);
    return 0;
}

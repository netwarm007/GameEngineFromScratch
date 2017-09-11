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
Create Vulkan command buffer
*/

/* This is part of the draw cube progression */

#include <util_init.hpp>
#include <assert.h>
#include <cstdlib>

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Command Buffer Sample";

    init_global_layer_properties(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_device(info);
    init_queue_family_index(info);

    /* VULKAN_KEY_START */

    /* Create a command pool to allocate our command buffer from */
    VkCommandPoolCreateInfo cmd_pool_info = {};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.pNext = NULL;
    cmd_pool_info.queueFamilyIndex = info.graphics_queue_family_index;
    cmd_pool_info.flags = 0;

    res = vkCreateCommandPool(info.device, &cmd_pool_info, NULL, &info.cmd_pool);
    assert(res == VK_SUCCESS);

    /* Create the command buffer from the command pool */
    VkCommandBufferAllocateInfo cmd = {};
    cmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd.pNext = NULL;
    cmd.commandPool = info.cmd_pool;
    cmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(info.device, &cmd, &info.cmd);
    assert(res == VK_SUCCESS);

    /* VULKAN_KEY_END */

    VkCommandBuffer cmd_bufs[1] = {info.cmd};
    vkFreeCommandBuffers(info.device, info.cmd_pool, 1, cmd_bufs);
    vkDestroyCommandPool(info.device, info.cmd_pool, NULL);
    destroy_device(info);
    destroy_instance(info);
    return 0;
}

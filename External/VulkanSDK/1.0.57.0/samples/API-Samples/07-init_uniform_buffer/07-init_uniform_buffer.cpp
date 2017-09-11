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
Create Uniform Buffer
*/

/* This is part of the draw cube progression */

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    bool U_ASSERT_ONLY pass;
    struct sample_info info = {};
    char sample_title[] = "Uniform Buffer Sample";

    init_global_layer_properties(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_queue_family_index(info);
    init_device(info);
    init_window_size(info, 50, 50);

    info.Projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    info.View = glm::lookAt(glm::vec3(-5, 3, -10),  // Camera is at (-5,3,-10), in World Space
                            glm::vec3(0, 0, 0),     // and looks at the origin
                            glm::vec3(0, -1, 0)     // Head is up (set to 0,-1,0 to look upside-down)
                            );
    info.Model = glm::mat4(1.0f);

    // Vulkan clip space has inverted Y and half Z.
    // clang-format off
    info.Clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f,-1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.5f, 0.0f,
                          0.0f, 0.0f, 0.5f, 1.0f);
    // clang-format on
    info.MVP = info.Clip * info.Projection * info.View * info.Model;

    /* VULKAN_KEY_START */
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = NULL;
    buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf_info.size = sizeof(info.MVP);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_info.flags = 0;
    res = vkCreateBuffer(info.device, &buf_info, NULL, &info.uniform_data.buf);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(info.device, info.uniform_data.buf, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = NULL;
    alloc_info.memoryTypeIndex = 0;

    alloc_info.allocationSize = mem_reqs.size;
    pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex);
    assert(pass && "No mappable, coherent memory");

    res = vkAllocateMemory(info.device, &alloc_info, NULL, &(info.uniform_data.mem));
    assert(res == VK_SUCCESS);

    uint8_t *pData;
    res = vkMapMemory(info.device, info.uniform_data.mem, 0, mem_reqs.size, 0, (void **)&pData);
    assert(res == VK_SUCCESS);

    memcpy(pData, &info.MVP, sizeof(info.MVP));

    vkUnmapMemory(info.device, info.uniform_data.mem);

    res = vkBindBufferMemory(info.device, info.uniform_data.buf, info.uniform_data.mem, 0);
    assert(res == VK_SUCCESS);

    info.uniform_data.buffer_info.buffer = info.uniform_data.buf;
    info.uniform_data.buffer_info.offset = 0;
    info.uniform_data.buffer_info.range = sizeof(info.MVP);
    /* VULKAN_KEY_END */

    vkDestroyBuffer(info.device, info.uniform_data.buf, NULL);
    vkFreeMemory(info.device, info.uniform_data.mem, NULL);
    destroy_device(info);
    destroy_instance(info);
    return 0;
}

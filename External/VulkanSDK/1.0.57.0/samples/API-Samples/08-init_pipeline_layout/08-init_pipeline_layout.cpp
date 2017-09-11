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
Create Descriptor Layout and Pipeline Layout
*/

/* This is part of the draw cube progression */

#include <util_init.hpp>
#include <assert.h>
#include <cstdlib>

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    char sample_title[] = "Descriptor / Pipeline Layout Sample";

    init_global_layer_properties(info);
    init_instance(info, sample_title);
    init_enumerate_device(info);
    init_queue_family_index(info);
    init_device(info);

    /* VULKAN_KEY_START */
    /* Start with just our uniform buffer that has our transformation matrices
     * (for the vertex shader). The fragment shader we intend to use needs no
     * external resources, so nothing else is necessary
     */

    /* Note that when we start using textures, this is where our sampler will
     * need to be specified
     */
    VkDescriptorSetLayoutBinding layout_binding = {};
    layout_binding.binding = 0;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    layout_binding.pImmutableSamplers = NULL;

    /* Next take layout bindings and use them to create a descriptor set layout
     */
    VkDescriptorSetLayoutCreateInfo descriptor_layout = {};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.pNext = NULL;
    descriptor_layout.bindingCount = 1;
    descriptor_layout.pBindings = &layout_binding;

    info.desc_layout.resize(NUM_DESCRIPTOR_SETS);
    res = vkCreateDescriptorSetLayout(info.device, &descriptor_layout, NULL, info.desc_layout.data());
    assert(res == VK_SUCCESS);

    /* Now use the descriptor layout to create a pipeline layout */
    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
    pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pPipelineLayoutCreateInfo.pNext = NULL;
    pPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
    pPipelineLayoutCreateInfo.setLayoutCount = NUM_DESCRIPTOR_SETS;
    pPipelineLayoutCreateInfo.pSetLayouts = info.desc_layout.data();

    res = vkCreatePipelineLayout(info.device, &pPipelineLayoutCreateInfo, NULL, &info.pipeline_layout);
    assert(res == VK_SUCCESS);
    /* VULKAN_KEY_END */

    for (int i = 0; i < NUM_DESCRIPTOR_SETS; i++) vkDestroyDescriptorSetLayout(info.device, info.desc_layout[i], NULL);
    vkDestroyPipelineLayout(info.device, info.pipeline_layout, NULL);
    destroy_device(info);
    destroy_instance(info);
    return 0;
}

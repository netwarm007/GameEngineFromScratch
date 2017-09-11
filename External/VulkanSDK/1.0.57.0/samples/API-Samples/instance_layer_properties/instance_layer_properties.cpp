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
Get global layer properties to know what
layers are available to enable at CreateInstance time.
*/

#include <util_init.hpp>
#include <cstdlib>

int sample_main(int argc, char *argv[]) {
    VkResult res;
    struct sample_info info;
    uint32_t instance_layer_count;
    VkLayerProperties *vk_props = NULL;

    init_global_layer_properties(info);

    /* VULKAN_KEY_START */

    /*
     * It's possible, though very rare, that the number of
     * instance layers could change. For example, installing something
     * could include new layers that the loader would pick up
     * between the initial query for the count and the
     * request for VkLayerProperties. The loader indicates that
     * by returning a VK_INCOMPLETE status and will update the
     * the count parameter.
     * The count parameter will be updated with the number of
     * entries loaded into the data pointer - in case the number
     * of layers went down or is smaller than the size given.
     */
    do {
        res = vkEnumerateInstanceLayerProperties(&instance_layer_count, NULL);
        if (res) break;

        if (instance_layer_count == 0) {
            break;
        }

        vk_props = (VkLayerProperties *)realloc(vk_props, instance_layer_count * sizeof(VkLayerProperties));

        res = vkEnumerateInstanceLayerProperties(&instance_layer_count, vk_props);
    } while (res == VK_INCOMPLETE);

    std::cout << "Instance Layers:" << std::endl;
    for (uint32_t i = 0; i < instance_layer_count; i++) {
        VkLayerProperties *props = &vk_props[i];
        uint32_t major, minor, patch;
        std::cout << props->layerName << ":" << std::endl;
        extract_version(props->specVersion, major, minor, patch);
        std::cout << "\tVersion: " << props->implementationVersion << std::endl;
        std::cout << "\tAPI Version: "
                  << "(" << major << "." << minor << "." << patch << ")" << std::endl;
        std::cout << "\tDescription: " << props->description << std::endl;
        std::cout << std::endl << std::endl;
    }

    if (instance_layer_count == 0) {
        std::cout << "Set the environment variable VK_LAYER_PATH to point to the location of your layers" << std::endl;
    }

    std::cout << std::endl;

    free(vk_props);

    /* VULKAN_KEY_END */

    return 0;
}

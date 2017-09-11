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
Get global extension properties to know what
extension are available to enable at CreateInstance time.
*/

#include <util_init.hpp>
#include <cstdlib>

int sample_main(int argc, char *argv[]) {
    VkResult res;
    VkExtensionProperties *vk_props = NULL;
    uint32_t instance_extension_count;

    struct sample_info info = {};
    init_global_layer_properties(info);

    /* VULKAN_KEY_START */

    /*
     * It's possible, though very rare, that the number of
     * instance layers could change. For example, installing something
     * could include new layers that the loader would pick up
     * between the initial query for the count and the
     * request for VkLayerProperties. If that happens,
     * the number of VkLayerProperties could exceed the count
     * previously given. To alert the app to this change
     * vkEnumerateInstanceExtensionProperties will return a VK_INCOMPLETE
     * status.
     * The count parameter will be updated with the number of
     * entries actually loaded into the data pointer.
     */

    do {
        res = vkEnumerateInstanceExtensionProperties(NULL, &instance_extension_count, NULL);
        if (res) break;

        if (instance_extension_count == 0) {
            break;
        }

        vk_props = (VkExtensionProperties *)realloc(vk_props, instance_extension_count * sizeof(VkExtensionProperties));

        res = vkEnumerateInstanceExtensionProperties(NULL, &instance_extension_count, vk_props);
    } while (res == VK_INCOMPLETE);

    std::cout << "Instance Extensions:" << std::endl;
    for (uint32_t i = 0; i < instance_extension_count; i++) {
        VkExtensionProperties *props = &vk_props[i];
        std::cout << props->extensionName << ":" << std::endl;
        std::cout << "\tVersion: " << props->specVersion << std::endl;
        std::cout << std::endl << std::endl;
    }

    std::cout << std::endl;

    /* VULKAN_KEY_END */

    return 0;
}

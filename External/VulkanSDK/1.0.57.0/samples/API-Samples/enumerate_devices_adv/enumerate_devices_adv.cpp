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
enumerate physical devices
*/

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <util_init.hpp>

int sample_main(int argc, char *argv[]) {
    struct sample_info info = {};
    init_global_layer_properties(info);
    init_instance(info, "vulkansamples_enumerate");

    /* VULKAN_KEY_START */

    // Query the count.
    uint32_t gpu_count = 0;
    VkResult U_ASSERT_ONLY res = vkEnumeratePhysicalDevices(info.inst, &gpu_count, NULL);
    assert(!res && gpu_count > 0);

    // Query the gpu info.
    VkPhysicalDevice *gpu = new VkPhysicalDevice[gpu_count];
    res = vkEnumeratePhysicalDevices(info.inst, &gpu_count, gpu);
    assert(res == VK_SUCCESS);

    for (uint32_t i = 0; i < gpu_count; ++i) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(gpu[i], &properties);

        std::cout << "apiVersion: ";
        std::cout << ((properties.apiVersion >> 22) & 0xfff) << '.';  // Major.
        std::cout << ((properties.apiVersion >> 12) & 0x3ff) << '.';  // Minor.
        std::cout << (properties.apiVersion & 0xfff);                 // Patch.
        std::cout << '\n';

        std::cout << "driverVersion: " << properties.driverVersion << '\n';

        std::cout << std::showbase << std::internal << std::setfill('0') << std::hex;
        std::cout << "vendorId: " << std::setw(6) << properties.vendorID << '\n';
        std::cout << "deviceId: " << std::setw(6) << properties.deviceID << '\n';
        std::cout << std::noshowbase << std::right << std::setfill(' ') << std::dec;

        std::cout << "deviceType: ";
        switch (properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                std::cout << "VK_PHYSICAL_DEVICE_TYPE_OTHER";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                std::cout << "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                std::cout << "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                std::cout << "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU";
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                std::cout << "VK_PHYSICAL_DEVICE_TYPE_CPU";
                break;
            default:
                break;
        }
        std::cout << '\n';

        std::cout << "deviceName: " << properties.deviceName << '\n';

        std::cout << "pipelineCacheUUID: ";
        std::cout << std::setfill('0') << std::hex;
        print_UUID(properties.pipelineCacheUUID);
        std::cout << std::setfill(' ') << std::dec;
        std::cout << '\n';
        std::cout << '\n';
    }

    delete[] gpu;

    /* VULKAN_KEY_END */

    vkDestroyInstance(info.inst, NULL);

    return 0;
}

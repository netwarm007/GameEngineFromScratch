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
Show how to enable validation layers and provide callback
*/

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <util_init.hpp>

VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT msgFlags, VkDebugReportObjectTypeEXT objType, uint64_t srcObject,
                                       size_t location, int32_t msgCode, const char *pLayerPrefix, const char *pMsg,
                                       void *pUserData) {
    std::ostringstream message;

    if (msgFlags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        message << "ERROR: ";
    } else if (msgFlags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        message << "WARNING: ";
    } else if (msgFlags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
        message << "PERFORMANCE WARNING: ";
    } else if (msgFlags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        message << "INFO: ";
    } else if (msgFlags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
        message << "DEBUG: ";
    }
    message << "[" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;

    std::cout << message.str() << std::endl;

    /*
     * false indicates that layer should not bail-out of an
     * API call that had validation failures. This may mean that the
     * app dies inside the driver due to invalid parameter(s).
     * That's what would happen without validation layers, so we'll
     * keep that behavior here.
     */
    return false;
}

int sample_main(int argc, char *argv[]) {
    struct sample_info info = {};
    init_global_layer_properties(info);

    /* VULKAN_KEY_START */

    /* Use standard_validation meta layer that enables all
     * recommended validation layers
     */
    info.instance_layer_names.push_back("VK_LAYER_LUNARG_standard_validation");
    if (!demo_check_layers(info.instance_layer_properties, info.instance_layer_names)) {
        /* If standard validation is not present, search instead for the
         * individual layers that make it up, in the correct order.
         */
        info.instance_layer_names.clear();
        info.instance_layer_names.push_back("VK_LAYER_GOOGLE_threading");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_parameter_validation");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_object_tracker");
        info.instance_layer_names.push_back("VK_LAYER_LUNARG_core_validation");
        info.instance_layer_names.push_back("VK_LAYER_GOOGLE_unique_objects");

        if (!demo_check_layers(info.instance_layer_properties, info.instance_layer_names)) {
            std::cout << "Set the environment variable VK_LAYER_PATH to point to the location of your layers" << std::endl;
            exit(1);
        }
    }

    /* Enable debug callback extension */
    info.instance_extension_names.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = NULL;
    app_info.pApplicationName = "vulkansamples_enable_validation_with_callback";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "Vulkan Samples";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_info = {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = NULL;
    inst_info.flags = 0;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = info.instance_layer_names.size();
    inst_info.ppEnabledLayerNames = info.instance_layer_names.size() ? info.instance_layer_names.data() : NULL;
    inst_info.enabledExtensionCount = info.instance_extension_names.size();
    inst_info.ppEnabledExtensionNames = info.instance_extension_names.data();

    VkResult res = vkCreateInstance(&inst_info, NULL, &info.inst);
    assert(res == VK_SUCCESS);

    init_enumerate_device(info);

    float queue_priorities[1] = {0.0};
    VkDeviceQueueCreateInfo queue_info = {};

    vkGetPhysicalDeviceQueueFamilyProperties(info.gpus[0], &info.queue_family_count, NULL);
    assert(info.queue_family_count >= 1);

    info.queue_props.resize(info.queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(info.gpus[0], &info.queue_family_count, info.queue_props.data());
    assert(info.queue_family_count >= 1);

    bool found = false;
    for (unsigned int i = 0; i < info.queue_family_count; i++) {
        if (info.queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_info.queueFamilyIndex = i;
            found = true;
            break;
        }
    }
    assert(found);
    assert(info.queue_family_count >= 1);

    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.pNext = NULL;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priorities;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = NULL;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = info.device_extension_names.size();
    device_info.ppEnabledExtensionNames = device_info.enabledExtensionCount ? info.device_extension_names.data() : NULL;
    device_info.pEnabledFeatures = NULL;

    res = vkCreateDevice(info.gpus[0], &device_info, NULL, &info.device);
    assert(res == VK_SUCCESS);

    VkDebugReportCallbackEXT debug_report_callback;

    info.dbgCreateDebugReportCallback =
        (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(info.inst, "vkCreateDebugReportCallbackEXT");
    if (!info.dbgCreateDebugReportCallback) {
        std::cout << "GetInstanceProcAddr: Unable to find "
                     "vkCreateDebugReportCallbackEXT function."
                  << std::endl;
        exit(1);
    }

    info.dbgDestroyDebugReportCallback =
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(info.inst, "vkDestroyDebugReportCallbackEXT");
    if (!info.dbgDestroyDebugReportCallback) {
        std::cout << "GetInstanceProcAddr: Unable to find "
                     "vkDestroyDebugReportCallbackEXT function."
                  << std::endl;
        exit(1);
    }

    VkDebugReportCallbackCreateInfoEXT create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    create_info.pNext = NULL;
    create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    create_info.pfnCallback = dbgFunc;
    create_info.pUserData = NULL;
    res = info.dbgCreateDebugReportCallback(info.inst, &create_info, NULL, &debug_report_callback);
    switch (res) {
        case VK_SUCCESS:
            break;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            std::cout << "dbgCreateDebugReportCallback: out of host memory\n" << std::endl;
            exit(1);
            break;
        default:
            std::cout << "dbgCreateDebugReportCallback: unknown failure\n" << std::endl;
            exit(1);
            break;
    }

    /* Create a command pool */
    VkCommandPoolCreateInfo cmd_pool_info = {};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.pNext = NULL;
    cmd_pool_info.queueFamilyIndex = info.graphics_queue_family_index;
    cmd_pool_info.flags = 0;

    res = vkCreateCommandPool(info.device, &cmd_pool_info, NULL, &info.cmd_pool);
    assert(res == VK_SUCCESS);

    /*
     * Destroying the device before destroying the command pool above
     * will trigger a validation error.
     */
    std::cout << "*** INTENTIONALLY calling vkDestroyDevice before destroying command pool ***\n";
    std::cout << "*** The following error message is EXPECTED ***\n";
    vkDestroyDevice(info.device, NULL);

    /* Clean up callback */
    info.dbgDestroyDebugReportCallback(info.inst, debug_report_callback, NULL);

    /* VULKAN_KEY_END */

    vkDestroyInstance(info.inst, NULL);

    return 0;
}

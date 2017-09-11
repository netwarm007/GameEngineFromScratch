/* Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google Inc.
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
 *
 * Author: Jeremy Hayes <jeremy@lunarg.com>
 * Author: Tony Barbour <tony@LunarG.com>
 * Author: Mark Lobodzinski <mark@LunarG.com>
 * Author: Dustin Graves <dustin@lunarg.com>
 * Author: Chris Forbes <chrisforbes@google.com>
 */

#define NOMINMAX

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>

#include "vk_loader_platform.h"
#include "vulkan/vk_layer.h"
#include "vk_layer_config.h"
#include "vk_dispatch_table_helper.h"

#include "vk_layer_table.h"
#include "vk_layer_data.h"
#include "vk_layer_logging.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_utils.h"

#include "parameter_name.h"
#include "parameter_validation.h"

// TODO: remove on NDK update (r15 will probably have proper STL impl)
#ifdef __ANDROID__
namespace std {

template <typename T>
std::string to_string(T var) {
    std::ostringstream ss;
    ss << var;
    return ss.str();
}
}
#endif

namespace parameter_validation {


// TODO : This can be much smarter, using separate locks for separate global data
static std::mutex global_lock;

static uint32_t loader_layer_if_version = CURRENT_LOADER_LAYER_INTERFACE_VERSION;
static std::unordered_map<void *, layer_data *> layer_data_map;
static std::unordered_map<void *, instance_layer_data *> instance_layer_data_map;

static void init_parameter_validation(instance_layer_data *my_data, const VkAllocationCallbacks *pAllocator) {
    layer_debug_actions(my_data->report_data, my_data->logging_callback, pAllocator, "lunarg_parameter_validation");
}



VKAPI_ATTR VkResult VKAPI_CALL CreateDebugReportCallbackEXT(VkInstance instance,
                                                            const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator,
                                                            VkDebugReportCallbackEXT *pMsgCallback) {
    auto data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = data->dispatch_table.CreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pMsgCallback);

    if (result == VK_SUCCESS) {
        result = layer_create_msg_callback(data->report_data, false, pCreateInfo, pAllocator, pMsgCallback);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT msgCallback,
                                                         const VkAllocationCallbacks *pAllocator) {
    auto data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    data->dispatch_table.DestroyDebugReportCallbackEXT(instance, msgCallback, pAllocator);

    layer_destroy_msg_callback(data->report_data, msgCallback, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL DebugReportMessageEXT(VkInstance instance, VkDebugReportFlagsEXT flags,
                                                 VkDebugReportObjectTypeEXT objType, uint64_t object, size_t location,
                                                 int32_t msgCode, const char *pLayerPrefix, const char *pMsg) {
    auto data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    data->dispatch_table.DebugReportMessageEXT(instance, flags, objType, object, location, msgCode, pLayerPrefix, pMsg);
}

static const VkExtensionProperties instance_extensions[] = {{VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_SPEC_VERSION}};

static const VkLayerProperties global_layer = {
    "VK_LAYER_LUNARG_parameter_validation", VK_LAYER_API_VERSION, 1, "LunarG Validation Layer",
};

template <typename T>
bool OutputExtensionError(const T *layer_data, const std::string &api_name, const std::string &extension_name) {
    return log_msg(layer_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                   EXTENSION_NOT_ENABLED, LayerName, "Attemped to call %s() but its required extension %s has not been enabled\n",
                   api_name.c_str(), extension_name.c_str());
}

static const int MaxParamCheckerStringLength = 256;

static bool validate_string(debug_report_data *report_data, const char *apiName, const ParameterName &stringName,
                            const char *validateString) {
    assert(apiName != nullptr);
    assert(validateString != nullptr);

    bool skip = false;

    VkStringErrorFlags result = vk_string_validate(MaxParamCheckerStringLength, validateString);

    if (result == VK_STRING_ERROR_NONE) {
        return skip;
    } else if (result & VK_STRING_ERROR_LENGTH) {
        skip = log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                       INVALID_USAGE, LayerName, "%s: string %s exceeds max length %d", apiName, stringName.get_name().c_str(),
                       MaxParamCheckerStringLength);
    } else if (result & VK_STRING_ERROR_BAD_DATA) {
        skip = log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                       INVALID_USAGE, LayerName, "%s: string %s contains invalid characters or is badly formed", apiName,
                       stringName.get_name().c_str());
    }
    return skip;
}

static bool ValidateDeviceQueueFamily(layer_data *device_data, uint32_t queue_family, const char *cmd_name,
                                      const char *parameter_name, int32_t error_code, bool optional = false,
                                      const char *vu_note = nullptr) {
    bool skip = false;

    if (!vu_note) vu_note = validation_error_map[error_code];

    if (!optional && queue_family == VK_QUEUE_FAMILY_IGNORED) {
        skip |= log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
                        HandleToUint64(device_data->device), __LINE__, error_code, LayerName,
                        "%s: %s is VK_QUEUE_FAMILY_IGNORED, but it is required to provide a valid queue family index value. %s",
                        cmd_name, parameter_name, vu_note);
    } else if (device_data->queueFamilyIndexMap.find(queue_family) == device_data->queueFamilyIndexMap.end()) {
        skip |= log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
                        HandleToUint64(device_data->device), __LINE__, error_code, LayerName,
                        "%s: %s (= %" PRIu32
                        ") is not one of the queue families given via VkDeviceQueueCreateInfo structures when "
                        "the device was created. %s",
                        cmd_name, parameter_name, queue_family, vu_note);
    }

    return skip;
}

static bool ValidateQueueFamilies(layer_data *device_data, uint32_t queue_family_count, const uint32_t *queue_families,
                                  const char *cmd_name, const char *array_parameter_name, int32_t unique_error_code,
                                  int32_t valid_error_code, bool optional = false, const char *unique_vu_note = nullptr,
                                  const char *valid_vu_note = nullptr) {
    bool skip = false;

    if (!unique_vu_note) unique_vu_note = validation_error_map[unique_error_code];
    if (!valid_vu_note) valid_vu_note = validation_error_map[valid_error_code];

    if (queue_families) {
        std::unordered_set<uint32_t> set;

        for (uint32_t i = 0; i < queue_family_count; ++i) {
            std::string parameter_name = std::string(array_parameter_name) + "[" + std::to_string(i) + "]";

            if (set.count(queue_families[i])) {
                skip |= log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
                                HandleToUint64(device_data->device), __LINE__, VALIDATION_ERROR_056002e8, LayerName,
                                "%s: %s (=%" PRIu32 ") is not unique within %s array. %s", cmd_name, parameter_name.c_str(),
                                queue_families[i], array_parameter_name, unique_vu_note);
            } else {
                set.insert(queue_families[i]);

                skip |= ValidateDeviceQueueFamily(device_data, queue_families[i], cmd_name, parameter_name.c_str(),
                                                  valid_error_code, optional, valid_vu_note);
            }
        }
    }

    return skip;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                              VkInstance *pInstance) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    assert(chain_info != nullptr);
    assert(chain_info->u.pLayerInfo != nullptr);

    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(NULL, "vkCreateInstance");
    if (fpCreateInstance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);

    if (result == VK_SUCCESS) {
        auto my_instance_data = GetLayerDataPtr(get_dispatch_key(*pInstance), instance_layer_data_map);
        assert(my_instance_data != nullptr);

        layer_init_instance_dispatch_table(*pInstance, &my_instance_data->dispatch_table, fpGetInstanceProcAddr);
        my_instance_data->instance = *pInstance;
        my_instance_data->report_data =
            debug_report_create_instance(&my_instance_data->dispatch_table, *pInstance, pCreateInfo->enabledExtensionCount,
                                         pCreateInfo->ppEnabledExtensionNames);

        // Look for one or more debug report create info structures
        // and setup a callback(s) for each one found.
        if (!layer_copy_tmp_callbacks(pCreateInfo->pNext, &my_instance_data->num_tmp_callbacks,
                                      &my_instance_data->tmp_dbg_create_infos, &my_instance_data->tmp_callbacks)) {
            if (my_instance_data->num_tmp_callbacks > 0) {
                // Setup the temporary callback(s) here to catch early issues:
                if (layer_enable_tmp_callbacks(my_instance_data->report_data, my_instance_data->num_tmp_callbacks,
                                               my_instance_data->tmp_dbg_create_infos, my_instance_data->tmp_callbacks)) {
                    // Failure of setting up one or more of the callback.
                    // Therefore, clean up and don't use those callbacks:
                    layer_free_tmp_callbacks(my_instance_data->tmp_dbg_create_infos, my_instance_data->tmp_callbacks);
                    my_instance_data->num_tmp_callbacks = 0;
                }
            }
        }

        init_parameter_validation(my_instance_data, pAllocator);
        my_instance_data->extensions.InitFromInstanceCreateInfo(pCreateInfo);

        // Ordinarily we'd check these before calling down the chain, but none of the layer
        // support is in place until now, if we survive we can report the issue now.
        parameter_validation_vkCreateInstance(my_instance_data, pCreateInfo, pAllocator, pInstance);

        if (pCreateInfo->pApplicationInfo) {
            if (pCreateInfo->pApplicationInfo->pApplicationName) {
                validate_string(my_instance_data->report_data, "vkCreateInstance",
                                "pCreateInfo->VkApplicationInfo->pApplicationName",
                                pCreateInfo->pApplicationInfo->pApplicationName);
            }

            if (pCreateInfo->pApplicationInfo->pEngineName) {
                validate_string(my_instance_data->report_data, "vkCreateInstance", "pCreateInfo->VkApplicationInfo->pEngineName",
                                pCreateInfo->pApplicationInfo->pEngineName);
            }
        }

        // Disable the tmp callbacks:
        if (my_instance_data->num_tmp_callbacks > 0) {
            layer_disable_tmp_callbacks(my_instance_data->report_data, my_instance_data->num_tmp_callbacks,
                                        my_instance_data->tmp_callbacks);
        }
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    // Grab the key before the instance is destroyed.
    dispatch_key key = get_dispatch_key(instance);
    bool skip = false;
    auto my_data = GetLayerDataPtr(key, instance_layer_data_map);
    assert(my_data != NULL);

    // Enable the temporary callback(s) here to catch vkDestroyInstance issues:
    bool callback_setup = false;
    if (my_data->num_tmp_callbacks > 0) {
        if (!layer_enable_tmp_callbacks(my_data->report_data, my_data->num_tmp_callbacks, my_data->tmp_dbg_create_infos,
                                        my_data->tmp_callbacks)) {
            callback_setup = true;
        }
    }

    skip |= parameter_validation_vkDestroyInstance(my_data, pAllocator);

    // Disable and cleanup the temporary callback(s):
    if (callback_setup) {
        layer_disable_tmp_callbacks(my_data->report_data, my_data->num_tmp_callbacks, my_data->tmp_callbacks);
    }
    if (my_data->num_tmp_callbacks > 0) {
        layer_free_tmp_callbacks(my_data->tmp_dbg_create_infos, my_data->tmp_callbacks);
        my_data->num_tmp_callbacks = 0;
    }

    if (!skip) {
        my_data->dispatch_table.DestroyInstance(instance, pAllocator);

        // Clean up logging callback, if any
        while (my_data->logging_callback.size() > 0) {
            VkDebugReportCallbackEXT callback = my_data->logging_callback.back();
            layer_destroy_msg_callback(my_data->report_data, callback, pAllocator);
            my_data->logging_callback.pop_back();
        }

        layer_debug_report_destroy_instance(my_data->report_data);
    }

    FreeLayerDataPtr(key, instance_layer_data_map);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                        VkPhysicalDevice *pPhysicalDevices) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkEnumeratePhysicalDevices(my_data, pPhysicalDeviceCount, pPhysicalDevices);

    if (!skip) {
        result = my_data->dispatch_table.EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
        validate_result(my_data->report_data, "vkEnumeratePhysicalDevices", {}, result);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceFeatures(my_data, pFeatures);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceFeatures(physicalDevice, pFeatures);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format,
                                                             VkFormatProperties *pFormatProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceFormatProperties(my_data, format, pFormatProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceImageFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format,
                                                                      VkImageType type, VkImageTiling tiling,
                                                                      VkImageUsageFlags usage, VkImageCreateFlags flags,
                                                                      VkImageFormatProperties *pImageFormatProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceImageFormatProperties(my_data, format, type, tiling, usage, flags,
                                                                          pImageFormatProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceImageFormatProperties(physicalDevice, format, type, tiling, usage, flags,
                                                                                pImageFormatProperties);
        const std::vector<VkResult> ignore_list = {VK_ERROR_FORMAT_NOT_SUPPORTED};
        validate_result(my_data->report_data, "vkGetPhysicalDeviceImageFormatProperties", ignore_list, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceProperties(my_data, pProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceProperties(physicalDevice, pProperties);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice,
                                                                  uint32_t *pQueueFamilyPropertyCount,
                                                                  VkQueueFamilyProperties *pQueueFamilyProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceQueueFamilyProperties(my_data, pQueueFamilyPropertyCount,
                                                                          pQueueFamilyProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount,
                                                                       pQueueFamilyProperties);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice,
                                                             VkPhysicalDeviceMemoryProperties *pMemoryProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceMemoryProperties(my_data, pMemoryProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceMemoryProperties(physicalDevice, pMemoryProperties);
    }
}

static bool ValidateDeviceCreateInfo(instance_layer_data *instance_data, VkPhysicalDevice physicalDevice,
                                     const VkDeviceCreateInfo *pCreateInfo) {
    bool skip = false;

    if ((pCreateInfo->enabledLayerCount > 0) && (pCreateInfo->ppEnabledLayerNames != NULL)) {
        for (size_t i = 0; i < pCreateInfo->enabledLayerCount; i++) {
            skip |= validate_string(instance_data->report_data, "vkCreateDevice", "pCreateInfo->ppEnabledLayerNames",
                                    pCreateInfo->ppEnabledLayerNames[i]);
        }
    }

    bool maint1 = false;
    bool negative_viewport = false;

    if ((pCreateInfo->enabledExtensionCount > 0) && (pCreateInfo->ppEnabledExtensionNames != NULL)) {
        for (size_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
            skip |= validate_string(instance_data->report_data, "vkCreateDevice", "pCreateInfo->ppEnabledExtensionNames",
                                    pCreateInfo->ppEnabledExtensionNames[i]);
            if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_MAINTENANCE1_EXTENSION_NAME) == 0) maint1 = true;
            if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_AMD_NEGATIVE_VIEWPORT_HEIGHT_EXTENSION_NAME) == 0)
                negative_viewport = true;
        }
    }

    if (maint1 && negative_viewport) {
        skip |= log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                        __LINE__, VALIDATION_ERROR_056002ec, LayerName,
                        "VkDeviceCreateInfo->ppEnabledExtensionNames must not simultaneously include VK_KHR_maintenance1 and "
                        "VK_AMD_negative_viewport_height. %s",
                        validation_error_map[VALIDATION_ERROR_056002ec]);
    }

    if (pCreateInfo->pNext != NULL && pCreateInfo->pEnabledFeatures) {
        // Check for get_physical_device_properties2 struct
        struct std_header {
            VkStructureType sType;
            const void *pNext;
        };
        std_header *cur_pnext = (std_header *)pCreateInfo->pNext;
        while (cur_pnext) {
            if (VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR == cur_pnext->sType) {
                // Cannot include VkPhysicalDeviceFeatures2KHR and have non-null pEnabledFeatures
                skip |= log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,
                                0, __LINE__, INVALID_USAGE, LayerName,
                                "VkDeviceCreateInfo->pNext includes a VkPhysicalDeviceFeatures2KHR struct when "
                                "pCreateInfo->pEnabledFeatures is non-NULL.");
                break;
            }
            cur_pnext = (std_header *)cur_pnext->pNext;
        }
    }
    if (pCreateInfo->pNext != NULL && pCreateInfo->pEnabledFeatures) {
        // Check for get_physical_device_properties2 struct
        struct std_header {
            VkStructureType sType;
            const void *pNext;
        };
        std_header *cur_pnext = (std_header *)pCreateInfo->pNext;
        while (cur_pnext) {
            if (VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR == cur_pnext->sType) {
                // Cannot include VkPhysicalDeviceFeatures2KHR and have non-null pEnabledFeatures
                skip |= log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,
                                0, __LINE__, INVALID_USAGE, LayerName,
                                "VkDeviceCreateInfo->pNext includes a VkPhysicalDeviceFeatures2KHR struct when "
                                "pCreateInfo->pEnabledFeatures is non-NULL.");
                break;
            }
            cur_pnext = (std_header *)cur_pnext->pNext;
        }
    }

    // Validate pCreateInfo->pQueueCreateInfos
    if (pCreateInfo->pQueueCreateInfos) {
        std::unordered_set<uint32_t> set;

        for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
            const uint32_t requested_queue_family = pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex;
            if (requested_queue_family == VK_QUEUE_FAMILY_IGNORED) {
                skip |= log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT,
                                VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT, HandleToUint64(physicalDevice), __LINE__,
                                VALIDATION_ERROR_06c002fa, LayerName,
                                "vkCreateDevice: pCreateInfo->pQueueCreateInfos[%" PRIu32
                                "].queueFamilyIndex is "
                                "VK_QUEUE_FAMILY_IGNORED, but it is required to provide a valid queue family index value. %s",
                                i, validation_error_map[VALIDATION_ERROR_06c002fa]);
            } else if (set.count(requested_queue_family)) {
                skip |= log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT,
                                VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT, HandleToUint64(physicalDevice), __LINE__,
                                VALIDATION_ERROR_056002e8, LayerName,
                                "vkCreateDevice: pCreateInfo->pQueueCreateInfos[%" PRIu32 "].queueFamilyIndex (=%" PRIu32
                                ") is "
                                "not unique within pCreateInfo->pQueueCreateInfos array. %s",
                                i, requested_queue_family, validation_error_map[VALIDATION_ERROR_056002e8]);
            } else {
                set.insert(requested_queue_family);
            }

            if (pCreateInfo->pQueueCreateInfos[i].pQueuePriorities != nullptr) {
                for (uint32_t j = 0; j < pCreateInfo->pQueueCreateInfos[i].queueCount; ++j) {
                    const float queue_priority = pCreateInfo->pQueueCreateInfos[i].pQueuePriorities[j];
                    if (!(queue_priority >= 0.f) || !(queue_priority <= 1.f)) {
                        skip |= log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT,
                                        VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT, HandleToUint64(physicalDevice), __LINE__,
                                        VALIDATION_ERROR_06c002fe, LayerName,
                                        "vkCreateDevice: pCreateInfo->pQueueCreateInfos[%" PRIu32 "].pQueuePriorities[%" PRIu32
                                        "] (=%f) is not between 0 and 1 (inclusive). %s",
                                        i, j, queue_priority, validation_error_map[VALIDATION_ERROR_06c002fe]);
                    }
                }
            }
        }
    }

    return skip;
}

void storeCreateDeviceData(VkDevice device, const VkDeviceCreateInfo *pCreateInfo) {
    layer_data *my_device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    if ((pCreateInfo != nullptr) && (pCreateInfo->pQueueCreateInfos != nullptr)) {
        for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
            my_device_data->queueFamilyIndexMap.insert(
                std::make_pair(pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex, pCreateInfo->pQueueCreateInfos[i].queueCount));
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    /*
     * NOTE: We do not validate physicalDevice or any dispatchable
     * object as the first parameter. We couldn't get here if it was wrong!
     */

    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_instance_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_instance_data != nullptr);
    std::unique_lock<std::mutex> lock(global_lock);

    skip |= parameter_validation_vkCreateDevice(my_instance_data, pCreateInfo, pAllocator, pDevice);

    if (pCreateInfo != NULL) skip |= ValidateDeviceCreateInfo(my_instance_data, physicalDevice, pCreateInfo);

    if (!skip) {
        VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
        assert(chain_info != nullptr);
        assert(chain_info->u.pLayerInfo != nullptr);

        PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
        PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
        PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(my_instance_data->instance, "vkCreateDevice");
        if (fpCreateDevice == NULL) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        // Advance the link info for the next element on the chain
        chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

        lock.unlock();

        result = fpCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);

        lock.lock();

        validate_result(my_instance_data->report_data, "vkCreateDevice", {}, result);


        if (result == VK_SUCCESS) {
            layer_data *my_device_data = GetLayerDataPtr(get_dispatch_key(*pDevice), layer_data_map);
            assert(my_device_data != nullptr);

            my_device_data->report_data = layer_debug_report_create_device(my_instance_data->report_data, *pDevice);
            layer_init_device_dispatch_table(*pDevice, &my_device_data->dispatch_table, fpGetDeviceProcAddr);

            my_device_data->extensions.InitFromDeviceCreateInfo(&my_instance_data->extensions, pCreateInfo);

            storeCreateDeviceData(*pDevice, pCreateInfo);

            // Query and save physical device limits for this device
            VkPhysicalDeviceProperties device_properties = {};
            my_instance_data->dispatch_table.GetPhysicalDeviceProperties(physicalDevice, &device_properties);
            memcpy(&my_device_data->device_limits, &device_properties.limits, sizeof(VkPhysicalDeviceLimits));
            my_device_data->physical_device = physicalDevice;
            my_device_data->device = *pDevice;

            // Save app-enabled features in this device's layer_data structure
            if (pCreateInfo->pEnabledFeatures) {
                my_device_data->physical_device_features = *pCreateInfo->pEnabledFeatures;
            } else {
                memset(&my_device_data->physical_device_features, 0, sizeof(VkPhysicalDeviceFeatures));
            }
        }
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    dispatch_key key = get_dispatch_key(device);
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyDevice(my_data, pAllocator);

    if (!skip) {
        layer_debug_report_destroy_device(device);

#if DISPATCH_MAP_DEBUG
        fprintf(stderr, "Device:  0x%p, key:  0x%p\n", device, key);
#endif

        my_data->dispatch_table.DestroyDevice(device, pAllocator);
    }

    FreeLayerDataPtr(key, layer_data_map);
}

static bool PreGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex) {
    bool skip = false;
    layer_data *my_device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_device_data != nullptr);

    skip |= ValidateDeviceQueueFamily(my_device_data, queueFamilyIndex, "vkGetDeviceQueue", "queueFamilyIndex",
                                      VALIDATION_ERROR_29600300);

    const auto &queue_data = my_device_data->queueFamilyIndexMap.find(queueFamilyIndex);
    if (queue_data != my_device_data->queueFamilyIndexMap.end() && queue_data->second <= queueIndex) {
        skip |= log_msg(my_device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
                        HandleToUint64(device), __LINE__, VALIDATION_ERROR_29600302, LayerName,
                        "vkGetDeviceQueue: queueIndex (=%" PRIu32
                        ") is not less than the number of queues requested from "
                        "queueFamilyIndex (=%" PRIu32 ") when the device was created (i.e. is not less than %" PRIu32 "). %s",
                        queueIndex, queueFamilyIndex, queue_data->second, validation_error_map[VALIDATION_ERROR_29600302]);
    }

    return skip;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue *pQueue) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);
    std::unique_lock<std::mutex> lock(global_lock);

    skip |= parameter_validation_vkGetDeviceQueue(my_data, queueFamilyIndex, queueIndex, pQueue);

    if (!skip) {
        PreGetDeviceQueue(device, queueFamilyIndex, queueIndex);

        lock.unlock();

        my_data->dispatch_table.GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkQueueSubmit(my_data, submitCount, pSubmits, fence);

    if (!skip) {
        result = my_data->dispatch_table.QueueSubmit(queue, submitCount, pSubmits, fence);

        validate_result(my_data->report_data, "vkQueueSubmit", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queue) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    assert(my_data != NULL);

    VkResult result = my_data->dispatch_table.QueueWaitIdle(queue);

    validate_result(my_data->report_data, "vkQueueWaitIdle", {}, result);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    VkResult result = my_data->dispatch_table.DeviceWaitIdle(device);

    validate_result(my_data->report_data, "vkDeviceWaitIdle", {}, result);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                                              const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkAllocateMemory(my_data, pAllocateInfo, pAllocator, pMemory);

    if (!skip) {
        result = my_data->dispatch_table.AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);

        validate_result(my_data->report_data, "vkAllocateMemory", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkFreeMemory(my_data, memory, pAllocator);

    if (!skip) {
        my_data->dispatch_table.FreeMemory(device, memory, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                                         VkMemoryMapFlags flags, void **ppData) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkMapMemory(my_data, memory, offset, size, flags, ppData);

    if (!skip) {
        result = my_data->dispatch_table.MapMemory(device, memory, offset, size, flags, ppData);

        validate_result(my_data->report_data, "vkMapMemory", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkUnmapMemory(my_data, memory);

    if (!skip) {
        my_data->dispatch_table.UnmapMemory(device, memory);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                                       const VkMappedMemoryRange *pMemoryRanges) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkFlushMappedMemoryRanges(my_data, memoryRangeCount, pMemoryRanges);

    if (!skip) {
        result = my_data->dispatch_table.FlushMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);

        validate_result(my_data->report_data, "vkFlushMappedMemoryRanges", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                                            const VkMappedMemoryRange *pMemoryRanges) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkInvalidateMappedMemoryRanges(my_data, memoryRangeCount, pMemoryRanges);

    if (!skip) {
        result = my_data->dispatch_table.InvalidateMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);

        validate_result(my_data->report_data, "vkInvalidateMappedMemoryRanges", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory memory,
                                                     VkDeviceSize *pCommittedMemoryInBytes) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDeviceMemoryCommitment(my_data, memory, pCommittedMemoryInBytes);

    if (!skip) {
        my_data->dispatch_table.GetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                                VkDeviceSize memoryOffset) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkBindBufferMemory(my_data, buffer, memory, memoryOffset);

    if (!skip) {
        result = my_data->dispatch_table.BindBufferMemory(device, buffer, memory, memoryOffset);

        validate_result(my_data->report_data, "vkBindBufferMemory", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkBindImageMemory(my_data, image, memory, memoryOffset);

    if (!skip) {
        result = my_data->dispatch_table.BindImageMemory(device, image, memory, memoryOffset);

        validate_result(my_data->report_data, "vkBindImageMemory", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(VkDevice device, VkBuffer buffer,
                                                       VkMemoryRequirements *pMemoryRequirements) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetBufferMemoryRequirements(my_data, buffer, pMemoryRequirements);

    if (!skip) {
        my_data->dispatch_table.GetBufferMemoryRequirements(device, buffer, pMemoryRequirements);
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements *pMemoryRequirements) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetImageMemoryRequirements(my_data, image, pMemoryRequirements);

    if (!skip) {
        my_data->dispatch_table.GetImageMemoryRequirements(device, image, pMemoryRequirements);
    }
}

static bool PostGetImageSparseMemoryRequirements(VkDevice device, VkImage image, uint32_t *pNumRequirements,
                                                 VkSparseImageMemoryRequirements *pSparseMemoryRequirements) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    if (pSparseMemoryRequirements != nullptr) {
        if ((pSparseMemoryRequirements->formatProperties.aspectMask &
             (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT |
              VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    UNRECOGNIZED_VALUE, LayerName,
                    "vkGetImageSparseMemoryRequirements parameter, VkImageAspect "
                    "pSparseMemoryRequirements->formatProperties.aspectMask, is an unrecognized enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements(VkDevice device, VkImage image, uint32_t *pSparseMemoryRequirementCount,
                                                            VkSparseImageMemoryRequirements *pSparseMemoryRequirements) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetImageSparseMemoryRequirements(my_data, image, pSparseMemoryRequirementCount,
                                                                    pSparseMemoryRequirements);

    if (!skip) {
        my_data->dispatch_table.GetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount,
                                                                 pSparseMemoryRequirements);

        PostGetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
    }
}

static bool PostGetPhysicalDeviceSparseImageFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type,
                                                             VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                                                             VkImageTiling tiling, uint32_t *pNumProperties,
                                                             VkSparseImageFormatProperties *pProperties) {
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    if (pProperties != nullptr) {
        if ((pProperties->aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT |
                                        VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                    LayerName,
                    "vkGetPhysicalDeviceSparseImageFormatProperties parameter, VkImageAspect pProperties->aspectMask, is an "
                    "unrecognized enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceSparseImageFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format,
                                                                        VkImageType type, VkSampleCountFlagBits samples,
                                                                        VkImageUsageFlags usage, VkImageTiling tiling,
                                                                        uint32_t *pPropertyCount,
                                                                        VkSparseImageFormatProperties *pProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceSparseImageFormatProperties(my_data, format, type, samples, usage,
                                                                                tiling, pPropertyCount, pProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format, type, samples, usage, tiling,
                                                                             pPropertyCount, pProperties);

        PostGetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format, type, samples, usage, tiling, pPropertyCount,
                                                         pProperties);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL QueueBindSparse(VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo *pBindInfo,
                                               VkFence fence) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkQueueBindSparse(my_data, bindInfoCount, pBindInfo, fence);

    if (!skip) {
        result = my_data->dispatch_table.QueueBindSparse(queue, bindInfoCount, pBindInfo, fence);

        validate_result(my_data->report_data, "vkQueueBindSparse", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice device, const VkFenceCreateInfo *pCreateInfo,
                                           const VkAllocationCallbacks *pAllocator, VkFence *pFence) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateFence(my_data, pCreateInfo, pAllocator, pFence);

    if (!skip) {
        result = my_data->dispatch_table.CreateFence(device, pCreateInfo, pAllocator, pFence);

        validate_result(my_data->report_data, "vkCreateFence", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyFence(my_data, fence, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyFence(device, fence, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkResetFences(my_data, fenceCount, pFences);

    if (!skip) {
        result = my_data->dispatch_table.ResetFences(device, fenceCount, pFences);

        validate_result(my_data->report_data, "vkResetFences", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(VkDevice device, VkFence fence) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetFenceStatus(my_data, fence);

    if (!skip) {
        result = my_data->dispatch_table.GetFenceStatus(device, fence);

        validate_result(my_data->report_data, "vkGetFenceStatus", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences, VkBool32 waitAll,
                                             uint64_t timeout) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkWaitForFences(my_data, fenceCount, pFences, waitAll, timeout);

    if (!skip) {
        result = my_data->dispatch_table.WaitForFences(device, fenceCount, pFences, waitAll, timeout);

        validate_result(my_data->report_data, "vkWaitForFences", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo,
                                               const VkAllocationCallbacks *pAllocator, VkSemaphore *pSemaphore) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateSemaphore(my_data, pCreateInfo, pAllocator, pSemaphore);

    if (!skip) {
        result = my_data->dispatch_table.CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);

        validate_result(my_data->report_data, "vkCreateSemaphore", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySemaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroySemaphore(my_data, semaphore, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroySemaphore(device, semaphore, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateEvent(VkDevice device, const VkEventCreateInfo *pCreateInfo,
                                           const VkAllocationCallbacks *pAllocator, VkEvent *pEvent) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateEvent(my_data, pCreateInfo, pAllocator, pEvent);

    if (!skip) {
        result = my_data->dispatch_table.CreateEvent(device, pCreateInfo, pAllocator, pEvent);

        validate_result(my_data->report_data, "vkCreateEvent", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyEvent(VkDevice device, VkEvent event, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyEvent(my_data, event, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyEvent(device, event, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetEventStatus(VkDevice device, VkEvent event) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetEventStatus(my_data, event);

    if (!skip) {
        result = my_data->dispatch_table.GetEventStatus(device, event);

        validate_result(my_data->report_data, "vkGetEventStatus", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL SetEvent(VkDevice device, VkEvent event) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkSetEvent(my_data, event);

    if (!skip) {
        result = my_data->dispatch_table.SetEvent(device, event);

        validate_result(my_data->report_data, "vkSetEvent", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetEvent(VkDevice device, VkEvent event) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkResetEvent(my_data, event);

    if (!skip) {
        result = my_data->dispatch_table.ResetEvent(device, event);

        validate_result(my_data->report_data, "vkResetEvent", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                                               const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateQueryPool(device_data, pCreateInfo, pAllocator, pQueryPool);

    // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
    if (pCreateInfo != nullptr) {
        // If queryType is VK_QUERY_TYPE_PIPELINE_STATISTICS, pipelineStatistics must be a valid combination of
        // VkQueryPipelineStatisticFlagBits values
        if ((pCreateInfo->queryType == VK_QUERY_TYPE_PIPELINE_STATISTICS) && (pCreateInfo->pipelineStatistics != 0) &&
            ((pCreateInfo->pipelineStatistics & (~AllVkQueryPipelineStatisticFlagBits)) != 0)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_11c00630, LayerName,
                            "vkCreateQueryPool(): if pCreateInfo->queryType is "
                            "VK_QUERY_TYPE_PIPELINE_STATISTICS, pCreateInfo->pipelineStatistics must be "
                            "a valid combination of VkQueryPipelineStatisticFlagBits values. %s",
                            validation_error_map[VALIDATION_ERROR_11c00630]);
        }
    }

    if (!skip) {
        result = device_data->dispatch_table.CreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);

        validate_result(report_data, "vkCreateQueryPool", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyQueryPool(my_data, queryPool, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyQueryPool(device, queryPool, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount,
                                                   size_t dataSize, void *pData, VkDeviceSize stride, VkQueryResultFlags flags) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetQueryPoolResults(my_data, queryPool, firstQuery, queryCount, dataSize, pData,
                                                       stride, flags);

    if (!skip) {
        result =
            my_data->dispatch_table.GetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);

        validate_result(my_data->report_data, "vkGetQueryPoolResults", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);

    std::unique_lock<std::mutex> lock(global_lock);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateBuffer(device_data, pCreateInfo, pAllocator, pBuffer);

    if (pCreateInfo != nullptr) {
        // Buffer size must be greater than 0 (error 00663)
        skip |=
            ValidateGreaterThan(report_data, "vkCreateBuffer", "pCreateInfo->size", static_cast<uint32_t>(pCreateInfo->size), 0u);

        // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
        if (pCreateInfo->sharingMode == VK_SHARING_MODE_CONCURRENT) {
            // If sharingMode is VK_SHARING_MODE_CONCURRENT, queueFamilyIndexCount must be greater than 1
            if (pCreateInfo->queueFamilyIndexCount <= 1) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_01400724, LayerName,
                                "vkCreateBuffer: if pCreateInfo->sharingMode is VK_SHARING_MODE_CONCURRENT, "
                                "pCreateInfo->queueFamilyIndexCount must be greater than 1. %s",
                                validation_error_map[VALIDATION_ERROR_01400724]);
            }

            // If sharingMode is VK_SHARING_MODE_CONCURRENT, pQueueFamilyIndices must be a pointer to an array of
            // queueFamilyIndexCount uint32_t values
            if (pCreateInfo->pQueueFamilyIndices == nullptr) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_01400722, LayerName,
                                "vkCreateBuffer: if pCreateInfo->sharingMode is VK_SHARING_MODE_CONCURRENT, "
                                "pCreateInfo->pQueueFamilyIndices must be a pointer to an array of "
                                "pCreateInfo->queueFamilyIndexCount uint32_t values. %s",
                                validation_error_map[VALIDATION_ERROR_01400722]);
            } else {
                // TODO: Not in the spec VUs. Probably missing -- KhronosGroup/Vulkan-Docs#501. Update error codes when resolved.
                skip |= ValidateQueueFamilies(device_data, pCreateInfo->queueFamilyIndexCount, pCreateInfo->pQueueFamilyIndices,
                                              "vkCreateBuffer", "pCreateInfo->pQueueFamilyIndices", INVALID_USAGE, INVALID_USAGE,
                                              false, "", "");
            }
        }

        // If flags contains VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT or VK_BUFFER_CREATE_SPARSE_ALIASED_BIT, it must also contain
        // VK_BUFFER_CREATE_SPARSE_BINDING_BIT
        if (((pCreateInfo->flags & (VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT | VK_BUFFER_CREATE_SPARSE_ALIASED_BIT)) != 0) &&
            ((pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != VK_BUFFER_CREATE_SPARSE_BINDING_BIT)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_0140072c, LayerName,
                            "vkCreateBuffer: if pCreateInfo->flags contains VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT or "
                            "VK_BUFFER_CREATE_SPARSE_ALIASED_BIT, it must also contain VK_BUFFER_CREATE_SPARSE_BINDING_BIT. %s",
                            validation_error_map[VALIDATION_ERROR_0140072c]);
        }
    }

    lock.unlock();

    if (!skip) {
        result = device_data->dispatch_table.CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);

        validate_result(report_data, "vkCreateBuffer", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyBuffer(my_data, buffer, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyBuffer(device, buffer, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBufferView(VkDevice device, const VkBufferViewCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator, VkBufferView *pView) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateBufferView(my_data, pCreateInfo, pAllocator, pView);

    if (!skip) {
        result = my_data->dispatch_table.CreateBufferView(device, pCreateInfo, pAllocator, pView);

        validate_result(my_data->report_data, "vkCreateBufferView", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBufferView(VkDevice device, VkBufferView bufferView, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyBufferView(my_data, bufferView, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyBufferView(device, bufferView, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice device, const VkImageCreateInfo *pCreateInfo,
                                           const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);

    std::unique_lock<std::mutex> lock(global_lock);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateImage(device_data, pCreateInfo, pAllocator, pImage);

    if (pCreateInfo != nullptr) {

        if ((device_data->physical_device_features.textureCompressionETC2 == false) &&
            FormatIsCompressed_ETC2_EAC(pCreateInfo->format)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCreateImage(): Attempting to create VkImage with format %s. The textureCompressionETC2 feature is "
                            "not enabled: neither ETC2 nor EAC formats can be used to create images.",
                            string_VkFormat(pCreateInfo->format));
        }

        if ((device_data->physical_device_features.textureCompressionASTC_LDR == false) &&
            FormatIsCompressed_ASTC_LDR(pCreateInfo->format)) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        DEVICE_FEATURE, LayerName,
                        "vkCreateImage(): Attempting to create VkImage with format %s. The textureCompressionASTC_LDR feature is "
                        "not enabled: ASTC formats cannot be used to create images.",
                        string_VkFormat(pCreateInfo->format));
        }

        if ((device_data->physical_device_features.textureCompressionBC == false) &&
            FormatIsCompressed_BC(pCreateInfo->format)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCreateImage(): Attempting to create VkImage with format %s. The textureCompressionBC feature is "
                            "not enabled: BC compressed formats cannot be used to create images.",
                            string_VkFormat(pCreateInfo->format));
        }

        // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
        if (pCreateInfo->sharingMode == VK_SHARING_MODE_CONCURRENT) {
            // If sharingMode is VK_SHARING_MODE_CONCURRENT, queueFamilyIndexCount must be greater than 1
            if (pCreateInfo->queueFamilyIndexCount <= 1) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_09e0075c, LayerName,
                                "vkCreateImage(): if pCreateInfo->sharingMode is VK_SHARING_MODE_CONCURRENT, "
                                "pCreateInfo->queueFamilyIndexCount must be greater than 1. %s",
                                validation_error_map[VALIDATION_ERROR_09e0075c]);
            }

            // If sharingMode is VK_SHARING_MODE_CONCURRENT, pQueueFamilyIndices must be a pointer to an array of
            // queueFamilyIndexCount uint32_t values
            if (pCreateInfo->pQueueFamilyIndices == nullptr) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_09e0075a, LayerName,
                                "vkCreateImage(): if pCreateInfo->sharingMode is VK_SHARING_MODE_CONCURRENT, "
                                "pCreateInfo->pQueueFamilyIndices must be a pointer to an array of "
                                "pCreateInfo->queueFamilyIndexCount uint32_t values. %s",
                                validation_error_map[VALIDATION_ERROR_09e0075a]);
            } else {
                // TODO: Not in the spec VUs. Probably missing -- KhronosGroup/Vulkan-Docs#501. Update error codes when resolved.
                skip |= ValidateQueueFamilies(device_data, pCreateInfo->queueFamilyIndexCount, pCreateInfo->pQueueFamilyIndices,
                                              "vkCreateImage", "pCreateInfo->pQueueFamilyIndices", INVALID_USAGE, INVALID_USAGE,
                                              false, "", "");
            }
        }

        // width, height, and depth members of extent must be greater than 0
        skip |= ValidateGreaterThan(report_data, "vkCreateImage", "pCreateInfo->extent.width", pCreateInfo->extent.width, 0u);
        skip |= ValidateGreaterThan(report_data, "vkCreateImage", "pCreateInfo->extent.height", pCreateInfo->extent.height, 0u);
        skip |= ValidateGreaterThan(report_data, "vkCreateImage", "pCreateInfo->extent.depth", pCreateInfo->extent.depth, 0u);

        // mipLevels must be greater than 0
        skip |= ValidateGreaterThan(report_data, "vkCreateImage", "pCreateInfo->mipLevels", pCreateInfo->mipLevels, 0u);

        // arrayLayers must be greater than 0
        skip |= ValidateGreaterThan(report_data, "vkCreateImage", "pCreateInfo->arrayLayers", pCreateInfo->arrayLayers, 0u);

        // If imageType is VK_IMAGE_TYPE_1D, both extent.height and extent.depth must be 1
        if ((pCreateInfo->imageType == VK_IMAGE_TYPE_1D) && (pCreateInfo->extent.height != 1) && (pCreateInfo->extent.depth != 1)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_09e00778, LayerName,
                            "vkCreateImage(): if pCreateInfo->imageType is VK_IMAGE_TYPE_1D, both "
                            "pCreateInfo->extent.height and pCreateInfo->extent.depth must be 1. %s",
                            validation_error_map[VALIDATION_ERROR_09e00778]);
        }

        if (pCreateInfo->imageType == VK_IMAGE_TYPE_2D) {
            // If imageType is VK_IMAGE_TYPE_2D and flags contains VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, extent.width and
            // extent.height must be equal
            if ((pCreateInfo->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
                (pCreateInfo->extent.width != pCreateInfo->extent.height)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_09e00774, LayerName,
                                "vkCreateImage(): if pCreateInfo->imageType is VK_IMAGE_TYPE_2D and "
                                "pCreateInfo->flags contains VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, "
                                "pCreateInfo->extent.width and pCreateInfo->extent.height must be equal. %s",
                                validation_error_map[VALIDATION_ERROR_09e00774]);
            }

            if (pCreateInfo->extent.depth != 1) {
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_09e0077a, LayerName,
                    "vkCreateImage(): if pCreateInfo->imageType is VK_IMAGE_TYPE_2D, pCreateInfo->extent.depth must be 1. %s",
                    validation_error_map[VALIDATION_ERROR_09e0077a]);
            }
        }

        // mipLevels must be less than or equal to floor(log2(max(extent.width,extent.height,extent.depth)))+1
        uint32_t maxDim = std::max(std::max(pCreateInfo->extent.width, pCreateInfo->extent.height), pCreateInfo->extent.depth);
        if (pCreateInfo->mipLevels > (floor(log2(maxDim)) + 1)) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0077c, LayerName,
                        "vkCreateImage(): pCreateInfo->mipLevels must be less than or equal to "
                        "floor(log2(max(pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->extent.depth)))+1. %s",
                        validation_error_map[VALIDATION_ERROR_09e0077c]);
        }

        // If flags contains VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT or VK_IMAGE_CREATE_SPARSE_ALIASED_BIT, it must also contain
        // VK_IMAGE_CREATE_SPARSE_BINDING_BIT
        if (((pCreateInfo->flags & (VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT)) != 0) &&
            ((pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) != VK_IMAGE_CREATE_SPARSE_BINDING_BIT)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_09e007b6, LayerName,
                            "vkCreateImage: if pCreateInfo->flags contains VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT or "
                            "VK_IMAGE_CREATE_SPARSE_ALIASED_BIT, it must also contain VK_IMAGE_CREATE_SPARSE_BINDING_BIT. %s",
                            validation_error_map[VALIDATION_ERROR_09e007b6]);
        }

        // Check for combinations of attributes that are incompatible with having VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT set
        if ((pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) != 0) {
            // Linear tiling is unsupported
            if (VK_IMAGE_TILING_LINEAR == pCreateInfo->tiling) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                INVALID_USAGE, LayerName,
                                "vkCreateImage: if pCreateInfo->flags contains VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT "
                                "then image tiling of VK_IMAGE_TILING_LINEAR is not supported");
            }

            // Sparse 1D image isn't valid
            if (VK_IMAGE_TYPE_1D == pCreateInfo->imageType) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_09e00794, LayerName,
                                "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 1D image. %s",
                                validation_error_map[VALIDATION_ERROR_09e00794]);
            }

            // Sparse 2D image when device doesn't support it
            if ((VK_FALSE == device_data->physical_device_features.sparseResidencyImage2D) &&
                (VK_IMAGE_TYPE_2D == pCreateInfo->imageType)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_09e00796, LayerName,
                                "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 2D image if corresponding "
                                "feature is not enabled on the device. %s",
                                validation_error_map[VALIDATION_ERROR_09e00796]);
            }

            // Sparse 3D image when device doesn't support it
            if ((VK_FALSE == device_data->physical_device_features.sparseResidencyImage3D) &&
                (VK_IMAGE_TYPE_3D == pCreateInfo->imageType)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_09e00798, LayerName,
                                "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 3D image if corresponding "
                                "feature is not enabled on the device. %s",
                                validation_error_map[VALIDATION_ERROR_09e00798]);
            }

            // Multi-sample 2D image when device doesn't support it
            if (VK_IMAGE_TYPE_2D == pCreateInfo->imageType) {
                if ((VK_FALSE == device_data->physical_device_features.sparseResidency2Samples) &&
                    (VK_SAMPLE_COUNT_2_BIT == pCreateInfo->samples)) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0079a, LayerName,
                        "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 2-sample image if corresponding "
                        "feature is not enabled on the device. %s",
                        validation_error_map[VALIDATION_ERROR_09e0079a]);
                } else if ((VK_FALSE == device_data->physical_device_features.sparseResidency4Samples) &&
                           (VK_SAMPLE_COUNT_4_BIT == pCreateInfo->samples)) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0079c, LayerName,
                        "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 4-sample image if corresponding "
                        "feature is not enabled on the device. %s",
                        validation_error_map[VALIDATION_ERROR_09e0079c]);
                } else if ((VK_FALSE == device_data->physical_device_features.sparseResidency8Samples) &&
                           (VK_SAMPLE_COUNT_8_BIT == pCreateInfo->samples)) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0079e, LayerName,
                        "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 8-sample image if corresponding "
                        "feature is not enabled on the device. %s",
                        validation_error_map[VALIDATION_ERROR_09e0079e]);
                } else if ((VK_FALSE == device_data->physical_device_features.sparseResidency16Samples) &&
                           (VK_SAMPLE_COUNT_16_BIT == pCreateInfo->samples)) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e007a0, LayerName,
                        "vkCreateImage: cannot specify VK_IMAGE_CREATE_SPARSE_BINDING_BIT for 16-sample image if corresponding "
                        "feature is not enabled on the device. %s",
                        validation_error_map[VALIDATION_ERROR_09e007a0]);
                }
            }
        }
    }

    lock.unlock();

    if (!skip) {
        result = device_data->dispatch_table.CreateImage(device, pCreateInfo, pAllocator, pImage);

        validate_result(report_data, "vkCreateImage", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyImage(my_data, image, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyImage(device, image, pAllocator);
    }
}

static bool PreGetImageSubresourceLayout(VkDevice device, const VkImageSubresource *pSubresource) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    if (pSubresource != nullptr) {
        if ((pSubresource->aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT |
                                         VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    UNRECOGNIZED_VALUE, LayerName,
                    "vkGetImageSubresourceLayout parameter, VkImageAspect pSubresource->aspectMask, is an unrecognized enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL GetImageSubresourceLayout(VkDevice device, VkImage image, const VkImageSubresource *pSubresource,
                                                     VkSubresourceLayout *pLayout) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetImageSubresourceLayout(my_data, image, pSubresource, pLayout);

    if (!skip) {
        PreGetImageSubresourceLayout(device, pSubresource);

        my_data->dispatch_table.GetImageSubresourceLayout(device, image, pSubresource, pLayout);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
                                               const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);
    debug_report_data *report_data = my_data->report_data;

    skip |= parameter_validation_vkCreateImageView(my_data, pCreateInfo, pAllocator, pView);

    if (pCreateInfo != nullptr) {
        if ((pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_1D) || (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_2D)) {
            if ((pCreateInfo->subresourceRange.layerCount != 1) &&
                (pCreateInfo->subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName,
                                "vkCreateImageView: if pCreateInfo->viewType is VK_IMAGE_TYPE_%dD, "
                                "pCreateInfo->subresourceRange.layerCount must be 1",
                                ((pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_1D) ? 1 : 2));
            }
        } else if ((pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_1D_ARRAY) ||
                   (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY)) {
            if ((pCreateInfo->subresourceRange.layerCount < 1) &&
                (pCreateInfo->subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName,
                                "vkCreateImageView: if pCreateInfo->viewType is VK_IMAGE_TYPE_%dD_ARRAY, "
                                "pCreateInfo->subresourceRange.layerCount must be >= 1",
                                ((pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_1D_ARRAY) ? 1 : 2));
            }
        } else if (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE) {
            if ((pCreateInfo->subresourceRange.layerCount != 6) &&
                (pCreateInfo->subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName,
                                "vkCreateImageView: if pCreateInfo->viewType is VK_IMAGE_TYPE_CUBE, "
                                "pCreateInfo->subresourceRange.layerCount must be 6");
            }
        } else if (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
            if (((pCreateInfo->subresourceRange.layerCount == 0) || ((pCreateInfo->subresourceRange.layerCount % 6) != 0)) &&
                (pCreateInfo->subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName,
                                "vkCreateImageView: if pCreateInfo->viewType is VK_IMAGE_TYPE_CUBE_ARRAY, "
                                "pCreateInfo->subresourceRange.layerCount must be a multiple of 6");
            }
            if (!my_data->physical_device_features.imageCubeArray) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName, "vkCreateImageView: Device feature imageCubeArray not enabled.");
            }
        } else if (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_3D) {
            if (pCreateInfo->subresourceRange.baseArrayLayer != 0) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName,
                                "vkCreateImageView: if pCreateInfo->viewType is VK_IMAGE_TYPE_3D, "
                                "pCreateInfo->subresourceRange.baseArrayLayer must be 0");
            }

            if ((pCreateInfo->subresourceRange.layerCount != 1) &&
                (pCreateInfo->subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 1,
                                LayerName,
                                "vkCreateImageView: if pCreateInfo->viewType is VK_IMAGE_TYPE_3D, "
                                "pCreateInfo->subresourceRange.layerCount must be 1");
            }
        }
    }

    if (!skip) {
        result = my_data->dispatch_table.CreateImageView(device, pCreateInfo, pAllocator, pView);

        validate_result(my_data->report_data, "vkCreateImageView", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyImageView(my_data, imageView, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyImageView(device, imageView, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                  const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateShaderModule(my_data, pCreateInfo, pAllocator, pShaderModule);

    if (!skip) {
        result = my_data->dispatch_table.CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);

        validate_result(my_data->report_data, "vkCreateShaderModule", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                               const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyShaderModule(my_data, shaderModule, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyShaderModule(device, shaderModule, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(VkDevice device, const VkPipelineCacheCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkPipelineCache *pPipelineCache) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreatePipelineCache(my_data, pCreateInfo, pAllocator, pPipelineCache);

    if (!skip) {
        result = my_data->dispatch_table.CreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache);

        validate_result(my_data->report_data, "vkCreatePipelineCache", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(VkDevice device, VkPipelineCache pipelineCache,
                                                const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyPipelineCache(my_data, pipelineCache, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyPipelineCache(device, pipelineCache, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(VkDevice device, VkPipelineCache pipelineCache, size_t *pDataSize,
                                                    void *pData) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPipelineCacheData(my_data, pipelineCache, pDataSize, pData);

    if (!skip) {
        result = my_data->dispatch_table.GetPipelineCacheData(device, pipelineCache, pDataSize, pData);

        validate_result(my_data->report_data, "vkGetPipelineCacheData", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount,
                                                   const VkPipelineCache *pSrcCaches) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkMergePipelineCaches(my_data, dstCache, srcCacheCount, pSrcCaches);

    if (!skip) {
        result = my_data->dispatch_table.MergePipelineCaches(device, dstCache, srcCacheCount, pSrcCaches);

        validate_result(my_data->report_data, "vkMergePipelineCaches", {}, result);
    }

    return result;
}

static bool PreCreateGraphicsPipelines(VkDevice device, const VkGraphicsPipelineCreateInfo *pCreateInfos) {
    layer_data *data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    bool skip = false;

    // TODO: Handle count
    if (pCreateInfos != nullptr) {
        if (pCreateInfos->flags & VK_PIPELINE_CREATE_DERIVATIVE_BIT) {
            if (pCreateInfos->basePipelineIndex != -1) {
                if (pCreateInfos->basePipelineHandle != VK_NULL_HANDLE) {
                    skip |= log_msg(
                        data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_096005a8, LayerName,
                        "vkCreateGraphicsPipelines parameter, pCreateInfos->basePipelineHandle, must be VK_NULL_HANDLE if "
                        "pCreateInfos->flags "
                        "contains the VK_PIPELINE_CREATE_DERIVATIVE_BIT flag and pCreateInfos->basePipelineIndex is not -1. %s",
                        validation_error_map[VALIDATION_ERROR_096005a8]);
                }
            }

            if (pCreateInfos->basePipelineHandle != VK_NULL_HANDLE) {
                if (pCreateInfos->basePipelineIndex != -1) {
                    skip |= log_msg(
                        data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_096005aa, LayerName,
                        "vkCreateGraphicsPipelines parameter, pCreateInfos->basePipelineIndex, must be -1 if pCreateInfos->flags "
                        "contains the VK_PIPELINE_CREATE_DERIVATIVE_BIT flag and pCreateInfos->basePipelineHandle is not "
                        "VK_NULL_HANDLE. %s",
                        validation_error_map[VALIDATION_ERROR_096005aa]);
                }
            }
        }

        if (pCreateInfos->pRasterizationState != nullptr) {
            if (pCreateInfos->pRasterizationState->cullMode & ~VK_CULL_MODE_FRONT_AND_BACK) {
                skip |=
                    log_msg(data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            UNRECOGNIZED_VALUE, LayerName,
                            "vkCreateGraphicsPipelines parameter, VkCullMode pCreateInfos->pRasterizationState->cullMode, is an "
                            "unrecognized enumerator");
            }

            if ((pCreateInfos->pRasterizationState->polygonMode != VK_POLYGON_MODE_FILL) &&
                (data->physical_device_features.fillModeNonSolid == false)) {
                skip |= log_msg(
                    data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    DEVICE_FEATURE, LayerName,
                    "vkCreateGraphicsPipelines parameter, VkPolygonMode pCreateInfos->pRasterizationState->polygonMode cannot be "
                    "VK_POLYGON_MODE_POINT or VK_POLYGON_MODE_LINE if VkPhysicalDeviceFeatures->fillModeNonSolid is false.");
            }
        }

        size_t i = 0;
        for (size_t j = 0; j < pCreateInfos[i].stageCount; j++) {
            skip |= validate_string(data->report_data, "vkCreateGraphicsPipelines",
                                    ParameterName("pCreateInfos[%i].pStages[%i].pName", ParameterName::IndexVector{i, j}),
                                    pCreateInfos[i].pStages[j].pName);
        }
    }

    return skip;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                       const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                       const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateGraphicsPipelines(device_data, pipelineCache, createInfoCount, pCreateInfos, pAllocator,
                                                           pPipelines);

    if (pCreateInfos != nullptr) {
        for (uint32_t i = 0; i < createInfoCount; ++i) {
            // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
            if (pCreateInfos[i].pVertexInputState != nullptr) {
                auto const &vertex_input_state = pCreateInfos[i].pVertexInputState;
                for (uint32_t d = 0; d < vertex_input_state->vertexBindingDescriptionCount; ++d) {
                    auto const &vertex_bind_desc = vertex_input_state->pVertexBindingDescriptions[d];
                    if (vertex_bind_desc.binding >= device_data->device_limits.maxVertexInputBindings) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_14c004d4, LayerName,
                                        "vkCreateGraphicsPipelines: parameter "
                                        "pCreateInfos[%u].pVertexInputState->pVertexBindingDescriptions[%u].binding (%u) is "
                                        "greater than or equal to VkPhysicalDeviceLimits::maxVertexInputBindings (%u). %s",
                                        i, d, vertex_bind_desc.binding, device_data->device_limits.maxVertexInputBindings,
                                        validation_error_map[VALIDATION_ERROR_14c004d4]);
                    }

                    if (vertex_bind_desc.stride > device_data->device_limits.maxVertexInputBindingStride) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_14c004d6, LayerName,
                                        "vkCreateGraphicsPipelines: parameter "
                                        "pCreateInfos[%u].pVertexInputState->pVertexBindingDescriptions[%u].stride (%u) is greater "
                                        "than VkPhysicalDeviceLimits::maxVertexInputBindingStride (%u). %s",
                                        i, d, vertex_bind_desc.stride, device_data->device_limits.maxVertexInputBindingStride,
                                        validation_error_map[VALIDATION_ERROR_14c004d6]);
                    }
                }

                for (uint32_t d = 0; d < vertex_input_state->vertexAttributeDescriptionCount; ++d) {
                    auto const &vertex_attrib_desc = vertex_input_state->pVertexAttributeDescriptions[d];
                    if (vertex_attrib_desc.location >= device_data->device_limits.maxVertexInputAttributes) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_14a004d8, LayerName,
                                        "vkCreateGraphicsPipelines: parameter "
                                        "pCreateInfos[%u].pVertexInputState->pVertexAttributeDescriptions[%u].location (%u) is "
                                        "greater than or equal to VkPhysicalDeviceLimits::maxVertexInputAttributes (%u). %s",
                                        i, d, vertex_attrib_desc.location, device_data->device_limits.maxVertexInputAttributes,
                                        validation_error_map[VALIDATION_ERROR_14a004d8]);
                    }

                    if (vertex_attrib_desc.binding >= device_data->device_limits.maxVertexInputBindings) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_14a004da, LayerName,
                                        "vkCreateGraphicsPipelines: parameter "
                                        "pCreateInfos[%u].pVertexInputState->pVertexAttributeDescriptions[%u].binding (%u) is "
                                        "greater than or equal to VkPhysicalDeviceLimits::maxVertexInputBindings (%u). %s",
                                        i, d, vertex_attrib_desc.binding, device_data->device_limits.maxVertexInputBindings,
                                        validation_error_map[VALIDATION_ERROR_14a004da]);
                    }

                    if (vertex_attrib_desc.offset > device_data->device_limits.maxVertexInputAttributeOffset) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_14a004dc, LayerName,
                                        "vkCreateGraphicsPipelines: parameter "
                                        "pCreateInfos[%u].pVertexInputState->pVertexAttributeDescriptions[%u].offset (%u) is "
                                        "greater than VkPhysicalDeviceLimits::maxVertexInputAttributeOffset (%u). %s",
                                        i, d, vertex_attrib_desc.offset, device_data->device_limits.maxVertexInputAttributeOffset,
                                        validation_error_map[VALIDATION_ERROR_14a004dc]);
                    }
                }
            }

            if (pCreateInfos[i].pStages != nullptr) {
                bool has_control = false;
                bool has_eval = false;

                for (uint32_t stage_index = 0; stage_index < pCreateInfos[i].stageCount; ++stage_index) {
                    if (pCreateInfos[i].pStages[stage_index].stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
                        has_control = true;
                    } else if (pCreateInfos[i].pStages[stage_index].stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
                        has_eval = true;
                    }
                }

                // pTessellationState is ignored without both tessellation control and tessellation evaluation shaders stages
                if (has_control && has_eval) {
                    if (pCreateInfos[i].pTessellationState == nullptr) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_096005b6, LayerName,
                                        "vkCreateGraphicsPipelines: if pCreateInfos[%d].pStages includes a tessellation control "
                                        "shader stage and a tessellation evaluation shader stage, "
                                        "pCreateInfos[%d].pTessellationState must not be NULL. %s",
                                        i, i, validation_error_map[VALIDATION_ERROR_096005b6]);
                    } else {
                        skip |= validate_struct_pnext(
                            report_data, "vkCreateGraphicsPipelines",
                            ParameterName("pCreateInfos[%i].pTessellationState->pNext", ParameterName::IndexVector{i}), NULL,
                            pCreateInfos[i].pTessellationState->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_0961c40d);

                        skip |= validate_reserved_flags(
                            report_data, "vkCreateGraphicsPipelines",
                            ParameterName("pCreateInfos[%i].pTessellationState->flags", ParameterName::IndexVector{i}),
                            pCreateInfos[i].pTessellationState->flags, VALIDATION_ERROR_10809005);

                        if (pCreateInfos[i].pTessellationState->sType !=
                            VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, VALIDATION_ERROR_1082b00b, LayerName,
                                            "vkCreateGraphicsPipelines: parameter pCreateInfos[%d].pTessellationState->sType must "
                                            "be VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO. %s",
                                            i, validation_error_map[VALIDATION_ERROR_1082b00b]);
                        }

                        if (pCreateInfos[i].pTessellationState->patchControlPoints == 0 ||
                            pCreateInfos[i].pTessellationState->patchControlPoints >
                                device_data->device_limits.maxTessellationPatchSize) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, VALIDATION_ERROR_1080097c, LayerName,
                                            "vkCreateGraphicsPipelines: invalid parameter "
                                            "pCreateInfos[%d].pTessellationState->patchControlPoints value %u. patchControlPoints "
                                            "should be >0 and <=%u. %s",
                                            i, pCreateInfos[i].pTessellationState->patchControlPoints,
                                            device_data->device_limits.maxTessellationPatchSize,
                                            validation_error_map[VALIDATION_ERROR_1080097c]);
                        }
                    }
                }
            }

            // pViewportState, pMultisampleState, pDepthStencilState, and pColorBlendState are ignored when
            // rasterization is disabled
            if ((pCreateInfos[i].pRasterizationState != nullptr) &&
                (pCreateInfos[i].pRasterizationState->rasterizerDiscardEnable == VK_FALSE)) {
                if (pCreateInfos[i].pViewportState == nullptr) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_096005dc, LayerName,
                                    "vkCreateGraphicsPipelines: if pCreateInfos[%d].pRasterizationState->rasterizerDiscardEnable "
                                    "is VK_FALSE, pCreateInfos[%d].pViewportState must be a pointer to a valid "
                                    "VkPipelineViewportStateCreateInfo structure. %s",
                                    i, i, validation_error_map[VALIDATION_ERROR_096005dc]);
                } else {
                    if (pCreateInfos[i].pViewportState->scissorCount != pCreateInfos[i].pViewportState->viewportCount) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_10c00988, LayerName,
                                        "Graphics Pipeline viewport count (%u) must match scissor count (%u). %s",
                                        pCreateInfos[i].pViewportState->viewportCount, pCreateInfos[i].pViewportState->scissorCount,
                                        validation_error_map[VALIDATION_ERROR_10c00988]);
                    }

                    skip |= validate_struct_pnext(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pViewportState->pNext", ParameterName::IndexVector{i}), NULL,
                        pCreateInfos[i].pViewportState->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_10c1c40d);

                    skip |= validate_reserved_flags(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pViewportState->flags", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pViewportState->flags, VALIDATION_ERROR_10c09005);

                    if (pCreateInfos[i].pViewportState->sType != VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, INVALID_STRUCT_STYPE, LayerName,
                                        "vkCreateGraphicsPipelines: parameter pCreateInfos[%d].pViewportState->sType must be "
                                        "VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO",
                                        i);
                    }

                    if (device_data->physical_device_features.multiViewport == false) {
                        if (pCreateInfos[i].pViewportState->viewportCount != 1) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, VALIDATION_ERROR_10c00980, LayerName,
                                            "vkCreateGraphicsPipelines: The multiViewport feature is not enabled, so "
                                            "pCreateInfos[%d].pViewportState->viewportCount must be 1 but is %d. %s",
                                            i, pCreateInfos[i].pViewportState->viewportCount,
                                            validation_error_map[VALIDATION_ERROR_10c00980]);
                        }
                        if (pCreateInfos[i].pViewportState->scissorCount != 1) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, VALIDATION_ERROR_10c00982, LayerName,
                                            "vkCreateGraphicsPipelines: The multiViewport feature is not enabled, so "
                                            "pCreateInfos[%d].pViewportState->scissorCount must be 1 but is %d. %s",
                                            i, pCreateInfos[i].pViewportState->scissorCount,
                                            validation_error_map[VALIDATION_ERROR_10c00982]);
                        }
                    } else {
                        if ((pCreateInfos[i].pViewportState->viewportCount < 1) ||
                            (pCreateInfos[i].pViewportState->viewportCount > device_data->device_limits.maxViewports)) {
                            skip |=
                                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_10c00984, LayerName,
                                        "vkCreateGraphicsPipelines: multiViewport feature is enabled; "
                                        "pCreateInfos[%d].pViewportState->viewportCount is %d but must be between 1 and "
                                        "maxViewports (%d), inclusive. %s",
                                        i, pCreateInfos[i].pViewportState->viewportCount, device_data->device_limits.maxViewports,
                                        validation_error_map[VALIDATION_ERROR_10c00984]);
                        }
                        if ((pCreateInfos[i].pViewportState->scissorCount < 1) ||
                            (pCreateInfos[i].pViewportState->scissorCount > device_data->device_limits.maxViewports)) {
                            skip |=
                                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_10c00986, LayerName,
                                        "vkCreateGraphicsPipelines: multiViewport feature is enabled; "
                                        "pCreateInfos[%d].pViewportState->scissorCount is %d but must be between 1 and "
                                        "maxViewports (%d), inclusive. %s",
                                        i, pCreateInfos[i].pViewportState->scissorCount, device_data->device_limits.maxViewports,
                                        validation_error_map[VALIDATION_ERROR_10c00986]);
                        }
                    }

                    if (pCreateInfos[i].pDynamicState != nullptr) {
                        bool has_dynamic_viewport = false;
                        bool has_dynamic_scissor = false;

                        for (uint32_t state_index = 0; state_index < pCreateInfos[i].pDynamicState->dynamicStateCount;
                             ++state_index) {
                            if (pCreateInfos[i].pDynamicState->pDynamicStates[state_index] == VK_DYNAMIC_STATE_VIEWPORT) {
                                has_dynamic_viewport = true;
                            } else if (pCreateInfos[i].pDynamicState->pDynamicStates[state_index] == VK_DYNAMIC_STATE_SCISSOR) {
                                has_dynamic_scissor = true;
                            }
                        }

                        // If no element of the pDynamicStates member of pDynamicState is VK_DYNAMIC_STATE_VIEWPORT, the pViewports
                        // member of pViewportState must be a pointer to an array of pViewportState->viewportCount VkViewport
                        // structures
                        if (!has_dynamic_viewport && (pCreateInfos[i].pViewportState->pViewports == nullptr)) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, VALIDATION_ERROR_096005d6, LayerName,
                                            "vkCreateGraphicsPipelines: if pCreateInfos[%d].pDynamicState->pDynamicStates does not "
                                            "contain VK_DYNAMIC_STATE_VIEWPORT, pCreateInfos[%d].pViewportState->pViewports must "
                                            "not be NULL. %s",
                                            i, i, validation_error_map[VALIDATION_ERROR_096005d6]);
                        }

                        // If no element of the pDynamicStates member of pDynamicState is VK_DYNAMIC_STATE_SCISSOR, the pScissors
                        // member
                        // of pViewportState must be a pointer to an array of pViewportState->scissorCount VkRect2D structures
                        if (!has_dynamic_scissor && (pCreateInfos[i].pViewportState->pScissors == nullptr)) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, VALIDATION_ERROR_096005d8, LayerName,
                                            "vkCreateGraphicsPipelines: if pCreateInfos[%d].pDynamicState->pDynamicStates does not "
                                            "contain VK_DYNAMIC_STATE_SCISSOR, pCreateInfos[%d].pViewportState->pScissors must not "
                                            "be NULL. %s",
                                            i, i, validation_error_map[VALIDATION_ERROR_096005d8]);
                        }
                    }
                }

                if (pCreateInfos[i].pMultisampleState == nullptr) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_096005de, LayerName,
                                    "vkCreateGraphicsPipelines: if pCreateInfos[%d].pRasterizationState->rasterizerDiscardEnable "
                                    "is VK_FALSE, pCreateInfos[%d].pMultisampleState must not be NULL. %s",
                                    i, i, validation_error_map[VALIDATION_ERROR_096005de]);
                } else {
                    skip |= validate_struct_pnext(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pMultisampleState->pNext", ParameterName::IndexVector{i}), NULL,
                        pCreateInfos[i].pMultisampleState->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_1001c40d);

                    skip |= validate_reserved_flags(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pMultisampleState->flags", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pMultisampleState->flags, VALIDATION_ERROR_10009005);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pMultisampleState->sampleShadingEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pMultisampleState->sampleShadingEnable);

                    skip |= validate_array(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pMultisampleState->rasterizationSamples", ParameterName::IndexVector{i}),
                        ParameterName("pCreateInfos[%i].pMultisampleState->pSampleMask", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pMultisampleState->rasterizationSamples, pCreateInfos[i].pMultisampleState->pSampleMask,
                        true, false, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pMultisampleState->alphaToCoverageEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pMultisampleState->alphaToCoverageEnable);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pMultisampleState->alphaToOneEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pMultisampleState->alphaToOneEnable);

                    if (pCreateInfos[i].pMultisampleState->sType != VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, INVALID_STRUCT_STYPE, LayerName,
                                        "vkCreateGraphicsPipelines: parameter pCreateInfos[%d].pMultisampleState->sType must be "
                                        "VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO",
                                        i);
                    }
                }

                // TODO: Conditional NULL check based on subpass depth/stencil attachment
                if (pCreateInfos[i].pDepthStencilState != nullptr) {
                    skip |= validate_struct_pnext(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->pNext", ParameterName::IndexVector{i}), NULL,
                        pCreateInfos[i].pDepthStencilState->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_0f61c40d);

                    skip |= validate_reserved_flags(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->flags", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pDepthStencilState->flags, VALIDATION_ERROR_0f609005);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->depthTestEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pDepthStencilState->depthTestEnable);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->depthWriteEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pDepthStencilState->depthWriteEnable);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->depthCompareOp", ParameterName::IndexVector{i}),
                        "VkCompareOp", AllVkCompareOpEnums, pCreateInfos[i].pDepthStencilState->depthCompareOp,
                        VALIDATION_ERROR_0f604001);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->depthBoundsTestEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pDepthStencilState->depthBoundsTestEnable);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->stencilTestEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pDepthStencilState->stencilTestEnable);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->front.failOp", ParameterName::IndexVector{i}),
                        "VkStencilOp", AllVkStencilOpEnums, pCreateInfos[i].pDepthStencilState->front.failOp,
                        VALIDATION_ERROR_13a08601);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->front.passOp", ParameterName::IndexVector{i}),
                        "VkStencilOp", AllVkStencilOpEnums, pCreateInfos[i].pDepthStencilState->front.passOp,
                        VALIDATION_ERROR_13a27801);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->front.depthFailOp", ParameterName::IndexVector{i}),
                        "VkStencilOp", AllVkStencilOpEnums, pCreateInfos[i].pDepthStencilState->front.depthFailOp,
                        VALIDATION_ERROR_13a04201);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->front.compareOp", ParameterName::IndexVector{i}),
                        "VkCompareOp", AllVkCompareOpEnums, pCreateInfos[i].pDepthStencilState->front.compareOp,
                        VALIDATION_ERROR_0f604001);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->back.failOp", ParameterName::IndexVector{i}),
                        "VkStencilOp", AllVkStencilOpEnums, pCreateInfos[i].pDepthStencilState->back.failOp,
                        VALIDATION_ERROR_13a08601);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->back.passOp", ParameterName::IndexVector{i}),
                        "VkStencilOp", AllVkStencilOpEnums, pCreateInfos[i].pDepthStencilState->back.passOp,
                        VALIDATION_ERROR_13a27801);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->back.depthFailOp", ParameterName::IndexVector{i}),
                        "VkStencilOp", AllVkStencilOpEnums, pCreateInfos[i].pDepthStencilState->back.depthFailOp,
                        VALIDATION_ERROR_13a04201);

                    skip |= validate_ranged_enum(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pDepthStencilState->back.compareOp", ParameterName::IndexVector{i}),
                        "VkCompareOp", AllVkCompareOpEnums, pCreateInfos[i].pDepthStencilState->back.compareOp,
                        VALIDATION_ERROR_0f604001);

                    if (pCreateInfos[i].pDepthStencilState->sType != VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, INVALID_STRUCT_STYPE, LayerName,
                                        "vkCreateGraphicsPipelines: parameter pCreateInfos[%d].pDepthStencilState->sType must be "
                                        "VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO",
                                        i);
                    }
                }

                // TODO: Conditional NULL check based on subpass color attachment
                if (pCreateInfos[i].pColorBlendState != nullptr) {
                    skip |= validate_struct_pnext(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pColorBlendState->pNext", ParameterName::IndexVector{i}), NULL,
                        pCreateInfos[i].pColorBlendState->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_0f41c40d);

                    skip |= validate_reserved_flags(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pColorBlendState->flags", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pColorBlendState->flags, VALIDATION_ERROR_0f409005);

                    skip |= validate_bool32(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pColorBlendState->logicOpEnable", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pColorBlendState->logicOpEnable);

                    skip |= validate_array(
                        report_data, "vkCreateGraphicsPipelines",
                        ParameterName("pCreateInfos[%i].pColorBlendState->attachmentCount", ParameterName::IndexVector{i}),
                        ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments", ParameterName::IndexVector{i}),
                        pCreateInfos[i].pColorBlendState->attachmentCount, pCreateInfos[i].pColorBlendState->pAttachments, false,
                        true, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);

                    if (pCreateInfos[i].pColorBlendState->pAttachments != NULL) {
                        for (uint32_t attachmentIndex = 0; attachmentIndex < pCreateInfos[i].pColorBlendState->attachmentCount;
                             ++attachmentIndex) {
                            skip |= validate_bool32(report_data, "vkCreateGraphicsPipelines",
                                                    ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].blendEnable",
                                                                  ParameterName::IndexVector{i, attachmentIndex}),
                                                    pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].blendEnable);

                            skip |= validate_ranged_enum(
                                report_data, "vkCreateGraphicsPipelines",
                                ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].srcColorBlendFactor",
                                              ParameterName::IndexVector{i, attachmentIndex}),
                                "VkBlendFactor", AllVkBlendFactorEnums,
                                pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].srcColorBlendFactor,
                                VALIDATION_ERROR_0f22cc01);

                            skip |= validate_ranged_enum(
                                report_data, "vkCreateGraphicsPipelines",
                                ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].dstColorBlendFactor",
                                              ParameterName::IndexVector{i, attachmentIndex}),
                                "VkBlendFactor", AllVkBlendFactorEnums,
                                pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].dstColorBlendFactor,
                                VALIDATION_ERROR_0f207001);

                            skip |= validate_ranged_enum(
                                report_data, "vkCreateGraphicsPipelines",
                                ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].colorBlendOp",
                                              ParameterName::IndexVector{i, attachmentIndex}),
                                "VkBlendOp", AllVkBlendOpEnums,
                                pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].colorBlendOp,
                                VALIDATION_ERROR_0f202001);

                            skip |= validate_ranged_enum(
                                report_data, "vkCreateGraphicsPipelines",
                                ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].srcAlphaBlendFactor",
                                              ParameterName::IndexVector{i, attachmentIndex}),
                                "VkBlendFactor", AllVkBlendFactorEnums,
                                pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].srcAlphaBlendFactor,
                                VALIDATION_ERROR_0f22c601);

                            skip |= validate_ranged_enum(
                                report_data, "vkCreateGraphicsPipelines",
                                ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].dstAlphaBlendFactor",
                                              ParameterName::IndexVector{i, attachmentIndex}),
                                "VkBlendFactor", AllVkBlendFactorEnums,
                                pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].dstAlphaBlendFactor,
                                VALIDATION_ERROR_0f206a01);

                            skip |= validate_ranged_enum(
                                report_data, "vkCreateGraphicsPipelines",
                                ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].alphaBlendOp",
                                              ParameterName::IndexVector{i, attachmentIndex}),
                                "VkBlendOp", AllVkBlendOpEnums,
                                pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].alphaBlendOp,
                                VALIDATION_ERROR_0f200801);

                            skip |=
                                validate_flags(report_data, "vkCreateGraphicsPipelines",
                                               ParameterName("pCreateInfos[%i].pColorBlendState->pAttachments[%i].colorWriteMask",
                                                             ParameterName::IndexVector{i, attachmentIndex}),
                                               "VkColorComponentFlagBits", AllVkColorComponentFlagBits,
                                               pCreateInfos[i].pColorBlendState->pAttachments[attachmentIndex].colorWriteMask,
                                               false, false, VALIDATION_ERROR_0f202201);
                        }
                    }

                    if (pCreateInfos[i].pColorBlendState->sType != VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, INVALID_STRUCT_STYPE, LayerName,
                                        "vkCreateGraphicsPipelines: parameter pCreateInfos[%d].pColorBlendState->sType must be "
                                        "VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO",
                                        i);
                    }

                    // If logicOpEnable is VK_TRUE, logicOp must be a valid VkLogicOp value
                    if (pCreateInfos[i].pColorBlendState->logicOpEnable == VK_TRUE) {
                        skip |= validate_ranged_enum(
                            report_data, "vkCreateGraphicsPipelines",
                            ParameterName("pCreateInfos[%i].pColorBlendState->logicOp", ParameterName::IndexVector{i}), "VkLogicOp",
                            AllVkLogicOpEnums, pCreateInfos[i].pColorBlendState->logicOp, VALIDATION_ERROR_0f4004be);
                    }
                }
            }
        }
        skip |= PreCreateGraphicsPipelines(device, pCreateInfos);
    }

    if (!skip) {
        result = device_data->dispatch_table.CreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos,
                                                                     pAllocator, pPipelines);
        validate_result(report_data, "vkCreateGraphicsPipelines", {}, result);
    }

    return result;
}

bool PreCreateComputePipelines(VkDevice device, const VkComputePipelineCreateInfo *pCreateInfos) {
    layer_data *data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    bool skip = false;
    if (pCreateInfos != nullptr) {
        // TODO: Handle count!
        uint32_t i = 0;
        skip |= validate_string(data->report_data, "vkCreateComputePipelines",
                                ParameterName("pCreateInfos[%i].stage.pName", ParameterName::IndexVector{i}),
                                pCreateInfos[i].stage.pName);
    }

    return skip;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateComputePipelines(my_data, pipelineCache, createInfoCount, pCreateInfos,
                                                          pAllocator, pPipelines);
    skip |= PreCreateComputePipelines(device, pCreateInfos);

    if (!skip) {
        result = my_data->dispatch_table.CreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator,
                                                                pPipelines);
        validate_result(my_data->report_data, "vkCreateComputePipelines", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyPipeline(my_data, pipeline, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyPipeline(device, pipeline, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreatePipelineLayout(my_data, pCreateInfo, pAllocator, pPipelineLayout);

    if (!skip) {
        result = my_data->dispatch_table.CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);

        validate_result(my_data->report_data, "vkCreatePipelineLayout", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout,
                                                 const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyPipelineLayout(my_data, pipelineLayout, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyPipelineLayout(device, pipelineLayout, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(VkDevice device, const VkSamplerCreateInfo *pCreateInfo,
                                             const VkAllocationCallbacks *pAllocator, VkSampler *pSampler) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != NULL);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateSampler(device_data, pCreateInfo, pAllocator, pSampler);

    if (pCreateInfo != nullptr) {
        if ((device_data->physical_device_features.samplerAnisotropy == false) && (pCreateInfo->maxAnisotropy != 1.0)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCreateSampler(): The samplerAnisotropy feature was not enabled at device-creation time, so the "
                            "maxAnisotropy member of the VkSamplerCreateInfo structure must be 1.0 but is %f.",
                            pCreateInfo->maxAnisotropy);
        }

        // If compareEnable is VK_TRUE, compareOp must be a valid VkCompareOp value
        if (pCreateInfo->compareEnable == VK_TRUE) {
            skip |= validate_ranged_enum(report_data, "vkCreateSampler", "pCreateInfo->compareOp", "VkCompareOp",
                                         AllVkCompareOpEnums, pCreateInfo->compareOp, VALIDATION_ERROR_12600870);
        }

        // If any of addressModeU, addressModeV or addressModeW are VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, borderColor must be a
        // valid VkBorderColor value
        if ((pCreateInfo->addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) ||
            (pCreateInfo->addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) ||
            (pCreateInfo->addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)) {
            skip |= validate_ranged_enum(report_data, "vkCreateSampler", "pCreateInfo->borderColor", "VkBorderColor",
                                         AllVkBorderColorEnums, pCreateInfo->borderColor, VALIDATION_ERROR_1260086c);
        }

        // If any of addressModeU, addressModeV or addressModeW are VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, the
        // VK_KHR_sampler_mirror_clamp_to_edge extension must be enabled
        if (!device_data->extensions.vk_khr_sampler_mirror_clamp_to_edge &&
            ((pCreateInfo->addressModeU == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE) ||
             (pCreateInfo->addressModeV == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE) ||
             (pCreateInfo->addressModeW == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE))) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1260086e, LayerName,
                        "vkCreateSampler(): A VkSamplerAddressMode value is set to VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE "
                        "but the VK_KHR_sampler_mirror_clamp_to_edge extension has not been enabled. %s",
                        validation_error_map[VALIDATION_ERROR_1260086e]);
        }
    }

    if (!skip) {
        result = device_data->dispatch_table.CreateSampler(device, pCreateInfo, pAllocator, pSampler);

        validate_result(report_data, "vkCreateSampler", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroySampler(my_data, sampler, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroySampler(device, sampler, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                         const VkAllocationCallbacks *pAllocator,
                                                         VkDescriptorSetLayout *pSetLayout) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateDescriptorSetLayout(device_data, pCreateInfo, pAllocator, pSetLayout);

    // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
    if ((pCreateInfo != nullptr) && (pCreateInfo->pBindings != nullptr)) {
        for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i) {
            if (pCreateInfo->pBindings[i].descriptorCount != 0) {
                // If descriptorType is VK_DESCRIPTOR_TYPE_SAMPLER or VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, and descriptorCount
                // is not 0 and pImmutableSamplers is not NULL, pImmutableSamplers must be a pointer to an array of descriptorCount
                // valid VkSampler handles
                if (((pCreateInfo->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) ||
                     (pCreateInfo->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) &&
                    (pCreateInfo->pBindings[i].pImmutableSamplers != nullptr)) {
                    for (uint32_t descriptor_index = 0; descriptor_index < pCreateInfo->pBindings[i].descriptorCount;
                         ++descriptor_index) {
                        if (pCreateInfo->pBindings[i].pImmutableSamplers[descriptor_index] == VK_NULL_HANDLE) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, REQUIRED_PARAMETER, LayerName,
                                            "vkCreateDescriptorSetLayout: required parameter "
                                            "pCreateInfo->pBindings[%d].pImmutableSamplers[%d]"
                                            " specified as VK_NULL_HANDLE",
                                            i, descriptor_index);
                        }
                    }
                }

                // If descriptorCount is not 0, stageFlags must be a valid combination of VkShaderStageFlagBits values
                if ((pCreateInfo->pBindings[i].stageFlags != 0) &&
                    ((pCreateInfo->pBindings[i].stageFlags & (~AllVkShaderStageFlagBits)) != 0)) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_04e00236, LayerName,
                        "vkCreateDescriptorSetLayout(): if pCreateInfo->pBindings[%d].descriptorCount is not 0, "
                        "pCreateInfo->pBindings[%d].stageFlags must be a valid combination of VkShaderStageFlagBits values. %s",
                        i, i, validation_error_map[VALIDATION_ERROR_04e00236]);
                }
            }
        }
    }

    if (!skip) {
        result = device_data->dispatch_table.CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);

        validate_result(report_data, "vkCreateDescriptorSetLayout", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                      const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyDescriptorSetLayout(my_data, descriptorSetLayout, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pDescriptorPool) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateDescriptorPool(my_data, pCreateInfo, pAllocator, pDescriptorPool);

    /* TODOVV: How do we validate maxSets? Probably belongs in the limits layer? */

    if (!skip) {
        result = my_data->dispatch_table.CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);

        validate_result(my_data->report_data, "vkCreateDescriptorPool", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                 const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyDescriptorPool(my_data, descriptorPool, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyDescriptorPool(device, descriptorPool, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                   VkDescriptorPoolResetFlags flags) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkResetDescriptorPool(my_data, descriptorPool, flags);

    if (!skip) {
        result = my_data->dispatch_table.ResetDescriptorPool(device, descriptorPool, flags);

        validate_result(my_data->report_data, "vkResetDescriptorPool", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                      VkDescriptorSet *pDescriptorSets) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkAllocateDescriptorSets(my_data, pAllocateInfo, pDescriptorSets);

    if (!skip) {
        result = my_data->dispatch_table.AllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets);

        validate_result(my_data->report_data, "vkAllocateDescriptorSets", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount,
                                                  const VkDescriptorSet *pDescriptorSets) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkFreeDescriptorSets(device_data, descriptorPool, descriptorSetCount, pDescriptorSets);

    // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
    // This is an array of handles, where the elements are allowed to be VK_NULL_HANDLE, and does not require any validation beyond
    // validate_array()
    skip |= validate_array(report_data, "vkFreeDescriptorSets", "descriptorSetCount", "pDescriptorSets", descriptorSetCount,
                           pDescriptorSets, true, true, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);

    if (!skip) {
        result = device_data->dispatch_table.FreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets);

        validate_result(report_data, "vkFreeDescriptorSets", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                const VkWriteDescriptorSet *pDescriptorWrites, uint32_t descriptorCopyCount,
                                                const VkCopyDescriptorSet *pDescriptorCopies) {
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != NULL);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkUpdateDescriptorSets(device_data, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount,
                                                        pDescriptorCopies);

    // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
    if (pDescriptorWrites != NULL) {
        for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
            // descriptorCount must be greater than 0
            if (pDescriptorWrites[i].descriptorCount == 0) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_15c0441b, LayerName,
                            "vkUpdateDescriptorSets(): parameter pDescriptorWrites[%d].descriptorCount must be greater than 0. %s",
                            i, validation_error_map[VALIDATION_ERROR_15c0441b]);
            }

            // dstSet must be a valid VkDescriptorSet handle
            skip |= validate_required_handle(report_data, "vkUpdateDescriptorSets",
                                             ParameterName("pDescriptorWrites[%i].dstSet", ParameterName::IndexVector{i}),
                                             pDescriptorWrites[i].dstSet);

            if ((pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) ||
                (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ||
                (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) ||
                (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) ||
                (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT)) {
                // If descriptorType is VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                // VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE or VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                // pImageInfo must be a pointer to an array of descriptorCount valid VkDescriptorImageInfo structures
                if (pDescriptorWrites[i].pImageInfo == nullptr) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_15c00284, LayerName,
                                    "vkUpdateDescriptorSets(): if pDescriptorWrites[%d].descriptorType is "
                                    "VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, "
                                    "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE or "
                                    "VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, pDescriptorWrites[%d].pImageInfo must not be NULL. %s",
                                    i, i, validation_error_map[VALIDATION_ERROR_15c00284]);
                } else if (pDescriptorWrites[i].descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) {
                    // If descriptorType is VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    // VK_DESCRIPTOR_TYPE_STORAGE_IMAGE or VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, the imageView and imageLayout
                    // members of any given element of pImageInfo must be a valid VkImageView and VkImageLayout, respectively
                    for (uint32_t descriptor_index = 0; descriptor_index < pDescriptorWrites[i].descriptorCount;
                         ++descriptor_index) {
                        skip |= validate_required_handle(report_data, "vkUpdateDescriptorSets",
                                                         ParameterName("pDescriptorWrites[%i].pImageInfo[%i].imageView",
                                                                       ParameterName::IndexVector{i, descriptor_index}),
                                                         pDescriptorWrites[i].pImageInfo[descriptor_index].imageView);
                        skip |= validate_ranged_enum(report_data, "vkUpdateDescriptorSets",
                                                     ParameterName("pDescriptorWrites[%i].pImageInfo[%i].imageLayout",
                                                                   ParameterName::IndexVector{i, descriptor_index}),
                                                     "VkImageLayout", AllVkImageLayoutEnums,
                                                     pDescriptorWrites[i].pImageInfo[descriptor_index].imageLayout,
                                                     VALIDATION_ERROR_UNDEFINED);
                    }
                }
            } else if ((pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) ||
                       (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
                       (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
                       (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)) {
                // If descriptorType is VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, pBufferInfo must be a
                // pointer to an array of descriptorCount valid VkDescriptorBufferInfo structures
                if (pDescriptorWrites[i].pBufferInfo == nullptr) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_15c00288, LayerName,
                                    "vkUpdateDescriptorSets(): if pDescriptorWrites[%d].descriptorType is "
                                    "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, "
                                    "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, "
                                    "pDescriptorWrites[%d].pBufferInfo must not be NULL. %s",
                                    i, i, validation_error_map[VALIDATION_ERROR_15c00288]);
                } else {
                    for (uint32_t descriptorIndex = 0; descriptorIndex < pDescriptorWrites[i].descriptorCount; ++descriptorIndex) {
                        skip |= validate_required_handle(report_data, "vkUpdateDescriptorSets",
                                                         ParameterName("pDescriptorWrites[%i].pBufferInfo[%i].buffer",
                                                                       ParameterName::IndexVector{i, descriptorIndex}),
                                                         pDescriptorWrites[i].pBufferInfo[descriptorIndex].buffer);
                    }
                }
            } else if ((pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) ||
                       (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)) {
                // If descriptorType is VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                // pTexelBufferView must be a pointer to an array of descriptorCount valid VkBufferView handles
                if (pDescriptorWrites[i].pTexelBufferView == nullptr) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_15c00286, LayerName,
                                    "vkUpdateDescriptorSets(): if pDescriptorWrites[%d].descriptorType is "
                                    "VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, "
                                    "pDescriptorWrites[%d].pTexelBufferView must not be NULL. %s",
                                    i, i, validation_error_map[VALIDATION_ERROR_15c00286]);
                } else {
                    for (uint32_t descriptor_index = 0; descriptor_index < pDescriptorWrites[i].descriptorCount;
                         ++descriptor_index) {
                        skip |= validate_required_handle(report_data, "vkUpdateDescriptorSets",
                                                         ParameterName("pDescriptorWrites[%i].pTexelBufferView[%i]",
                                                                       ParameterName::IndexVector{i, descriptor_index}),
                                                         pDescriptorWrites[i].pTexelBufferView[descriptor_index]);
                    }
                }
            }

            if ((pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) ||
                (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)) {
                VkDeviceSize uniformAlignment = device_data->device_limits.minUniformBufferOffsetAlignment;
                for (uint32_t j = 0; j < pDescriptorWrites[i].descriptorCount; j++) {
                    if (pDescriptorWrites[i].pBufferInfo != NULL) {
                        if (SafeModulo(pDescriptorWrites[i].pBufferInfo[j].offset, uniformAlignment) != 0) {
                            skip |= log_msg(
                                device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT,
                                VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT, 0, __LINE__, VALIDATION_ERROR_15c0028e, LayerName,
                                "vkUpdateDescriptorSets(): pDescriptorWrites[%d].pBufferInfo[%d].offset (0x%" PRIxLEAST64
                                ") must be a multiple of device limit minUniformBufferOffsetAlignment 0x%" PRIxLEAST64 ". %s",
                                i, j, pDescriptorWrites[i].pBufferInfo[j].offset, uniformAlignment,
                                validation_error_map[VALIDATION_ERROR_15c0028e]);
                        }
                    }
                }
            } else if ((pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
                       (pDescriptorWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)) {
                VkDeviceSize storageAlignment = device_data->device_limits.minStorageBufferOffsetAlignment;
                for (uint32_t j = 0; j < pDescriptorWrites[i].descriptorCount; j++) {
                    if (pDescriptorWrites[i].pBufferInfo != NULL) {
                        if (SafeModulo(pDescriptorWrites[i].pBufferInfo[j].offset, storageAlignment) != 0) {
                            skip |= log_msg(
                                device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT,
                                VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT, 0, __LINE__, VALIDATION_ERROR_15c00290, LayerName,
                                "vkUpdateDescriptorSets(): pDescriptorWrites[%d].pBufferInfo[%d].offset (0x%" PRIxLEAST64
                                ") must be a multiple of device limit minStorageBufferOffsetAlignment 0x%" PRIxLEAST64 ". %s",
                                i, j, pDescriptorWrites[i].pBufferInfo[j].offset, storageAlignment,
                                validation_error_map[VALIDATION_ERROR_15c00290]);
                        }
                    }
                }
            }
        }
    }

    if (!skip) {
        device_data->dispatch_table.UpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount,
                                                         pDescriptorCopies);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo *pCreateInfo,
                                                 const VkAllocationCallbacks *pAllocator, VkFramebuffer *pFramebuffer) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateFramebuffer(my_data, pCreateInfo, pAllocator, pFramebuffer);

    if (!skip) {
        result = my_data->dispatch_table.CreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer);

        validate_result(my_data->report_data, "vkCreateFramebuffer", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyFramebuffer(my_data, framebuffer, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyFramebuffer(device, framebuffer, pAllocator);
    }
}

static bool PreCreateRenderPass(layer_data *dev_data, const VkRenderPassCreateInfo *pCreateInfo) {
    bool skip = false;
    uint32_t max_color_attachments = dev_data->device_limits.maxColorAttachments;

    for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i) {
        if (pCreateInfo->pAttachments[i].format == VK_FORMAT_UNDEFINED) {
            std::stringstream ss;
            ss << "vkCreateRenderPass: pCreateInfo->pAttachments[" << i << "].format is VK_FORMAT_UNDEFINED. "
               << validation_error_map[VALIDATION_ERROR_00809201];
            skip |= log_msg(dev_data->report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                            __LINE__, VALIDATION_ERROR_00809201, "IMAGE", "%s", ss.str().c_str());
        }
        if (pCreateInfo->pAttachments[i].finalLayout == VK_IMAGE_LAYOUT_UNDEFINED ||
            pCreateInfo->pAttachments[i].finalLayout == VK_IMAGE_LAYOUT_PREINITIALIZED) {
            skip |= log_msg(dev_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                            __LINE__, VALIDATION_ERROR_00800696, "DL",
                            "pCreateInfo->pAttachments[%d].finalLayout must not be VK_IMAGE_LAYOUT_UNDEFINED or "
                            "VK_IMAGE_LAYOUT_PREINITIALIZED. %s",
                            i, validation_error_map[VALIDATION_ERROR_00800696]);
        }
    }

    for (uint32_t i = 0; i < pCreateInfo->subpassCount; ++i) {
        if (pCreateInfo->pSubpasses[i].colorAttachmentCount > max_color_attachments) {
            skip |=
                log_msg(dev_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1400069a, "DL", "Cannot create a render pass with %d color attachments. Max is %d. %s",
                        pCreateInfo->pSubpasses[i].colorAttachmentCount, max_color_attachments,
                        validation_error_map[VALIDATION_ERROR_1400069a]);
        }
    }
    return skip;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateRenderPass(my_data, pCreateInfo, pAllocator, pRenderPass);
    skip |= PreCreateRenderPass(my_data, pCreateInfo);

    if (!skip) {
        result = my_data->dispatch_table.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);

        validate_result(my_data->report_data, "vkCreateRenderPass", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyRenderPass(my_data, renderPass, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyRenderPass(device, renderPass, pAllocator);
    }
}

VKAPI_ATTR void VKAPI_CALL GetRenderAreaGranularity(VkDevice device, VkRenderPass renderPass, VkExtent2D *pGranularity) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetRenderAreaGranularity(my_data, renderPass, pGranularity);

    if (!skip) {
        my_data->dispatch_table.GetRenderAreaGranularity(device, renderPass, pGranularity);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
                                                 const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= ValidateDeviceQueueFamily(my_data, pCreateInfo->queueFamilyIndex, "vkCreateCommandPool",
                                      "pCreateInfo->queueFamilyIndex", VALIDATION_ERROR_02c0004e);

    skip |= parameter_validation_vkCreateCommandPool(my_data, pCreateInfo, pAllocator, pCommandPool);

    if (!skip) {
        result = my_data->dispatch_table.CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);

        validate_result(my_data->report_data, "vkCreateCommandPool", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyCommandPool(my_data, commandPool, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyCommandPool(device, commandPool, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkResetCommandPool(my_data, commandPool, flags);

    if (!skip) {
        result = my_data->dispatch_table.ResetCommandPool(device, commandPool, flags);

        validate_result(my_data->report_data, "vkResetCommandPool", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
                                                      VkCommandBuffer *pCommandBuffers) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkAllocateCommandBuffers(my_data, pAllocateInfo, pCommandBuffers);

    if (!skip) {
        result = my_data->dispatch_table.AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);

        validate_result(my_data->report_data, "vkAllocateCommandBuffers", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                              const VkCommandBuffer *pCommandBuffers) {
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkFreeCommandBuffers(device_data, commandPool, commandBufferCount, pCommandBuffers);

    // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
    // This is an array of handles, where the elements are allowed to be VK_NULL_HANDLE, and does not require any validation beyond
    // validate_array()
    skip |= validate_array(report_data, "vkFreeCommandBuffers", "commandBufferCount", "pCommandBuffers", commandBufferCount,
                           pCommandBuffers, true, true, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);

    if (!skip) {
        device_data->dispatch_table.FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
    }
}

static bool PreBeginCommandBuffer(layer_data *dev_data, VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo) {
    bool skip = false;
    const VkCommandBufferInheritanceInfo *pInfo = pBeginInfo->pInheritanceInfo;

    if (pInfo != NULL) {
        if ((dev_data->physical_device_features.inheritedQueries == VK_FALSE) && (pInfo->occlusionQueryEnable != VK_FALSE)) {
            skip |= log_msg(dev_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_02a00070, LayerName,
                            "Cannot set inherited occlusionQueryEnable in vkBeginCommandBuffer() when device does not support "
                            "inheritedQueries. %s",
                            validation_error_map[VALIDATION_ERROR_02a00070]);
        }
        if ((dev_data->physical_device_features.inheritedQueries != VK_FALSE) && (pInfo->occlusionQueryEnable != VK_FALSE)) {
            skip |= validate_flags(dev_data->report_data, "vkBeginCommandBuffer", "pBeginInfo->pInheritanceInfo->queryFlags",
                                   "VkQueryControlFlagBits", AllVkQueryControlFlagBits, pInfo->queryFlags, false, false,
                                   VALIDATION_ERROR_02a00072);
        }
    }
    return skip;
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(device_data != nullptr);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkBeginCommandBuffer(device_data, pBeginInfo);

    // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
    // TODO: pBeginInfo->pInheritanceInfo must not be NULL if commandBuffer is a secondary command buffer
    skip |= validate_struct_type(report_data, "vkBeginCommandBuffer", "pBeginInfo->pInheritanceInfo",
                                 "VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO", pBeginInfo->pInheritanceInfo,
                                 VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, false, VALIDATION_ERROR_UNDEFINED);

    if (pBeginInfo->pInheritanceInfo != NULL) {
        skip |=
            validate_struct_pnext(report_data, "vkBeginCommandBuffer", "pBeginInfo->pInheritanceInfo->pNext", NULL,
                                  pBeginInfo->pInheritanceInfo->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_0281c40d);

        skip |= validate_bool32(report_data, "vkBeginCommandBuffer", "pBeginInfo->pInheritanceInfo->occlusionQueryEnable",
                                pBeginInfo->pInheritanceInfo->occlusionQueryEnable);

        // TODO: This only needs to be validated when the inherited queries feature is enabled
        // skip |= validate_flags(report_data, "vkBeginCommandBuffer", "pBeginInfo->pInheritanceInfo->queryFlags",
        // "VkQueryControlFlagBits", AllVkQueryControlFlagBits, pBeginInfo->pInheritanceInfo->queryFlags, false);

        // TODO: This must be 0 if the pipeline statistics queries feature is not enabled
        skip |= validate_flags(report_data, "vkBeginCommandBuffer", "pBeginInfo->pInheritanceInfo->pipelineStatistics",
                               "VkQueryPipelineStatisticFlagBits", AllVkQueryPipelineStatisticFlagBits,
                               pBeginInfo->pInheritanceInfo->pipelineStatistics, false, false, VALIDATION_ERROR_UNDEFINED);
    }

    skip |= PreBeginCommandBuffer(device_data, commandBuffer, pBeginInfo);

    if (!skip) {
        result = device_data->dispatch_table.BeginCommandBuffer(commandBuffer, pBeginInfo);

        validate_result(report_data, "vkBeginCommandBuffer", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer commandBuffer) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    VkResult result = my_data->dispatch_table.EndCommandBuffer(commandBuffer);

    validate_result(my_data->report_data, "vkEndCommandBuffer", {}, result);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    bool skip = parameter_validation_vkResetCommandBuffer(my_data, flags);

    if (!skip) {
        result = my_data->dispatch_table.ResetCommandBuffer(commandBuffer, flags);

        validate_result(my_data->report_data, "vkResetCommandBuffer", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                           VkPipeline pipeline) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBindPipeline(my_data, pipelineBindPoint, pipeline);

    if (!skip) {
        my_data->dispatch_table.CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
    }
}

static bool preCmdSetViewport(layer_data *my_data, uint32_t first_viewport, uint32_t viewport_count, const VkViewport *viewports) {
    debug_report_data *report_data = my_data->report_data;

    bool skip = validate_array(report_data, "vkCmdSetViewport", "viewportCount", "pViewports", viewport_count, viewports, true,
                               true, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);

    if (viewport_count > 0 && viewports != nullptr) {
        const VkPhysicalDeviceLimits &limits = my_data->device_limits;
        for (uint32_t viewportIndex = 0; viewportIndex < viewport_count; ++viewportIndex) {
            const VkViewport &viewport = viewports[viewportIndex];

            if (my_data->physical_device_features.multiViewport == false) {
                if (viewport_count != 1) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        DEVICE_FEATURE, LayerName,
                        "vkCmdSetViewport(): The multiViewport feature is not enabled, so viewportCount must be 1 but is %d.",
                        viewport_count);
                }
                if (first_viewport != 0) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        DEVICE_FEATURE, LayerName,
                        "vkCmdSetViewport(): The multiViewport feature is not enabled, so firstViewport must be 0 but is %d.",
                        first_viewport);
                }
            }

            if (viewport.width <= 0 || viewport.width > limits.maxViewportDimensions[0]) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_15000996, LayerName,
                                "vkCmdSetViewport %d: width (%f) exceeds permitted bounds (0,%u). %s", viewportIndex,
                                viewport.width, limits.maxViewportDimensions[0], validation_error_map[VALIDATION_ERROR_15000996]);
            }

            if (my_data->extensions.vk_amd_negative_viewport_height || my_data->extensions.vk_khr_maintenance1) {
                // Check lower bound against negative viewport height instead of zero
                if (viewport.height <= -(static_cast<int32_t>(limits.maxViewportDimensions[1])) ||
                    (viewport.height > limits.maxViewportDimensions[1])) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_1500099a, LayerName,
                                    "vkCmdSetViewport %d: height (%f) exceeds permitted bounds (-%u,%u). %s", viewportIndex,
                                    viewport.height, limits.maxViewportDimensions[1], limits.maxViewportDimensions[1],
                                    validation_error_map[VALIDATION_ERROR_1500099a]);
                }
            } else {
                if ((viewport.height <= 0) || (viewport.height > limits.maxViewportDimensions[1])) {
                    skip |=
                        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_15000998, LayerName,
                                "vkCmdSetViewport %d: height (%f) exceeds permitted bounds (0,%u). %s", viewportIndex,
                                viewport.height, limits.maxViewportDimensions[1], validation_error_map[VALIDATION_ERROR_15000998]);
                }
            }

            if (viewport.x < limits.viewportBoundsRange[0] || viewport.x > limits.viewportBoundsRange[1]) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_1500099e, LayerName,
                                "vkCmdSetViewport %d: x (%f) exceeds permitted bounds (%f,%f). %s", viewportIndex, viewport.x,
                                limits.viewportBoundsRange[0], limits.viewportBoundsRange[1],
                                validation_error_map[VALIDATION_ERROR_1500099e]);
            }

            if (viewport.y < limits.viewportBoundsRange[0] || viewport.y > limits.viewportBoundsRange[1]) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_1500099e, LayerName,
                                "vkCmdSetViewport %d: y (%f) exceeds permitted bounds (%f,%f). %s", viewportIndex, viewport.y,
                                limits.viewportBoundsRange[0], limits.viewportBoundsRange[1],
                                validation_error_map[VALIDATION_ERROR_1500099e]);
            }

            if (viewport.x + viewport.width > limits.viewportBoundsRange[1]) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_150009a0, LayerName,
                            "vkCmdSetViewport %d: x (%f) + width (%f) exceeds permitted bound (%f). %s", viewportIndex, viewport.x,
                            viewport.width, limits.viewportBoundsRange[1], validation_error_map[VALIDATION_ERROR_150009a0]);
            }

            if (viewport.y + viewport.height > limits.viewportBoundsRange[1]) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_150009a2, LayerName,
                            "vkCmdSetViewport %d: y (%f) + height (%f) exceeds permitted bound (%f). %s", viewportIndex, viewport.y,
                            viewport.height, limits.viewportBoundsRange[1], validation_error_map[VALIDATION_ERROR_150009a2]);
            }
        }
    }

    return skip;
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount,
                                          const VkViewport *pViewports) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= preCmdSetViewport(my_data, firstViewport, viewportCount, pViewports);

    if (!skip) {
        my_data->dispatch_table.CmdSetViewport(commandBuffer, firstViewport, viewportCount, pViewports);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor, uint32_t scissorCount,
                                         const VkRect2D *pScissors) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);
    debug_report_data *report_data = my_data->report_data;

    skip |= parameter_validation_vkCmdSetScissor(my_data, firstScissor, scissorCount, pScissors);

    if (my_data->physical_device_features.multiViewport == false) {
        if (scissorCount != 1) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCmdSetScissor(): The multiViewport feature is not enabled, so scissorCount must be 1 but is %d.",
                            scissorCount);
        }
        if (firstScissor != 0) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCmdSetScissor(): The multiViewport feature is not enabled, so firstScissor must be 0 but is %d.",
                            firstScissor);
        }
    }

    for (uint32_t scissorIndex = 0; scissorIndex < scissorCount; ++scissorIndex) {
        const VkRect2D &pScissor = pScissors[scissorIndex];

        if (pScissor.offset.x < 0) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_1d8004a6, LayerName, "vkCmdSetScissor %d: offset.x (%d) must not be negative. %s",
                            scissorIndex, pScissor.offset.x, validation_error_map[VALIDATION_ERROR_1d8004a6]);
        } else if (static_cast<int32_t>(pScissor.extent.width) > (INT_MAX - pScissor.offset.x)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_1d8004a8, LayerName,
                            "vkCmdSetScissor %d: adding offset.x (%d) and extent.width (%u) will overflow. %s", scissorIndex,
                            pScissor.offset.x, pScissor.extent.width, validation_error_map[VALIDATION_ERROR_1d8004a8]);
        }

        if (pScissor.offset.y < 0) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_1d8004a6, LayerName, "vkCmdSetScissor %d: offset.y (%d) must not be negative. %s",
                            scissorIndex, pScissor.offset.y, validation_error_map[VALIDATION_ERROR_1d8004a6]);
        } else if (static_cast<int32_t>(pScissor.extent.height) > (INT_MAX - pScissor.offset.y)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_1d8004aa, LayerName,
                            "vkCmdSetScissor %d: adding offset.y (%d) and extent.height (%u) will overflow. %s", scissorIndex,
                            pScissor.offset.y, pScissor.extent.height, validation_error_map[VALIDATION_ERROR_1d8004aa]);
        }
    }

    if (!skip) {
        my_data->dispatch_table.CmdSetScissor(commandBuffer, firstScissor, scissorCount, pScissors);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetLineWidth(VkCommandBuffer commandBuffer, float lineWidth) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    my_data->dispatch_table.CmdSetLineWidth(commandBuffer, lineWidth);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBias(VkCommandBuffer commandBuffer, float depthBiasConstantFactor, float depthBiasClamp,
                                           float depthBiasSlopeFactor) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    my_data->dispatch_table.CmdSetDepthBias(commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
}

VKAPI_ATTR void VKAPI_CALL CmdSetBlendConstants(VkCommandBuffer commandBuffer, const float blendConstants[4]) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetBlendConstants(my_data, blendConstants);

    if (!skip) {
        my_data->dispatch_table.CmdSetBlendConstants(commandBuffer, blendConstants);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBounds(VkCommandBuffer commandBuffer, float minDepthBounds, float maxDepthBounds) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    my_data->dispatch_table.CmdSetDepthBounds(commandBuffer, minDepthBounds, maxDepthBounds);
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilCompareMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask,
                                                    uint32_t compareMask) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetStencilCompareMask(my_data, faceMask, compareMask);

    if (!skip) {
        my_data->dispatch_table.CmdSetStencilCompareMask(commandBuffer, faceMask, compareMask);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilWriteMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t writeMask) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetStencilWriteMask(my_data, faceMask, writeMask);

    if (!skip) {
        my_data->dispatch_table.CmdSetStencilWriteMask(commandBuffer, faceMask, writeMask);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilReference(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t reference) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetStencilReference(my_data, faceMask, reference);

    if (!skip) {
        my_data->dispatch_table.CmdSetStencilReference(commandBuffer, faceMask, reference);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                 VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
                                                 const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
                                                 const uint32_t *pDynamicOffsets) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBindDescriptorSets(my_data, pipelineBindPoint, layout, firstSet,
                                                         descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);

    if (!skip) {
        my_data->dispatch_table.CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount,
                                                      pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                              VkIndexType indexType) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBindIndexBuffer(my_data, buffer, offset, indexType);

    if (!skip) {
        my_data->dispatch_table.CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
                                                const VkBuffer *pBuffers, const VkDeviceSize *pOffsets) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBindVertexBuffers(my_data, firstBinding, bindingCount, pBuffers, pOffsets);

    if (!skip) {
        my_data->dispatch_table.CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
    }
}

static bool PreCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                       uint32_t firstInstance) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    if (vertexCount == 0) {
        // TODO: Verify against Valid Usage section. I don't see a non-zero vertexCount listed, may need to add that and make
        // this an error or leave as is.
        log_msg(my_data->report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                REQUIRED_PARAMETER, LayerName, "vkCmdDraw parameter, uint32_t vertexCount, is 0");
        return false;
    }

    if (instanceCount == 0) {
        // TODO: Verify against Valid Usage section. I don't see a non-zero instanceCount listed, may need to add that and make
        // this an error or leave as is.
        log_msg(my_data->report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                REQUIRED_PARAMETER, LayerName, "vkCmdDraw parameter, uint32_t instanceCount, is 0");
        return false;
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                   uint32_t firstVertex, uint32_t firstInstance) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    PreCmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);

    my_data->dispatch_table.CmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                          uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    my_data->dispatch_table.CmdDrawIndexed(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                           uint32_t stride) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    if (!my_data->physical_device_features.multiDrawIndirect && ((count > 1))) {
        skip = log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                       DEVICE_FEATURE, LayerName,
                       "CmdDrawIndirect(): Device feature multiDrawIndirect disabled: count must be 0 or 1 but is %d", count);
    }
    skip |= parameter_validation_vkCmdDrawIndirect(my_data, buffer, offset, count, stride);

    if (!skip) {
        my_data->dispatch_table.CmdDrawIndirect(commandBuffer, buffer, offset, count, stride);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                  uint32_t count, uint32_t stride) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);
    if (!my_data->physical_device_features.multiDrawIndirect && ((count > 1))) {
        skip =
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    DEVICE_FEATURE, LayerName,
                    "CmdDrawIndexedIndirect(): Device feature multiDrawIndirect disabled: count must be 0 or 1 but is %d", count);
    }
    skip |= parameter_validation_vkCmdDrawIndexedIndirect(my_data, buffer, offset, count, stride);

    if (!skip) {
        my_data->dispatch_table.CmdDrawIndexedIndirect(commandBuffer, buffer, offset, count, stride);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    my_data->dispatch_table.CmdDispatch(commandBuffer, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdDispatchIndirect(my_data, buffer, offset);

    if (!skip) {
        my_data->dispatch_table.CmdDispatchIndirect(commandBuffer, buffer, offset);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                                         uint32_t regionCount, const VkBufferCopy *pRegions) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdCopyBuffer(my_data, srcBuffer, dstBuffer, regionCount, pRegions);

    if (!skip) {
        my_data->dispatch_table.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
    }
}

static bool PreCmdCopyImage(VkCommandBuffer commandBuffer, const VkImageCopy *pRegions) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    if (pRegions != nullptr) {
        if ((pRegions->srcSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                    VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(
                my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                VALIDATION_ERROR_0a600c01, LayerName,
                "vkCmdCopyImage() parameter, VkImageAspect pRegions->srcSubresource.aspectMask, is an unrecognized enumerator. %s",
                validation_error_map[VALIDATION_ERROR_0a600c01]);
            return false;
        }
        if ((pRegions->dstSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                    VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(
                my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                VALIDATION_ERROR_0a600c01, LayerName,
                "vkCmdCopyImage() parameter, VkImageAspect pRegions->dstSubresource.aspectMask, is an unrecognized enumerator. %s",
                validation_error_map[VALIDATION_ERROR_0a600c01]);
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                        VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                        const VkImageCopy *pRegions) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdCopyImage(my_data, srcImage, srcImageLayout, dstImage, dstImageLayout,
                                                regionCount, pRegions);

    if (!skip) {
        PreCmdCopyImage(commandBuffer, pRegions);

        my_data->dispatch_table.CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount,
                                             pRegions);
    }
}

static bool PreCmdBlitImage(VkCommandBuffer commandBuffer, const VkImageBlit *pRegions) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    if (pRegions != nullptr) {
        if ((pRegions->srcSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                    VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    UNRECOGNIZED_VALUE, LayerName,
                    "vkCmdBlitImage() parameter, VkImageAspect pRegions->srcSubresource.aspectMask, is an unrecognized enumerator");
            return false;
        }
        if ((pRegions->dstSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                    VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    UNRECOGNIZED_VALUE, LayerName,
                    "vkCmdBlitImage() parameter, VkImageAspect pRegions->dstSubresource.aspectMask, is an unrecognized enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                        VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                        const VkImageBlit *pRegions, VkFilter filter) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBlitImage(my_data, srcImage, srcImageLayout, dstImage, dstImageLayout,
                                                regionCount, pRegions, filter);

    if (!skip) {
        PreCmdBlitImage(commandBuffer, pRegions);

        my_data->dispatch_table.CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount,
                                             pRegions, filter);
    }
}

static bool PreCmdCopyBufferToImage(VkCommandBuffer commandBuffer, const VkBufferImageCopy *pRegions) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    if (pRegions != nullptr) {
        if ((pRegions->imageSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                      VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    UNRECOGNIZED_VALUE, LayerName,
                    "vkCmdCopyBufferToImage() parameter, VkImageAspect pRegions->imageSubresource.aspectMask, is an unrecognized "
                    "enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
                                                VkImageLayout dstImageLayout, uint32_t regionCount,
                                                const VkBufferImageCopy *pRegions) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdCopyBufferToImage(my_data, srcBuffer, dstImage, dstImageLayout, regionCount,
                                                        pRegions);

    if (!skip) {
        PreCmdCopyBufferToImage(commandBuffer, pRegions);

        my_data->dispatch_table.CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
    }
}

static bool PreCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, const VkBufferImageCopy *pRegions) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    if (pRegions != nullptr) {
        if ((pRegions->imageSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                      VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    UNRECOGNIZED_VALUE, LayerName,
                    "vkCmdCopyImageToBuffer parameter, VkImageAspect pRegions->imageSubresource.aspectMask, is an unrecognized "
                    "enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                                VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy *pRegions) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdCopyImageToBuffer(my_data, srcImage, srcImageLayout, dstBuffer, regionCount,
                                                        pRegions);

    if (!skip) {
        PreCmdCopyImageToBuffer(commandBuffer, pRegions);

        my_data->dispatch_table.CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                           VkDeviceSize dataSize, const void *pData) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdUpdateBuffer(my_data, dstBuffer, dstOffset, dataSize, pData);

    if (dstOffset & 3) {
        skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1e400048, LayerName,
                        "vkCmdUpdateBuffer() parameter, VkDeviceSize dstOffset (0x%" PRIxLEAST64 "), is not a multiple of 4. %s",
                        dstOffset, validation_error_map[VALIDATION_ERROR_1e400048]);
    }

    if ((dataSize <= 0) || (dataSize > 65536)) {
        skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1e40004a, LayerName,
                        "vkCmdUpdateBuffer() parameter, VkDeviceSize dataSize (0x%" PRIxLEAST64
                        "), must be greater than zero and less than or equal to 65536. %s",
                        dataSize, validation_error_map[VALIDATION_ERROR_1e40004a]);
    } else if (dataSize & 3) {
        skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1e40004c, LayerName,
                        "vkCmdUpdateBuffer() parameter, VkDeviceSize dataSize (0x%" PRIxLEAST64 "), is not a multiple of 4. %s",
                        dataSize, validation_error_map[VALIDATION_ERROR_1e40004c]);
    }

    if (!skip) {
        my_data->dispatch_table.CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                         VkDeviceSize size, uint32_t data) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdFillBuffer(my_data, dstBuffer, dstOffset, size, data);

    if (dstOffset & 3) {
        skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1b400032, LayerName,
                        "vkCmdFillBuffer() parameter, VkDeviceSize dstOffset (0x%" PRIxLEAST64 "), is not a multiple of 4. %s",
                        dstOffset, validation_error_map[VALIDATION_ERROR_1b400032]);
    }

    if (size != VK_WHOLE_SIZE) {
        if (size <= 0) {
            skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                            __LINE__, VALIDATION_ERROR_1b400034, LayerName,
                            "vkCmdFillBuffer() parameter, VkDeviceSize size (0x%" PRIxLEAST64 "), must be greater than zero. %s",
                            size, validation_error_map[VALIDATION_ERROR_1b400034]);
        } else if (size & 3) {
            skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                            __LINE__, VALIDATION_ERROR_1b400038, LayerName,
                            "vkCmdFillBuffer() parameter, VkDeviceSize size (0x%" PRIxLEAST64 "), is not a multiple of 4. %s", size,
                            validation_error_map[VALIDATION_ERROR_1b400038]);
        }
    }

    if (!skip) {
        my_data->dispatch_table.CmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout,
                                              const VkClearColorValue *pColor, uint32_t rangeCount,
                                              const VkImageSubresourceRange *pRanges) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdClearColorImage(my_data, image, imageLayout, pColor, rangeCount, pRanges);

    if (!skip) {
        my_data->dispatch_table.CmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout,
                                                     const VkClearDepthStencilValue *pDepthStencil, uint32_t rangeCount,
                                                     const VkImageSubresourceRange *pRanges) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdClearDepthStencilImage(my_data, image, imageLayout, pDepthStencil, rangeCount,
                                                             pRanges);

    if (!skip) {
        my_data->dispatch_table.CmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                               const VkClearAttachment *pAttachments, uint32_t rectCount,
                                               const VkClearRect *pRects) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdClearAttachments(my_data, attachmentCount, pAttachments, rectCount, pRects);

    if (!skip) {
        my_data->dispatch_table.CmdClearAttachments(commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
    }
}

static bool PreCmdResolveImage(VkCommandBuffer commandBuffer, const VkImageResolve *pRegions) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    if (pRegions != nullptr) {
        if ((pRegions->srcSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                    VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(
                my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                UNRECOGNIZED_VALUE, LayerName,
                "vkCmdResolveImage parameter, VkImageAspect pRegions->srcSubresource.aspectMask, is an unrecognized enumerator");
            return false;
        }
        if ((pRegions->dstSubresource.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                    VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
            log_msg(
                my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                UNRECOGNIZED_VALUE, LayerName,
                "vkCmdResolveImage parameter, VkImageAspect pRegions->dstSubresource.aspectMask, is an unrecognized enumerator");
            return false;
        }
    }

    return true;
}

VKAPI_ATTR void VKAPI_CALL CmdResolveImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
                                           VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount,
                                           const VkImageResolve *pRegions) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdResolveImage(my_data, srcImage, srcImageLayout, dstImage, dstImageLayout,
                                                   regionCount, pRegions);

    if (!skip) {
        PreCmdResolveImage(commandBuffer, pRegions);

        my_data->dispatch_table.CmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount,
                                                pRegions);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetEvent(my_data, event, stageMask);

    if (!skip) {
        my_data->dispatch_table.CmdSetEvent(commandBuffer, event, stageMask);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdResetEvent(my_data, event, stageMask);

    if (!skip) {
        my_data->dispatch_table.CmdResetEvent(commandBuffer, event, stageMask);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
                                         VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
                                         uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                         uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                         uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier *pImageMemoryBarriers) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdWaitEvents(my_data, eventCount, pEvents, srcStageMask, dstStageMask,
                                                 memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                                                 pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

    if (!skip) {
        my_data->dispatch_table.CmdWaitEvents(commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount,
                                              pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers,
                                              imageMemoryBarrierCount, pImageMemoryBarriers);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
                                              VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
                                              uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                              uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                              uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier *pImageMemoryBarriers) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdPipelineBarrier(my_data, srcStageMask, dstStageMask, dependencyFlags,
                                                      memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount,
                                                      pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

    if (!skip) {
        my_data->dispatch_table.CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount,
                                                   pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers,
                                                   imageMemoryBarrierCount, pImageMemoryBarriers);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t slot,
                                         VkQueryControlFlags flags) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBeginQuery(my_data, queryPool, slot, flags);

    if (!skip) {
        my_data->dispatch_table.CmdBeginQuery(commandBuffer, queryPool, slot, flags);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t slot) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdEndQuery(my_data, queryPool, slot);

    if (!skip) {
        my_data->dispatch_table.CmdEndQuery(commandBuffer, queryPool, slot);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery,
                                             uint32_t queryCount) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdResetQueryPool(my_data, queryPool, firstQuery, queryCount);

    if (!skip) {
        my_data->dispatch_table.CmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage,
                                             VkQueryPool queryPool, uint32_t query) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdWriteTimestamp(my_data, pipelineStage, queryPool, query);

    if (!skip) {
        my_data->dispatch_table.CmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, query);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery,
                                                   uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                                   VkDeviceSize stride, VkQueryResultFlags flags) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdCopyQueryPoolResults(my_data, queryPool, firstQuery, queryCount, dstBuffer,
                                                           dstOffset, stride, flags);

    if (!skip) {
        my_data->dispatch_table.CmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset,
                                                        stride, flags);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags,
                                            uint32_t offset, uint32_t size, const void *pValues) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdPushConstants(my_data, layout, stageFlags, offset, size, pValues);

    if (!skip) {
        my_data->dispatch_table.CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo *pRenderPassBegin,
                                              VkSubpassContents contents) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdBeginRenderPass(my_data, pRenderPassBegin, contents);

    if (!skip) {
        my_data->dispatch_table.CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdNextSubpass(my_data, contents);

    if (!skip) {
        my_data->dispatch_table.CmdNextSubpass(commandBuffer, contents);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(VkCommandBuffer commandBuffer) {
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    my_data->dispatch_table.CmdEndRenderPass(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount,
                                              const VkCommandBuffer *pCommandBuffers) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdExecuteCommands(my_data, commandBufferCount, pCommandBuffers);

    if (!skip) {
        my_data->dispatch_table.CmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(uint32_t *pCount, VkLayerProperties *pProperties) {
    return util_GetLayerProperties(1, &global_layer, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                              VkLayerProperties *pProperties) {
    return util_GetLayerProperties(1, &global_layer, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                    VkExtensionProperties *pProperties) {
    if (pLayerName && !strcmp(pLayerName, global_layer.layerName))
        return util_GetExtensionProperties(1, instance_extensions, pCount, pProperties);

    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                                  uint32_t *pCount, VkExtensionProperties *pProperties) {
    /* parameter_validation does not have any physical device extensions */
    if (pLayerName && !strcmp(pLayerName, global_layer.layerName)) return util_GetExtensionProperties(0, NULL, pCount, pProperties);

    assert(physicalDevice);

    return GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map)
        ->dispatch_table.EnumerateDeviceExtensionProperties(physicalDevice, NULL, pCount, pProperties);
}

static bool require_device_extension(layer_data *my_data, bool flag, char const *function_name, char const *extension_name) {
    if (!flag) {
        return log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                       EXTENSION_NOT_ENABLED, LayerName,
                       "%s() called even though the %s extension was not enabled for this VkDevice.", function_name,
                       extension_name);
    }

    return false;
}

// WSI Extension Functions

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *pCreateInfo,
                                                  const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchain) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(device_data != nullptr);
    std::unique_lock<std::mutex> lock(global_lock);
    debug_report_data *report_data = device_data->report_data;

    skip |= parameter_validation_vkCreateSwapchainKHR(device_data, pCreateInfo, pAllocator, pSwapchain);

    if (pCreateInfo != nullptr) {
        if ((device_data->physical_device_features.textureCompressionETC2 == false) &&
            FormatIsCompressed_ETC2_EAC(pCreateInfo->imageFormat)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCreateSwapchainKHR(): Attempting to create swapchain VkImage with format %s. The "
                            "textureCompressionETC2 feature is not enabled: neither ETC2 nor EAC formats can be used to create "
                            "images.",
                            string_VkFormat(pCreateInfo->imageFormat));
        }

        if ((device_data->physical_device_features.textureCompressionASTC_LDR == false) &&
            FormatIsCompressed_ASTC_LDR(pCreateInfo->imageFormat)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCreateSwapchainKHR(): Attempting to create swapchain VkImage with format %s. The "
                            "textureCompressionASTC_LDR feature is not enabled: ASTC formats cannot be used to create images.",
                            string_VkFormat(pCreateInfo->imageFormat));
        }

        if ((device_data->physical_device_features.textureCompressionBC == false) &&
            FormatIsCompressed_BC(pCreateInfo->imageFormat)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            DEVICE_FEATURE, LayerName,
                            "vkCreateSwapchainKHR(): Attempting to create swapchain VkImage with format %s. The "
                            "textureCompressionBC feature is not enabled: BC compressed formats cannot be used to create images.",
                            string_VkFormat(pCreateInfo->imageFormat));
        }

        // Validation for parameters excluded from the generated validation code due to a 'noautovalidity' tag in vk.xml
        if (pCreateInfo->imageSharingMode == VK_SHARING_MODE_CONCURRENT) {
            // If imageSharingMode is VK_SHARING_MODE_CONCURRENT, queueFamilyIndexCount must be greater than 1
            if (pCreateInfo->queueFamilyIndexCount <= 1) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_146009fc, LayerName,
                                "vkCreateSwapchainKHR(): if pCreateInfo->imageSharingMode is VK_SHARING_MODE_CONCURRENT, "
                                "pCreateInfo->queueFamilyIndexCount must be greater than 1. %s",
                                validation_error_map[VALIDATION_ERROR_146009fc]);
            }

            // If imageSharingMode is VK_SHARING_MODE_CONCURRENT, pQueueFamilyIndices must be a pointer to an array of
            // queueFamilyIndexCount uint32_t values
            if (pCreateInfo->pQueueFamilyIndices == nullptr) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_146009fa, LayerName,
                                "vkCreateSwapchainKHR(): if pCreateInfo->imageSharingMode is VK_SHARING_MODE_CONCURRENT, "
                                "pCreateInfo->pQueueFamilyIndices must be a pointer to an array of "
                                "pCreateInfo->queueFamilyIndexCount uint32_t values. %s",
                                validation_error_map[VALIDATION_ERROR_146009fa]);
            } else {
                // TODO: Not in the spec VUs. Probably missing -- KhronosGroup/Vulkan-Docs#501. Update error codes when resolved.
                skip |= ValidateQueueFamilies(device_data, pCreateInfo->queueFamilyIndexCount, pCreateInfo->pQueueFamilyIndices,
                                              "vkCreateSwapchainKHR", "pCreateInfo->pQueueFamilyIndices", INVALID_USAGE,
                                              INVALID_USAGE, false, "", "");
            }
        }

        // imageArrayLayers must be greater than 0
        skip |= ValidateGreaterThan(report_data, "vkCreateSwapchainKHR", "pCreateInfo->imageArrayLayers",
                                    pCreateInfo->imageArrayLayers, 0u);
    }

    lock.unlock();

    if (!skip) {
        result = device_data->dispatch_table.CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);

        validate_result(report_data, "vkCreateSwapchainKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t *pSwapchainImageCount,
                                                     VkImage *pSwapchainImages) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetSwapchainImagesKHR(my_data, swapchain, pSwapchainImageCount, pSwapchainImages);

    if (!skip) {
        result = my_data->dispatch_table.GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);

        validate_result(my_data->report_data, "vkGetSwapchainImagesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
                                                   VkSemaphore semaphore, VkFence fence, uint32_t *pImageIndex) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkAcquireNextImageKHR(my_data, swapchain, timeout, semaphore, fence, pImageIndex);

    if (!skip) {
        result = my_data->dispatch_table.AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);

        validate_result(my_data->report_data, "vkAcquireNextImageKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkQueuePresentKHR(my_data, pPresentInfo);

    if (pPresentInfo && pPresentInfo->pNext) {
        // Verify ext struct
        struct std_header {
            VkStructureType sType;
            const void *pNext;
        };
        std_header *pnext = (std_header *)pPresentInfo->pNext;
        while (pnext) {
            if (VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR == pnext->sType) {
                // TODO: This and all other pNext extension dependencies should be added to code-generation
                skip |= require_device_extension(my_data, my_data->extensions.vk_khr_incremental_present, "vkQueuePresentKHR",
                                                 VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME);
                VkPresentRegionsKHR *present_regions = (VkPresentRegionsKHR *)pnext;
                if (present_regions->swapchainCount != pPresentInfo->swapchainCount) {
                    skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, INVALID_USAGE, LayerName,
                                    "QueuePresentKHR(): pPresentInfo->swapchainCount has a value of %i"
                                    " but VkPresentRegionsKHR extension swapchainCount is %i. These values must be equal.",
                                    pPresentInfo->swapchainCount, present_regions->swapchainCount);
                }
                skip |= validate_struct_pnext(my_data->report_data, "QueuePresentKHR", "pCreateInfo->pNext->pNext", NULL,
                                              present_regions->pNext, 0, NULL, GeneratedHeaderVersion, VALIDATION_ERROR_1121c40d);
                skip |= validate_array(my_data->report_data, "QueuePresentKHR", "pCreateInfo->pNext->swapchainCount",
                                       "pCreateInfo->pNext->pRegions", present_regions->swapchainCount, present_regions->pRegions,
                                       true, false, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
                for (uint32_t i = 0; i < present_regions->swapchainCount; ++i) {
                    skip |= validate_array(my_data->report_data, "QueuePresentKHR", "pCreateInfo->pNext->pRegions[].rectangleCount",
                                           "pCreateInfo->pNext->pRegions[].pRectangles",
                                           present_regions->pRegions[i].rectangleCount, present_regions->pRegions[i].pRectangles,
                                           true, false, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
                }
            }
            pnext = (std_header *)pnext->pNext;
        }
    }

    if (!skip) {
        result = my_data->dispatch_table.QueuePresentKHR(queue, pPresentInfo);

        validate_result(my_data->report_data, "vkQueuePresentKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    layer_data *my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    /* No generated validation function for this call */

    if (!skip) {
        my_data->dispatch_table.DestroySwapchainKHR(device, swapchain, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
                                                                  VkSurfaceKHR surface, VkBool32 *pSupported) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceSurfaceSupportKHR(my_data, queueFamilyIndex, surface, pSupported);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);

        validate_result(my_data->report_data, "vkGetPhysicalDeviceSurfaceSupportKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                                                                       VkSurfaceCapabilitiesKHR *pSurfaceCapabilities) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(my_data, surface, pSurfaceCapabilities);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);

        validate_result(my_data->report_data, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                                                                  uint32_t *pSurfaceFormatCount,
                                                                  VkSurfaceFormatKHR *pSurfaceFormats) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceSurfaceFormatsKHR(my_data, surface, pSurfaceFormatCount,
                                                                      pSurfaceFormats);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount,
                                                                            pSurfaceFormats);

        validate_result(my_data->report_data, "vkGetPhysicalDeviceSurfaceFormatsKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                                                                       uint32_t *pPresentModeCount,
                                                                       VkPresentModeKHR *pPresentModes) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceSurfacePresentModesKHR(my_data, surface, pPresentModeCount,
                                                                           pPresentModes);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount,
                                                                                 pPresentModes);

        validate_result(my_data->report_data, "vkGetPhysicalDeviceSurfacePresentModesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySurfaceKHR(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);

    if (!skip) {
        my_data->dispatch_table.DestroySurfaceKHR(instance, surface, pAllocator);
    }
}

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL CreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR *pCreateInfo,
                                                     const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    if (pCreateInfo->hwnd == nullptr) {
        skip |= log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_15a00a38, LayerName,
                        "vkCreateWin32SurfaceKHR(): hwnd must be a valid Win32 HWND but hwnd is NULL. %s",
                        validation_error_map[VALIDATION_ERROR_15a00a38]);
    }

    skip |= parameter_validation_vkCreateWin32SurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }

    validate_result(my_data->report_data, "vkCreateWin32SurfaceKHR", {}, result);

    return result;
}

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceWin32PresentationSupportKHR(VkPhysicalDevice physicalDevice,
                                                                            uint32_t queueFamilyIndex) {
    VkBool32 result = false;

    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    // TODO: codegen doesn't produce this function?
    // skip |= parameter_validation_vkGetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, queueFamilyIndex);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, queueFamilyIndex);
    }

    return result;
}
#endif  // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_XCB_KHR
VKAPI_ATTR VkResult VKAPI_CALL CreateXcbSurfaceKHR(VkInstance instance, const VkXcbSurfaceCreateInfoKHR *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkCreateXcbSurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateXcbSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }

    validate_result(my_data->report_data, "vkCreateXcbSurfaceKHR", {}, result);

    return result;
}

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceXcbPresentationSupportKHR(VkPhysicalDevice physicalDevice,
                                                                          uint32_t queueFamilyIndex, xcb_connection_t *connection,
                                                                          xcb_visualid_t visual_id) {
    VkBool32 result = false;

    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkGetPhysicalDeviceXcbPresentationSupportKHR(my_data, queueFamilyIndex, connection,
                                                                              visual_id);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceXcbPresentationSupportKHR(physicalDevice, queueFamilyIndex, connection,
                                                                                    visual_id);
    }

    return result;
}
#endif  // VK_USE_PLATFORM_XCB_KHR

#ifdef VK_USE_PLATFORM_XLIB_KHR
VKAPI_ATTR VkResult VKAPI_CALL CreateXlibSurfaceKHR(VkInstance instance, const VkXlibSurfaceCreateInfoKHR *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkCreateXlibSurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateXlibSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }

    validate_result(my_data->report_data, "vkCreateXlibSurfaceKHR", {}, result);

    return result;
}

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceXlibPresentationSupportKHR(VkPhysicalDevice physicalDevice,
                                                                           uint32_t queueFamilyIndex, Display *dpy,
                                                                           VisualID visualID) {
    VkBool32 result = false;

    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |=
        parameter_validation_vkGetPhysicalDeviceXlibPresentationSupportKHR(my_data, queueFamilyIndex, dpy, visualID);

    if (!skip) {
        result =
            my_data->dispatch_table.GetPhysicalDeviceXlibPresentationSupportKHR(physicalDevice, queueFamilyIndex, dpy, visualID);
    }
    return result;
}
#endif  // VK_USE_PLATFORM_XLIB_KHR

#ifdef VK_USE_PLATFORM_MIR_KHR
VKAPI_ATTR VkResult VKAPI_CALL CreateMirSurfaceKHR(VkInstance instance, const VkMirSurfaceCreateInfoKHR *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkCreateMirSurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateMirSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }

    validate_result(my_data->report_data, "vkCreateMirSurfaceKHR", {}, result);

    return result;
}

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceMirPresentationSupportKHR(VkPhysicalDevice physicalDevice,
                                                                          uint32_t queueFamilyIndex, MirConnection *connection) {
    VkBool32 result = false;

    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    bool skip = false;

    skip |= parameter_validation_vkGetPhysicalDeviceMirPresentationSupportKHR(my_data, queueFamilyIndex, connection);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceMirPresentationSupportKHR(physicalDevice, queueFamilyIndex, connection);
    }
    return result;
}
#endif  // VK_USE_PLATFORM_MIR_KHR

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
VKAPI_ATTR VkResult VKAPI_CALL CreateWaylandSurfaceKHR(VkInstance instance, const VkWaylandSurfaceCreateInfoKHR *pCreateInfo,
                                                       const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkCreateWaylandSurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateWaylandSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }

    validate_result(my_data->report_data, "vkCreateWaylandSurfaceKHR", {}, result);

    return result;
}

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceWaylandPresentationSupportKHR(VkPhysicalDevice physicalDevice,
                                                                              uint32_t queueFamilyIndex,
                                                                              struct wl_display *display) {
    VkBool32 result = false;

    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkGetPhysicalDeviceWaylandPresentationSupportKHR(my_data, queueFamilyIndex, display);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceWaylandPresentationSupportKHR(physicalDevice, queueFamilyIndex, display);
    }

    return result;
}
#endif  // VK_USE_PLATFORM_WAYLAND_KHR

#ifdef VK_USE_PLATFORM_ANDROID_KHR
VKAPI_ATTR VkResult VKAPI_CALL CreateAndroidSurfaceKHR(VkInstance instance, const VkAndroidSurfaceCreateInfoKHR *pCreateInfo,
                                                       const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkCreateAndroidSurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateAndroidSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    }

    validate_result(my_data->report_data, "vkCreateAndroidSurfaceKHR", {}, result);

    return result;
}
#endif  // VK_USE_PLATFORM_ANDROID_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateSharedSwapchainsKHR(VkDevice device, uint32_t swapchainCount,
                                                         const VkSwapchainCreateInfoKHR *pCreateInfos,
                                                         const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchains) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateSharedSwapchainsKHR(my_data, swapchainCount, pCreateInfos, pAllocator,
                                                             pSwapchains);

    if (!skip) {
        result = my_data->dispatch_table.CreateSharedSwapchainsKHR(device, swapchainCount, pCreateInfos, pAllocator, pSwapchains);

        validate_result(my_data->report_data, "vkCreateSharedSwapchainsKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceDisplayPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                                     VkDisplayPropertiesKHR *pProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceDisplayPropertiesKHR(my_data, pPropertyCount, pProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, pPropertyCount, pProperties);

        validate_result(my_data->report_data, "vkGetPhysicalDeviceDisplayPropertiesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceDisplayPlanePropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                                          VkDisplayPlanePropertiesKHR *pProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(my_data, pPropertyCount, pProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, pPropertyCount, pProperties);

        validate_result(my_data->report_data, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneSupportedDisplaysKHR(VkPhysicalDevice physicalDevice, uint32_t planeIndex,
                                                                   uint32_t *pDisplayCount, VkDisplayKHR *pDisplays) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDisplayPlaneSupportedDisplaysKHR(my_data, planeIndex, pDisplayCount, pDisplays);

    if (!skip) {
        result = my_data->dispatch_table.GetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, pDisplayCount, pDisplays);

        validate_result(my_data->report_data, "vkGetDisplayPlaneSupportedDisplaysKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayModePropertiesKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                                                           uint32_t *pPropertyCount, VkDisplayModePropertiesKHR *pProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDisplayModePropertiesKHR(my_data, display, pPropertyCount, pProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetDisplayModePropertiesKHR(physicalDevice, display, pPropertyCount, pProperties);

        validate_result(my_data->report_data, "vkGetDisplayModePropertiesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayModeKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                                                    const VkDisplayModeCreateInfoKHR *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkDisplayModeKHR *pMode) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateDisplayModeKHR(my_data, display, pCreateInfo, pAllocator, pMode);

    if (!skip) {
        result = my_data->dispatch_table.CreateDisplayModeKHR(physicalDevice, display, pCreateInfo, pAllocator, pMode);

        validate_result(my_data->report_data, "vkCreateDisplayModeKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode,
                                                              uint32_t planeIndex, VkDisplayPlaneCapabilitiesKHR *pCapabilities) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDisplayPlaneCapabilitiesKHR(my_data, mode, planeIndex, pCapabilities);

    if (!skip) {
        result = my_data->dispatch_table.GetDisplayPlaneCapabilitiesKHR(physicalDevice, mode, planeIndex, pCapabilities);

        validate_result(my_data->report_data, "vkGetDisplayPlaneCapabilitiesKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayPlaneSurfaceKHR(VkInstance instance, const VkDisplaySurfaceCreateInfoKHR *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateDisplayPlaneSurfaceKHR(my_data, pCreateInfo, pAllocator, pSurface);

    if (!skip) {
        result = my_data->dispatch_table.CreateDisplayPlaneSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);

        validate_result(my_data->report_data, "vkCreateDisplayPlaneSurfaceKHR", {}, result);
    }

    return result;
}

// Definitions for the VK_KHR_descriptor_update_template extension

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorUpdateTemplateKHR(VkDevice device,
                                                                 const VkDescriptorUpdateTemplateCreateInfoKHR *pCreateInfo,
                                                                 const VkAllocationCallbacks *pAllocator,
                                                                 VkDescriptorUpdateTemplateKHR *pDescriptorUpdateTemplate) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateDescriptorUpdateTemplateKHR(my_data, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);

    if (!skip) {
        result =
            my_data->dispatch_table.CreateDescriptorUpdateTemplateKHR(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
        validate_result(my_data->report_data, "vkCreateDescriptorUpdateTemplateKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorUpdateTemplateKHR(VkDevice device,
                                                              VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate,
                                                              const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyDescriptorUpdateTemplateKHR(my_data, descriptorUpdateTemplate, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyDescriptorUpdateTemplateKHR(device, descriptorUpdateTemplate, pAllocator);
    }
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSetWithTemplateKHR(VkDevice device, VkDescriptorSet descriptorSet,
                                                              VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate,
                                                              const void *pData) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkUpdateDescriptorSetWithTemplateKHR(my_data, descriptorSet, descriptorUpdateTemplate, pData);

    if (!skip) {
        my_data->dispatch_table.UpdateDescriptorSetWithTemplateKHR(device, descriptorSet, descriptorUpdateTemplate, pData);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer commandBuffer,
                                                               VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate,
                                                               VkPipelineLayout layout, uint32_t set, const void *pData) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdPushDescriptorSetWithTemplateKHR(my_data, descriptorUpdateTemplate, layout, set, pData);

    if (!skip) {
        my_data->dispatch_table.CmdPushDescriptorSetWithTemplateKHR(commandBuffer, descriptorUpdateTemplate, layout, set, pData);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainStatusKHR(VkDevice device, VkSwapchainKHR swapchain) {
    bool skip = false;
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    skip = parameter_validation_vkGetSwapchainStatusKHR(dev_data, swapchain);

    if (!skip) {
        result = dev_data->dispatch_table.GetSwapchainStatusKHR(device, swapchain);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2KHR(VkPhysicalDevice physicalDevice,
                                                                        const VkPhysicalDeviceSurfaceInfo2KHR *pSurfaceInfo,
                                                                        VkSurfaceCapabilities2KHR *pSurfaceCapabilities) {
    bool skip = false;
    instance_layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;

    skip = parameter_validation_vkGetPhysicalDeviceSurfaceCapabilities2KHR(instance_data, pSurfaceInfo, pSurfaceCapabilities);

    if (!skip) {
        result = instance_data->dispatch_table.GetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice, pSurfaceInfo,
                                                                                        pSurfaceCapabilities);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormats2KHR(VkPhysicalDevice physicalDevice,
                                                                   const VkPhysicalDeviceSurfaceInfo2KHR *pSurfaceInfo,
                                                                   uint32_t *pSurfaceFormatCount,
                                                                   VkSurfaceFormat2KHR *pSurfaceFormats) {
    bool skip = false;
    instance_layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    skip = parameter_validation_vkGetPhysicalDeviceSurfaceFormats2KHR(instance_data, pSurfaceInfo, pSurfaceFormatCount,
                                                                      pSurfaceFormats);
    if (!skip) {
        result = instance_data->dispatch_table.GetPhysicalDeviceSurfaceFormats2KHR(physicalDevice, pSurfaceInfo,
                                                                                   pSurfaceFormatCount, pSurfaceFormats);
    }
    return result;
}

// Definitions for the VK_KHR_get_physical_device_properties2 extension

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2KHR *pFeatures) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceFeatures2KHR(my_data, pFeatures);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceFeatures2KHR(physicalDevice, pFeatures);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties2KHR(VkPhysicalDevice physicalDevice,
                                                           VkPhysicalDeviceProperties2KHR *pProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceProperties2KHR(my_data, pProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceProperties2KHR(physicalDevice, pProperties);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties2KHR(VkPhysicalDevice physicalDevice, VkFormat format,
                                                                 VkFormatProperties2KHR *pFormatProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceFormatProperties2KHR(my_data, format, pFormatProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceFormatProperties2KHR(physicalDevice, format, pFormatProperties);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceImageFormatProperties2KHR(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2KHR *pImageFormatInfo,
    VkImageFormatProperties2KHR *pImageFormatProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceImageFormatProperties2KHR(my_data, pImageFormatInfo,
                                                                              pImageFormatProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceImageFormatProperties2KHR(physicalDevice, pImageFormatInfo,
                                                                                    pImageFormatProperties);
        const std::vector<VkResult> ignore_list = {VK_ERROR_FORMAT_NOT_SUPPORTED};
        validate_result(my_data->report_data, "vkGetPhysicalDeviceImageFormatProperties2KHR", ignore_list, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties2KHR(VkPhysicalDevice physicalDevice,
                                                                      uint32_t *pQueueFamilyPropertyCount,
                                                                      VkQueueFamilyProperties2KHR *pQueueFamilyProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceQueueFamilyProperties2KHR(my_data, pQueueFamilyPropertyCount,
                                                                              pQueueFamilyProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice, pQueueFamilyPropertyCount,
                                                                           pQueueFamilyProperties);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties2KHR(VkPhysicalDevice physicalDevice,
                                                                 VkPhysicalDeviceMemoryProperties2KHR *pMemoryProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceMemoryProperties2KHR(my_data, pMemoryProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceMemoryProperties2KHR(physicalDevice, pMemoryProperties);
    }
}

static bool PostGetPhysicalDeviceSparseImageFormatProperties2KHR(VkPhysicalDevice physicalDevice,
                                                                 const VkPhysicalDeviceSparseImageFormatInfo2KHR *pFormatInfo,
                                                                 uint32_t *pPropertyCount,
                                                                 VkSparseImageFormatProperties2KHR *pProperties) {
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    if (pProperties != nullptr) {
        for (uint32_t i = 0; i < *pPropertyCount; ++i) {
            if ((pProperties[i].properties.aspectMask & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT |
                                                         VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT)) == 0) {
                log_msg(my_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        1, LayerName,
                        "vkGetPhysicalDeviceSparseImageFormatProperties2KHR parameter, VkImageAspect "
                        "pProperties[%i].properties.aspectMask, is an "
                        "unrecognized enumerator",
                        i);
                return false;
            }
        }
    }
    return true;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceSparseImageFormatProperties2KHR(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2KHR *pFormatInfo, uint32_t *pPropertyCount,
    VkSparseImageFormatProperties2KHR *pProperties) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(my_data, pFormatInfo,
                                                                                    pPropertyCount, pProperties);

    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceSparseImageFormatProperties2KHR(physicalDevice, pFormatInfo, pPropertyCount,
                                                                                 pProperties);
        PostGetPhysicalDeviceSparseImageFormatProperties2KHR(physicalDevice, pFormatInfo, pPropertyCount, pProperties);
    }
}

// Definitions for the VK_KHR_external_fence_capabilities extension

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceExternalFencePropertiesKHR(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalFenceInfoKHR *pExternalFenceInfo,
    VkExternalFencePropertiesKHR *pExternalFenceProperties) {
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;
    skip |=
        parameter_validation_vkGetPhysicalDeviceExternalFencePropertiesKHR(my_data, pExternalFenceInfo, pExternalFenceProperties);
    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceExternalFencePropertiesKHR(physicalDevice, pExternalFenceInfo,
                                                                            pExternalFenceProperties);
    }
}

// Definitions for the VK_KHR_external_fence_fd extension

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceFdKHR(VkDevice device, const VkImportFenceFdInfoKHR *pImportFenceFdInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkImportFenceFdKHR(my_data, pImportFenceFdInfo);

    if (!skip) {
        result = my_data->dispatch_table.ImportFenceFdKHR(device, pImportFenceFdInfo);
        validate_result(my_data->report_data, "vkImportFenceFdKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceFdKHR(VkDevice device, const VkFenceGetFdInfoKHR *pGetFdInfo, int *pFd) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetFenceFdKHR(my_data, pGetFdInfo, pFd);

    if (!skip) {
        result = my_data->dispatch_table.GetFenceFdKHR(device, pGetFdInfo, pFd);
        validate_result(my_data->report_data, "vkGetFenceFdKHR", {}, result);
    }

    return result;
}

#ifdef VK_USE_PLATFORM_WIN32_KHR
// Definitions for the VK_KHR_external_fence_win32 extension

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceWin32HandleKHR(VkDevice device,
                                                         const VkImportFenceWin32HandleInfoKHR *pImportFenceWin32HandleInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkImportFenceWin32HandleKHR(my_data, pImportFenceWin32HandleInfo);

    if (!skip) {
        result = my_data->dispatch_table.ImportFenceWin32HandleKHR(device, pImportFenceWin32HandleInfo);
        validate_result(my_data->report_data, "vkImportFenceWin32HandleKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceWin32HandleKHR(VkDevice device, const VkFenceGetWin32HandleInfoKHR *pGetWin32HandleInfo,
                                                      HANDLE *pHandle) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetFenceWin32HandleKHR(my_data, pGetWin32HandleInfo, pHandle);

    if (!skip) {
        result = my_data->dispatch_table.GetFenceWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
        validate_result(my_data->report_data, "vkGetFenceWin32HandleKHR", {}, result);
    }

    return result;
}
#endif  // VK_USE_PLATFORM_WIN32_KHR

// Definitions for the VK_KHR_external_memory_capabilities extension

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceExternalBufferPropertiesKHR(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfoKHR *pExternalBufferInfo,
    VkExternalBufferPropertiesKHR *pExternalBufferProperties) {
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;
    skip |= parameter_validation_vkGetPhysicalDeviceExternalBufferPropertiesKHR(my_data, pExternalBufferInfo,
                                                                                pExternalBufferProperties);
    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceExternalBufferPropertiesKHR(physicalDevice, pExternalBufferInfo,
                                                                             pExternalBufferProperties);
    }
}

// Definitions for the VK_KHR_external_memory_fd extension

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR* pGetFdInfo, int *pFd) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetMemoryFdKHR(my_data, pGetFdInfo, pFd);

    if (!skip) {
        result = my_data->dispatch_table.GetMemoryFdKHR(device, pGetFdInfo, pFd);
        validate_result(my_data->report_data, "vkGetMemoryFdKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdPropertiesKHR(VkDevice device, VkExternalMemoryHandleTypeFlagBitsKHR handleType, int fd,
                                                        VkMemoryFdPropertiesKHR *pMemoryFdProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetMemoryFdPropertiesKHR(my_data, handleType, fd, pMemoryFdProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetMemoryFdPropertiesKHR(device, handleType, fd, pMemoryFdProperties);
        validate_result(my_data->report_data, "vkGetMemoryFdPropertiesKHR", {}, result);
    }

    return result;
}

// Definitions for the VK_KHR_external_memory_win32 extension

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleKHR(VkDevice device, const VkMemoryGetWin32HandleInfoKHR *pGetWin32HandleInfo,
                                                       HANDLE *pHandle) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetMemoryWin32HandleKHR(my_data, pGetWin32HandleInfo, pHandle);

    if (!skip) {
        result = my_data->dispatch_table.GetMemoryWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
        validate_result(my_data->report_data, "vkGetMemoryWin32HandleKHR", {}, result);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandlePropertiesKHR(VkDevice device, VkExternalMemoryHandleTypeFlagBitsKHR handleType,
                                                                 HANDLE handle,
                                                                 VkMemoryWin32HandlePropertiesKHR *pMemoryWin32HandleProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetMemoryWin32HandlePropertiesKHR(my_data, handleType, handle, pMemoryWin32HandleProperties);

    if (!skip) {
        result =
            my_data->dispatch_table.GetMemoryWin32HandlePropertiesKHR(device, handleType, handle, pMemoryWin32HandleProperties);
        validate_result(my_data->report_data, "vkGetMemoryWin32HandlePropertiesKHR", {}, result);
    }
    return result;
}
#endif  // VK_USE_PLATFORM_WIN32_KHR

// Definitions for the VK_KHR_external_semaphore_capabilities extension

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceExternalSemaphorePropertiesKHR(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalSemaphoreInfoKHR *pExternalSemaphoreInfo,
    VkExternalSemaphorePropertiesKHR *pExternalSemaphoreProperties) {
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;
    skip |= parameter_validation_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(my_data, pExternalSemaphoreInfo,
                                                                                   pExternalSemaphoreProperties);
    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceExternalSemaphorePropertiesKHR(physicalDevice, pExternalSemaphoreInfo,
                                                                                pExternalSemaphoreProperties);
    }
}

// Definitions for the VK_KHR_external_semaphore_fd extension

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreFdKHR(VkDevice device, const VkImportSemaphoreFdInfoKHR *pImportSemaphoreFdInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkImportSemaphoreFdKHR(my_data, pImportSemaphoreFdInfo);

    if (!skip) {
        result = my_data->dispatch_table.ImportSemaphoreFdKHR(device, pImportSemaphoreFdInfo);
        validate_result(my_data->report_data, "vkImportSemaphoreFdKHR", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreFdKHR(VkDevice device, const VkSemaphoreGetFdInfoKHR* pGetFdInfo, int *pFd) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetSemaphoreFdKHR(my_data, pGetFdInfo, pFd);

    if (!skip) {
        result = my_data->dispatch_table.GetSemaphoreFdKHR(device, pGetFdInfo, pFd);
        validate_result(my_data->report_data, "vkGetSemaphoreFdKHR", {}, result);
    }

    return result;
}

// Definitions for the VK_KHR_external_semaphore_win32 extension

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL
ImportSemaphoreWin32HandleKHR(VkDevice device, const VkImportSemaphoreWin32HandleInfoKHR *pImportSemaphoreWin32HandleInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkImportSemaphoreWin32HandleKHR(my_data, pImportSemaphoreWin32HandleInfo);
    if (!skip) {
        result = my_data->dispatch_table.ImportSemaphoreWin32HandleKHR(device, pImportSemaphoreWin32HandleInfo);
        validate_result(my_data->report_data, "vkImportSemaphoreWin32HandleKHR", {}, result);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreWin32HandleKHR(VkDevice device,
                                                          const VkSemaphoreGetWin32HandleInfoKHR *pGetWin32HandleInfo,
                                                          HANDLE *pHandle) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetSemaphoreWin32HandleKHR(my_data, pGetWin32HandleInfo, pHandle);
    if (!skip) {
        result = my_data->dispatch_table.GetSemaphoreWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
        validate_result(my_data->report_data, "vkGetSemaphoreWin32HandleKHR", {}, result);
    }
    return result;
}
#endif  // VK_USE_PLATFORM_WIN32_KHR

// Definitions for the VK_KHR_get_memory_requirements2 extension

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements2KHR(VkDevice device, const VkImageMemoryRequirementsInfo2KHR *pInfo,
                                                          VkMemoryRequirements2KHR *pMemoryRequirements) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetImageMemoryRequirements2KHR(my_data, pInfo, pMemoryRequirements);

    if (!skip) {
        my_data->dispatch_table.GetImageMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);
    }
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements2KHR(VkDevice device, const VkBufferMemoryRequirementsInfo2KHR *pInfo,
                                                           VkMemoryRequirements2KHR *pMemoryRequirements) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetBufferMemoryRequirements2KHR(my_data, pInfo, pMemoryRequirements);

    if (!skip) {
        my_data->dispatch_table.GetBufferMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements2KHR(VkDevice device,
                                                                const VkImageSparseMemoryRequirementsInfo2KHR *pInfo,
                                                                uint32_t *pSparseMemoryRequirementCount,
                                                                VkSparseImageMemoryRequirements2KHR *pSparseMemoryRequirements) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetImageSparseMemoryRequirements2KHR(my_data, pInfo, pSparseMemoryRequirementCount,
                                                                        pSparseMemoryRequirements);

    if (!skip) {
        my_data->dispatch_table.GetImageSparseMemoryRequirements2KHR(device, pInfo, pSparseMemoryRequirementCount,
                                                                     pSparseMemoryRequirements);
    }
}

// Definitions for the VK_KHR_maintenance1 extension

VKAPI_ATTR void VKAPI_CALL TrimCommandPoolKHR(VkDevice device, VkCommandPool commandPool, VkCommandPoolTrimFlagsKHR flags) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkTrimCommandPoolKHR(my_data, commandPool, flags);

    if (!skip) {
        my_data->dispatch_table.TrimCommandPoolKHR(device, commandPool, flags);
    }
}

// Definitions for the VK_KHR_push_descriptor extension

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                   VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                                   const VkWriteDescriptorSet *pDescriptorWrites) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdPushDescriptorSetKHR(my_data, pipelineBindPoint, layout, set, descriptorWriteCount,
                                                           pDescriptorWrites);

    if (!skip) {
        my_data->dispatch_table.CmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount,
                                                        pDescriptorWrites);
    }
}

// Definitions for the VK_KHX_device_group_creation extension

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDeviceGroupsKHX(
    VkInstance instance, uint32_t *pPhysicalDeviceGroupCount, VkPhysicalDeviceGroupPropertiesKHX *pPhysicalDeviceGroupProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkEnumeratePhysicalDeviceGroupsKHX(my_data, pPhysicalDeviceGroupCount,
                                                                    pPhysicalDeviceGroupProperties);

    if (!skip) {
        result = my_data->dispatch_table.EnumeratePhysicalDeviceGroupsKHX(instance, pPhysicalDeviceGroupCount,
                                                                          pPhysicalDeviceGroupProperties);
        validate_result(my_data->report_data, "vkEnumeratePhysicalDeviceGroupsKHX", {}, result);
    }
    return result;
}

// Definitions for the VK_KHX_device_group extension

VKAPI_ATTR void VKAPI_CALL GetDeviceGroupPeerMemoryFeaturesKHX(VkDevice device, uint32_t heapIndex, uint32_t localDeviceIndex,
                                                               uint32_t remoteDeviceIndex,
                                                               VkPeerMemoryFeatureFlagsKHX *pPeerMemoryFeatures) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDeviceGroupPeerMemoryFeaturesKHX(my_data, heapIndex, localDeviceIndex,
                                                                       remoteDeviceIndex, pPeerMemoryFeatures);

    if (!skip) {
        my_data->dispatch_table.GetDeviceGroupPeerMemoryFeaturesKHX(device, heapIndex, localDeviceIndex, remoteDeviceIndex,
                                                                    pPeerMemoryFeatures);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory2KHX(VkDevice device, uint32_t bindInfoCount,
                                                    const VkBindBufferMemoryInfoKHX *pBindInfos) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkBindBufferMemory2KHX(my_data, bindInfoCount, pBindInfos);

    if (!skip) {
        result = my_data->dispatch_table.BindBufferMemory2KHX(device, bindInfoCount, pBindInfos);
        validate_result(my_data->report_data, "vkBindBufferMemory2KHX", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2KHX(VkDevice device, uint32_t bindInfoCount,
                                                   const VkBindImageMemoryInfoKHX *pBindInfos) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkBindImageMemory2KHX(my_data, bindInfoCount, pBindInfos);

    if (!skip) {
        result = my_data->dispatch_table.BindImageMemory2KHX(device, bindInfoCount, pBindInfos);
        validate_result(my_data->report_data, "vkBindImageMemory2KHX", {}, result);
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdSetDeviceMaskKHX(VkCommandBuffer commandBuffer, uint32_t deviceMask) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetDeviceMaskKHX(my_data, deviceMask);

    if (!skip) {
        my_data->dispatch_table.CmdSetDeviceMaskKHX(commandBuffer, deviceMask);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL
GetDeviceGroupPresentCapabilitiesKHX(VkDevice device, VkDeviceGroupPresentCapabilitiesKHX *pDeviceGroupPresentCapabilities) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDeviceGroupPresentCapabilitiesKHX(my_data, pDeviceGroupPresentCapabilities);

    if (!skip) {
        result = my_data->dispatch_table.GetDeviceGroupPresentCapabilitiesKHX(device, pDeviceGroupPresentCapabilities);
        validate_result(my_data->report_data, "vkGetDeviceGroupPresentCapabilitiesKHX", {}, result);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupSurfacePresentModesKHX(VkDevice device, VkSurfaceKHR surface,
                                                                    VkDeviceGroupPresentModeFlagsKHX *pModes) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetDeviceGroupSurfacePresentModesKHX(my_data, surface, pModes);

    if (!skip) {
        result = my_data->dispatch_table.GetDeviceGroupSurfacePresentModesKHX(device, surface, pModes);
        validate_result(my_data->report_data, "vkGetDeviceGroupSurfacePresentModesKHX", {}, result);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImage2KHX(VkDevice device, const VkAcquireNextImageInfoKHX *pAcquireInfo,
                                                    uint32_t *pImageIndex) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkAcquireNextImage2KHX(my_data, pAcquireInfo, pImageIndex);

    if (!skip) {
        result = my_data->dispatch_table.AcquireNextImage2KHX(device, pAcquireInfo, pImageIndex);
        validate_result(my_data->report_data, "vkAcquireNextImage2KHX", {}, result);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchBaseKHX(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                                              uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY,
                                              uint32_t groupCountZ) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdDispatchBaseKHX(my_data, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY,
                                                      groupCountZ);

    if (!skip) {
        my_data->dispatch_table.CmdDispatchBaseKHX(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY,
                                                   groupCountZ);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDevicePresentRectanglesKHX(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                                                                     uint32_t *pRectCount, VkRect2D *pRects) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDevicePresentRectanglesKHX(my_data, surface, pRectCount, pRects);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDevicePresentRectanglesKHX(physicalDevice, surface, pRectCount, pRects);

        validate_result(my_data->report_data, "vkGetPhysicalDevicePresentRectanglesKHX", {}, result);
    }

    return result;
}

// Definitions for the VK_EXT_acquire_xlib_display extension

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
VKAPI_ATTR VkResult VKAPI_CALL AcquireXlibDisplayEXT(VkPhysicalDevice physicalDevice, Display *dpy, VkDisplayKHR display) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;
    skip |= parameter_validation_vkAcquireXlibDisplayEXT(my_data, dpy, display);
    if (!skip) {
        result = my_data->dispatch_table.AcquireXlibDisplayEXT(physicalDevice, dpy, display);
        validate_result(my_data->report_data, "vkAcquireXlibDisplayEXT", {}, result);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetRandROutputDisplayEXT(VkPhysicalDevice physicalDevice, Display *dpy, RROutput rrOutput,
                                                        VkDisplayKHR *pDisplay) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;
    skip |= parameter_validation_vkGetRandROutputDisplayEXT(my_data, dpy, rrOutput, pDisplay);
    if (!skip) {
        result = my_data->dispatch_table.GetRandROutputDisplayEXT(physicalDevice, dpy, rrOutput, pDisplay);
        validate_result(my_data->report_data, "vkGetRandROutputDisplayEXT", {}, result);
    }
    return result;
}
#endif  // VK_USE_PLATFORM_XLIB_XRANDR_EXT

// Definitions for the VK_EXT_debug_marker Extension

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectTagEXT(VkDevice device, const VkDebugMarkerObjectTagInfoEXT *pTagInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDebugMarkerSetObjectTagEXT(my_data, pTagInfo);

    if (!skip) {
        if (my_data->dispatch_table.DebugMarkerSetObjectTagEXT) {
            result = my_data->dispatch_table.DebugMarkerSetObjectTagEXT(device, pTagInfo);
            validate_result(my_data->report_data, "vkDebugMarkerSetObjectTagEXT", {}, result);
        } else {
            result = VK_SUCCESS;
        }
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectNameEXT(VkDevice device, const VkDebugMarkerObjectNameInfoEXT *pNameInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);
    if (pNameInfo->pObjectName) {
        my_data->report_data->debugObjectNameMap->insert(
            std::make_pair<uint64_t, std::string>((uint64_t &&)pNameInfo->object, pNameInfo->pObjectName));
    } else {
        my_data->report_data->debugObjectNameMap->erase(pNameInfo->object);
    }
    skip |= parameter_validation_vkDebugMarkerSetObjectNameEXT(my_data, pNameInfo);

    if (!skip) {
        if (my_data->dispatch_table.DebugMarkerSetObjectNameEXT) {
            result = my_data->dispatch_table.DebugMarkerSetObjectNameEXT(device, pNameInfo);
            validate_result(my_data->report_data, "vkDebugMarkerSetObjectNameEXT", {}, result);
        } else {
            result = VK_SUCCESS;
        }
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerBeginEXT(VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT *pMarkerInfo) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdDebugMarkerBeginEXT(my_data, pMarkerInfo);

    if (!skip && my_data->dispatch_table.CmdDebugMarkerBeginEXT) {
        my_data->dispatch_table.CmdDebugMarkerBeginEXT(commandBuffer, pMarkerInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerInsertEXT(VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT *pMarkerInfo) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdDebugMarkerInsertEXT(my_data, pMarkerInfo);

    if (!skip && my_data->dispatch_table.CmdDebugMarkerInsertEXT) {
        my_data->dispatch_table.CmdDebugMarkerInsertEXT(commandBuffer, pMarkerInfo);
    }
}

// Definitions for the VK_EXT_direct_mode_display extension

VKAPI_ATTR VkResult VKAPI_CALL ReleaseDisplayEXT(VkPhysicalDevice physicalDevice, VkDisplayKHR display) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;

    skip |= parameter_validation_vkReleaseDisplayEXT(my_data, display);

    if (!skip) {
        result = my_data->dispatch_table.ReleaseDisplayEXT(physicalDevice, display);
        validate_result(my_data->report_data, "vkGetRandROutputDisplayEXT", {}, result);
    }
    return result;
}

// Definitions for the VK_EXT_discard_rectangles extension

VKAPI_ATTR void VKAPI_CALL CmdSetDiscardRectangleEXT(VkCommandBuffer commandBuffer, uint32_t firstDiscardRectangle,
                                                     uint32_t discardRectangleCount, const VkRect2D *pDiscardRectangles) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetDiscardRectangleEXT(my_data, firstDiscardRectangle,
                                                             discardRectangleCount, pDiscardRectangles);

    if (!skip && my_data->dispatch_table.CmdSetDiscardRectangleEXT) {
        my_data->dispatch_table.CmdSetDiscardRectangleEXT(commandBuffer, firstDiscardRectangle, discardRectangleCount,
                                                          pDiscardRectangles);
    }
}

// Definitions for the VK_EXT_display_control extension

VKAPI_ATTR VkResult VKAPI_CALL DisplayPowerControlEXT(VkDevice device, VkDisplayKHR display,
                                                      const VkDisplayPowerInfoEXT *pDisplayPowerInfo) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDisplayPowerControlEXT(my_data, display, pDisplayPowerInfo);

    if (!skip) {
        if (my_data->dispatch_table.DisplayPowerControlEXT) {
            result = my_data->dispatch_table.DisplayPowerControlEXT(device, display, pDisplayPowerInfo);
            validate_result(my_data->report_data, "vkDisplayPowerControlEXT", {}, result);
        } else {
            result = VK_SUCCESS;
        }
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDeviceEventEXT(VkDevice device, const VkDeviceEventInfoEXT *pDeviceEventInfo,
                                                      const VkAllocationCallbacks *pAllocator, VkFence *pFence) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkRegisterDeviceEventEXT(my_data, pDeviceEventInfo, pAllocator, pFence);

    if (!skip) {
        if (my_data->dispatch_table.RegisterDeviceEventEXT) {
            result = my_data->dispatch_table.RegisterDeviceEventEXT(device, pDeviceEventInfo, pAllocator, pFence);
            validate_result(my_data->report_data, "vkRegisterDeviceEventEXT", {}, result);
        } else {
            result = VK_SUCCESS;
        }
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDisplayEventEXT(VkDevice device, VkDisplayKHR display,
                                                       const VkDisplayEventInfoEXT *pDisplayEventInfo,
                                                       const VkAllocationCallbacks *pAllocator, VkFence *pFence) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkRegisterDisplayEventEXT(my_data, display, pDisplayEventInfo, pAllocator, pFence);

    if (!skip) {
        if (my_data->dispatch_table.RegisterDisplayEventEXT) {
            result = my_data->dispatch_table.RegisterDisplayEventEXT(device, display, pDisplayEventInfo, pAllocator, pFence);
            validate_result(my_data->report_data, "vkRegisterDisplayEventEXT", {}, result);
        } else {
            result = VK_SUCCESS;
        }
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainCounterEXT(VkDevice device, VkSwapchainKHR swapchain,
                                                      VkSurfaceCounterFlagBitsEXT counter, uint64_t *pCounterValue) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetSwapchainCounterEXT(my_data, swapchain, counter, pCounterValue);

    if (!skip) {
        if (my_data->dispatch_table.GetSwapchainCounterEXT) {
            result = my_data->dispatch_table.GetSwapchainCounterEXT(device, swapchain, counter, pCounterValue);
            validate_result(my_data->report_data, "vkGetSwapchainCounterEXT", {}, result);
        } else {
            result = VK_SUCCESS;
        }
    }

    return result;
}

// Definitions for the VK_AMD_draw_indirect_count extension

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                   VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                   uint32_t stride) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdDrawIndirectCountAMD(my_data, buffer, offset, countBuffer, countBufferOffset,
                                                           maxDrawCount, stride);
    if (!skip) {
        my_data->dispatch_table.CmdDrawIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount,
                                                        stride);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                          VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                          uint32_t maxDrawCount, uint32_t stride) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdDrawIndexedIndirectCountAMD(my_data, buffer, offset, countBuffer,
                                                                  countBufferOffset, maxDrawCount, stride);
    if (!skip) {
        my_data->dispatch_table.CmdDrawIndexedIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset,
                                                               maxDrawCount, stride);
    }
}

// Definitions for the VK_EXT_display_surface_counter extension

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2EXT(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                                                                        VkSurfaceCapabilities2EXT *pSurfaceCapabilities) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    bool skip = false;
    skip |= parameter_validation_vkGetPhysicalDeviceSurfaceCapabilities2EXT(my_data, surface, pSurfaceCapabilities);
    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceSurfaceCapabilities2EXT(physicalDevice, surface, pSurfaceCapabilities);
        validate_result(my_data->report_data, "vkGetPhysicalDeviceSurfaceCapabilities2EXT", {}, result);
    }
    return result;
}

// Definitions for the VK_NV_clip_space_w_scaling Extension

VKAPI_ATTR void VKAPI_CALL CmdSetViewportWScalingNV(VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount,
                                                    const VkViewportWScalingNV *pViewportWScalings) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdSetViewportWScalingNV(my_data, firstViewport, viewportCount, pViewportWScalings);

    if (!skip) {
        my_data->dispatch_table.CmdSetViewportWScalingNV(commandBuffer, firstViewport, viewportCount, pViewportWScalings);
    }
}

// Definitions for the VK_NV_external_memory_capabilities Extension

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceExternalImageFormatPropertiesNV(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
    VkImageCreateFlags flags, VkExternalMemoryHandleTypeFlagsNV externalHandleType,
    VkExternalImageFormatPropertiesNV *pExternalImageFormatProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(
        my_data, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);

    if (!skip) {
        result = my_data->dispatch_table.GetPhysicalDeviceExternalImageFormatPropertiesNV(
            physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);
        const std::vector<VkResult> ignore_list = {VK_ERROR_FORMAT_NOT_SUPPORTED};
        validate_result(my_data->report_data, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV", ignore_list, result);
    }

    return result;
}

// VK_NV_external_memory_win32 Extension

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleNV(VkDevice device, VkDeviceMemory memory,
                                                      VkExternalMemoryHandleTypeFlagsNV handleType, HANDLE *pHandle) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkGetMemoryWin32HandleNV(my_data, memory, handleType, pHandle);

    if (!skip) {
        result = my_data->dispatch_table.GetMemoryWin32HandleNV(device, memory, handleType, pHandle);
    }

    return result;
}
#endif  // VK_USE_PLATFORM_WIN32_KHR

// VK_NVX_device_generated_commands Extension

VKAPI_ATTR void VKAPI_CALL CmdProcessCommandsNVX(VkCommandBuffer commandBuffer,
                                                 const VkCmdProcessCommandsInfoNVX *pProcessCommandsInfo) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdProcessCommandsNVX(my_data, pProcessCommandsInfo);
    if (!skip) {
        my_data->dispatch_table.CmdProcessCommandsNVX(commandBuffer, pProcessCommandsInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdReserveSpaceForCommandsNVX(VkCommandBuffer commandBuffer,
                                                         const VkCmdReserveSpaceForCommandsInfoNVX *pReserveSpaceInfo) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCmdReserveSpaceForCommandsNVX(my_data, pReserveSpaceInfo);
    if (!skip) {
        my_data->dispatch_table.CmdReserveSpaceForCommandsNVX(commandBuffer, pReserveSpaceInfo);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateIndirectCommandsLayoutNVX(VkDevice device,
                                                               const VkIndirectCommandsLayoutCreateInfoNVX *pCreateInfo,
                                                               const VkAllocationCallbacks *pAllocator,
                                                               VkIndirectCommandsLayoutNVX *pIndirectCommandsLayout) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateIndirectCommandsLayoutNVX(my_data, pCreateInfo, pAllocator,
                                                                   pIndirectCommandsLayout);
    if (!skip) {
        result = my_data->dispatch_table.CreateIndirectCommandsLayoutNVX(device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
        validate_result(my_data->report_data, "vkCreateIndirectCommandsLayoutNVX", {}, result);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyIndirectCommandsLayoutNVX(VkDevice device, VkIndirectCommandsLayoutNVX indirectCommandsLayout,
                                                            const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyIndirectCommandsLayoutNVX(my_data, indirectCommandsLayout, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyIndirectCommandsLayoutNVX(device, indirectCommandsLayout, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateObjectTableNVX(VkDevice device, const VkObjectTableCreateInfoNVX *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkObjectTableNVX *pObjectTable) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkCreateObjectTableNVX(my_data, pCreateInfo, pAllocator, pObjectTable);
    if (!skip) {
        result = my_data->dispatch_table.CreateObjectTableNVX(device, pCreateInfo, pAllocator, pObjectTable);
        validate_result(my_data->report_data, "vkCreateObjectTableNVX", {}, result);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyObjectTableNVX(VkDevice device, VkObjectTableNVX objectTable,
                                                 const VkAllocationCallbacks *pAllocator) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkDestroyObjectTableNVX(my_data, objectTable, pAllocator);

    if (!skip) {
        my_data->dispatch_table.DestroyObjectTableNVX(device, objectTable, pAllocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterObjectsNVX(VkDevice device, VkObjectTableNVX objectTable, uint32_t objectCount,
                                                  const VkObjectTableEntryNVX *const *ppObjectTableEntries,
                                                  const uint32_t *pObjectIndices) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkRegisterObjectsNVX(my_data, objectTable, objectCount, ppObjectTableEntries,
                                                      pObjectIndices);
    if (!skip) {
        result = my_data->dispatch_table.RegisterObjectsNVX(device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);
        validate_result(my_data->report_data, "vkRegisterObjectsNVX", {}, result);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL UnregisterObjectsNVX(VkDevice device, VkObjectTableNVX objectTable, uint32_t objectCount,
                                                    const VkObjectEntryTypeNVX *pObjectEntryTypes, const uint32_t *pObjectIndices) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);

    skip |= parameter_validation_vkUnregisterObjectsNVX(my_data, objectTable, objectCount, pObjectEntryTypes,
                                                        pObjectIndices);
    if (!skip) {
        result = my_data->dispatch_table.UnregisterObjectsNVX(device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);
        validate_result(my_data->report_data, "vkUnregisterObjectsNVX", {}, result);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceGeneratedCommandsPropertiesNVX(VkPhysicalDevice physicalDevice,
                                                                           VkDeviceGeneratedCommandsFeaturesNVX *pFeatures,
                                                                           VkDeviceGeneratedCommandsLimitsNVX *pLimits) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    assert(my_data != NULL);
    skip |= parameter_validation_vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX(my_data, pFeatures, pLimits);
    if (!skip) {
        my_data->dispatch_table.GetPhysicalDeviceGeneratedCommandsPropertiesNVX(physicalDevice, pFeatures, pLimits);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPastPresentationTimingGOOGLE(VkDevice device, VkSwapchainKHR swapchain,
                                                               uint32_t *pPresentationTimingCount,
                                                               VkPastPresentationTimingGOOGLE *pPresentationTimings) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);
    skip |= parameter_validation_vkGetPastPresentationTimingGOOGLE(my_data, swapchain, pPresentationTimingCount,
                                                                   pPresentationTimings);
    if (!skip) {
        result = my_data->dispatch_table.GetPastPresentationTimingGOOGLE(device, swapchain, pPresentationTimingCount,
                                                                         pPresentationTimings);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetRefreshCycleDurationGOOGLE(VkDevice device, VkSwapchainKHR swapchain,
                                                             VkRefreshCycleDurationGOOGLE *pDisplayTimingProperties) {
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);
    skip |= parameter_validation_vkGetRefreshCycleDurationGOOGLE(my_data, swapchain, pDisplayTimingProperties);
    if (!skip) {
        result = my_data->dispatch_table.GetRefreshCycleDurationGOOGLE(device, swapchain, pDisplayTimingProperties);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL SetHdrMetadataEXT(VkDevice device, uint32_t swapchainCount, const VkSwapchainKHR *pSwapchains,
                                             const VkHdrMetadataEXT *pMetadata) {
    bool skip = false;
    auto my_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    assert(my_data != NULL);
    skip |= parameter_validation_vkSetHdrMetadataEXT(my_data, swapchainCount, pSwapchains, pMetadata);
    if (!skip) {
        my_data->dispatch_table.SetHdrMetadataEXT(device, swapchainCount, pSwapchains, pMetadata);
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char *funcName) {
    const auto item = name_to_funcptr_map.find(funcName);
    if (item != name_to_funcptr_map.end()) {
        return reinterpret_cast<PFN_vkVoidFunction>(item->second);
    }

    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    const auto &table = device_data->dispatch_table;
    if (!table.GetDeviceProcAddr) return nullptr;
    return table.GetDeviceProcAddr(device, funcName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char *funcName) {
    const auto item = name_to_funcptr_map.find(funcName);
    if (item != name_to_funcptr_map.end()) {
        return reinterpret_cast<PFN_vkVoidFunction>(item->second);
    }

    auto instance_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    auto &table = instance_data->dispatch_table;
    if (!table.GetInstanceProcAddr) return nullptr;
    return table.GetInstanceProcAddr(instance, funcName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetPhysicalDeviceProcAddr(VkInstance instance, const char *funcName) {
    assert(instance);
    auto pdev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);

    if (!pdev_data->dispatch_table.GetPhysicalDeviceProcAddr) return nullptr;
    return pdev_data->dispatch_table.GetPhysicalDeviceProcAddr(instance, funcName);
}

}  // namespace parameter_validation

// loader-layer interface v0

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                                      VkExtensionProperties *pProperties) {
    return parameter_validation::EnumerateInstanceExtensionProperties(pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pCount,
                                                                                  VkLayerProperties *pProperties) {
    return parameter_validation::EnumerateInstanceLayerProperties(pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                                                VkLayerProperties *pProperties) {
    // the layer command handles VK_NULL_HANDLE just fine internally
    assert(physicalDevice == VK_NULL_HANDLE);
    return parameter_validation::EnumerateDeviceLayerProperties(VK_NULL_HANDLE, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                                    const char *pLayerName, uint32_t *pCount,
                                                                                    VkExtensionProperties *pProperties) {
    // the layer command handles VK_NULL_HANDLE just fine internally
    assert(physicalDevice == VK_NULL_HANDLE);
    return parameter_validation::EnumerateDeviceExtensionProperties(VK_NULL_HANDLE, pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice dev, const char *funcName) {
    return parameter_validation::GetDeviceProcAddr(dev, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *funcName) {
    return parameter_validation::GetInstanceProcAddr(instance, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance,
                                                                                           const char *funcName) {
    return parameter_validation::GetPhysicalDeviceProcAddr(instance, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    assert(pVersionStruct != NULL);
    assert(pVersionStruct->sType == LAYER_NEGOTIATE_INTERFACE_STRUCT);

    // Fill in the function pointers if our version is at least capable of having the structure contain them.
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = vk_layerGetPhysicalDeviceProcAddr;
    }

    if (pVersionStruct->loaderLayerInterfaceVersion < CURRENT_LOADER_LAYER_INTERFACE_VERSION) {
        parameter_validation::loader_layer_if_version = pVersionStruct->loaderLayerInterfaceVersion;
    } else if (pVersionStruct->loaderLayerInterfaceVersion > CURRENT_LOADER_LAYER_INTERFACE_VERSION) {
        pVersionStruct->loaderLayerInterfaceVersion = CURRENT_LOADER_LAYER_INTERFACE_VERSION;
    }

    return VK_SUCCESS;
}

/*
 * Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (c) 2015-2016 Google, Inc.
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
 * Author: Tobin Ehlis <tobine@google.com>
 * Author: Mark Lobodzinski <mark@lunarg.com>
 */

#define NOMINMAX

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <list>
#include <memory>
#include <algorithm>

// For Windows, this #include must come before other Vk headers.
#include "vk_loader_platform.h"

#include "unique_objects.h"
#include "vk_dispatch_table_helper.h"
#include "vk_layer_config.h"
#include "vk_layer_data.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_logging.h"
#include "vk_layer_table.h"
#include "vk_layer_utils.h"
#include "vk_layer_utils.h"
#include "vk_enum_string_helper.h"
#include "vk_validation_error_messages.h"
#include "vk_object_types.h"
#include "vulkan/vk_layer.h"

// This intentionally includes a cpp file
#include "vk_safe_struct.cpp"

#include "unique_objects_wrappers.h"

namespace unique_objects {

static uint32_t loader_layer_if_version = CURRENT_LOADER_LAYER_INTERFACE_VERSION;

static void initUniqueObjects(instance_layer_data *instance_data, const VkAllocationCallbacks *pAllocator) {
    layer_debug_actions(instance_data->report_data, instance_data->logging_callback, pAllocator, "google_unique_objects");
}

// Check enabled instance extensions against supported instance extension whitelist
static void InstanceExtensionWhitelist(const VkInstanceCreateInfo *pCreateInfo, VkInstance instance) {
    instance_layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);

    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        // Check for recognized instance extensions
        if (!white_list(pCreateInfo->ppEnabledExtensionNames[i], kUniqueObjectsSupportedInstanceExtensions)) {
            log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_UNDEFINED, "UniqueObjects",
                    "Instance Extension %s is not supported by this layer.  Using this extension may adversely affect "
                    "validation results and/or produce undefined behavior.",
                    pCreateInfo->ppEnabledExtensionNames[i]);
        }
    }
}

// Check enabled device extensions against supported device extension whitelist
static void DeviceExtensionWhitelist(const VkDeviceCreateInfo *pCreateInfo, VkDevice device) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        // Check for recognized device extensions
        if (!white_list(pCreateInfo->ppEnabledExtensionNames[i], kUniqueObjectsSupportedDeviceExtensions)) {
            log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_UNDEFINED, "UniqueObjects",
                    "Device Extension %s is not supported by this layer.  Using this extension may adversely affect "
                    "validation results and/or produce undefined behavior.",
                    pCreateInfo->ppEnabledExtensionNames[i]);
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                              VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance)fpGetInstanceProcAddr(NULL, "vkCreateInstance");
    if (fpCreateInstance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) {
        return result;
    }

    instance_layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(*pInstance), instance_layer_data_map);
    instance_data->instance = *pInstance;
    layer_init_instance_dispatch_table(*pInstance, &instance_data->dispatch_table, fpGetInstanceProcAddr);

    instance_data->instance = *pInstance;
    instance_data->report_data =
        debug_report_create_instance(&instance_data->dispatch_table, *pInstance, pCreateInfo->enabledExtensionCount,
                                     pCreateInfo->ppEnabledExtensionNames);

    // Set up temporary debug callbacks to output messages at CreateInstance-time
    if (!layer_copy_tmp_callbacks(pCreateInfo->pNext, &instance_data->num_tmp_callbacks, &instance_data->tmp_dbg_create_infos,
                                  &instance_data->tmp_callbacks)) {
        if (instance_data->num_tmp_callbacks > 0) {
            if (layer_enable_tmp_callbacks(instance_data->report_data, instance_data->num_tmp_callbacks,
                                           instance_data->tmp_dbg_create_infos, instance_data->tmp_callbacks)) {
                layer_free_tmp_callbacks(instance_data->tmp_dbg_create_infos, instance_data->tmp_callbacks);
                instance_data->num_tmp_callbacks = 0;
            }
        }
    }

    initUniqueObjects(instance_data, pAllocator);
    InstanceExtensionWhitelist(pCreateInfo, *pInstance);

    // Disable and free tmp callbacks, no longer necessary
    if (instance_data->num_tmp_callbacks > 0) {
        layer_disable_tmp_callbacks(instance_data->report_data, instance_data->num_tmp_callbacks, instance_data->tmp_callbacks);
        layer_free_tmp_callbacks(instance_data->tmp_dbg_create_infos, instance_data->tmp_callbacks);
        instance_data->num_tmp_callbacks = 0;
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    dispatch_key key = get_dispatch_key(instance);
    instance_layer_data *instance_data = GetLayerDataPtr(key, instance_layer_data_map);
    VkLayerInstanceDispatchTable *disp_table = &instance_data->dispatch_table;
    disp_table->DestroyInstance(instance, pAllocator);

    // Clean up logging callback, if any
    while (instance_data->logging_callback.size() > 0) {
        VkDebugReportCallbackEXT callback = instance_data->logging_callback.back();
        layer_destroy_msg_callback(instance_data->report_data, callback, pAllocator);
        instance_data->logging_callback.pop_back();
    }

    layer_debug_report_destroy_instance(instance_data->report_data);
    FreeLayerDataPtr(key, instance_layer_data_map);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    instance_layer_data *my_instance_data = GetLayerDataPtr(get_dispatch_key(gpu), instance_layer_data_map);
    VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(my_instance_data->instance, "vkCreateDevice");
    if (fpCreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateDevice(gpu, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    layer_data *my_device_data = GetLayerDataPtr(get_dispatch_key(*pDevice), layer_data_map);
    my_device_data->report_data = layer_debug_report_create_device(my_instance_data->report_data, *pDevice);

    // Setup layer's device dispatch table
    layer_init_device_dispatch_table(*pDevice, &my_device_data->dispatch_table, fpGetDeviceProcAddr);

    DeviceExtensionWhitelist(pCreateInfo, *pDevice);

    // Set gpu for this device in order to get at any objects mapped at instance level
    my_device_data->instance_data = my_instance_data;

    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    dispatch_key key = get_dispatch_key(device);
    layer_data *dev_data = GetLayerDataPtr(key, layer_data_map);

    layer_debug_report_destroy_device(device);
    dev_data->dispatch_table.DestroyDevice(device, pAllocator);

    FreeLayerDataPtr(key, layer_data_map);
}

static const VkLayerProperties globalLayerProps = {"VK_LAYER_GOOGLE_unique_objects",
                                                   VK_LAYER_API_VERSION,  // specVersion
                                                   1,                     // implementationVersion
                                                   "Google Validation Layer"};

/// Declare prototype for these functions
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetPhysicalDeviceProcAddr(VkInstance instance, const char *funcName);

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(uint32_t *pCount, VkLayerProperties *pProperties) {
    return util_GetLayerProperties(1, &globalLayerProps, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                              VkLayerProperties *pProperties) {
    return util_GetLayerProperties(1, &globalLayerProps, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                    VkExtensionProperties *pProperties) {
    if (pLayerName && !strcmp(pLayerName, globalLayerProps.layerName))
        return util_GetExtensionProperties(0, NULL, pCount, pProperties);

    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                                  uint32_t *pCount, VkExtensionProperties *pProperties) {
    if (pLayerName && !strcmp(pLayerName, globalLayerProps.layerName))
        return util_GetExtensionProperties(0, nullptr, pCount, pProperties);

    assert(physicalDevice);

    dispatch_key key = get_dispatch_key(physicalDevice);
    instance_layer_data *instance_data = GetLayerDataPtr(key, instance_layer_data_map);
    return instance_data->dispatch_table.EnumerateDeviceExtensionProperties(physicalDevice, NULL, pCount, pProperties);
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

    instance_layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    const auto &table = instance_data->dispatch_table;
    if (!table.GetInstanceProcAddr) return nullptr;
    return table.GetInstanceProcAddr(instance, funcName);
}


VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetPhysicalDeviceProcAddr(VkInstance instance, const char *funcName) {
    instance_layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkLayerInstanceDispatchTable *disp_table = &instance_data->dispatch_table;
    if (disp_table->GetPhysicalDeviceProcAddr == NULL) {
        return NULL;
    }
    return disp_table->GetPhysicalDeviceProcAddr(instance, funcName);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkComputePipelineCreateInfo *local_pCreateInfos = NULL;
    if (pCreateInfos) {
        std::lock_guard<std::mutex> lock(global_lock);
        local_pCreateInfos = new safe_VkComputePipelineCreateInfo[createInfoCount];
        for (uint32_t idx0 = 0; idx0 < createInfoCount; ++idx0) {
            local_pCreateInfos[idx0].initialize(&pCreateInfos[idx0]);
            if (pCreateInfos[idx0].basePipelineHandle) {
                local_pCreateInfos[idx0].basePipelineHandle = Unwrap(device_data, pCreateInfos[idx0].basePipelineHandle);
            }
            if (pCreateInfos[idx0].layout) {
                local_pCreateInfos[idx0].layout = Unwrap(device_data, pCreateInfos[idx0].layout);
            }
            if (pCreateInfos[idx0].stage.module) {
                local_pCreateInfos[idx0].stage.module = Unwrap(device_data, pCreateInfos[idx0].stage.module);
            }
        }
    }
    if (pipelineCache) {
        std::lock_guard<std::mutex> lock(global_lock);
        pipelineCache = Unwrap(device_data, pipelineCache);
    }

    VkResult result = device_data->dispatch_table.CreateComputePipelines(
        device, pipelineCache, createInfoCount, local_pCreateInfos->ptr(), pAllocator, pPipelines);
    delete[] local_pCreateInfos;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t i = 0; i < createInfoCount; ++i) {
            if (pPipelines[i] != VK_NULL_HANDLE) {
                pPipelines[i] = WrapNew(device_data, pPipelines[i]);
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                       const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                       const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkGraphicsPipelineCreateInfo *local_pCreateInfos = nullptr;
    if (pCreateInfos) {
        local_pCreateInfos = new safe_VkGraphicsPipelineCreateInfo[createInfoCount];
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t idx0 = 0; idx0 < createInfoCount; ++idx0) {
            local_pCreateInfos[idx0].initialize(&pCreateInfos[idx0]);
            if (pCreateInfos[idx0].basePipelineHandle) {
                local_pCreateInfos[idx0].basePipelineHandle = Unwrap(device_data, pCreateInfos[idx0].basePipelineHandle);
            }
            if (pCreateInfos[idx0].layout) {
                local_pCreateInfos[idx0].layout = Unwrap(device_data, pCreateInfos[idx0].layout);
            }
            if (pCreateInfos[idx0].pStages) {
                for (uint32_t idx1 = 0; idx1 < pCreateInfos[idx0].stageCount; ++idx1) {
                    if (pCreateInfos[idx0].pStages[idx1].module) {
                        local_pCreateInfos[idx0].pStages[idx1].module = Unwrap(device_data, pCreateInfos[idx0].pStages[idx1].module);
                    }
                }
            }
            if (pCreateInfos[idx0].renderPass) {
                local_pCreateInfos[idx0].renderPass = Unwrap(device_data, pCreateInfos[idx0].renderPass);
            }
        }
    }
    if (pipelineCache) {
        std::lock_guard<std::mutex> lock(global_lock);
        pipelineCache = Unwrap(device_data, pipelineCache);
    }

    VkResult result = device_data->dispatch_table.CreateGraphicsPipelines(
        device, pipelineCache, createInfoCount, local_pCreateInfos->ptr(), pAllocator, pPipelines);
    delete[] local_pCreateInfos;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t i = 0; i < createInfoCount; ++i) {
            if (pPipelines[i] != VK_NULL_HANDLE) {
                pPipelines[i] = WrapNew(device_data, pPipelines[i]);
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *pCreateInfo,
                                                  const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchain) {
    layer_data *my_map_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkSwapchainCreateInfoKHR *local_pCreateInfo = NULL;
    if (pCreateInfo) {
        std::lock_guard<std::mutex> lock(global_lock);
        local_pCreateInfo = new safe_VkSwapchainCreateInfoKHR(pCreateInfo);
        local_pCreateInfo->oldSwapchain = Unwrap(my_map_data, pCreateInfo->oldSwapchain);
        // Surface is instance-level object
        local_pCreateInfo->surface = Unwrap(my_map_data->instance_data, pCreateInfo->surface);
    }

    VkResult result = my_map_data->dispatch_table.CreateSwapchainKHR(
        device, local_pCreateInfo->ptr(), pAllocator, pSwapchain);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSwapchain = WrapNew(my_map_data, *pSwapchain);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSharedSwapchainsKHR(VkDevice device, uint32_t swapchainCount,
                                                         const VkSwapchainCreateInfoKHR *pCreateInfos,
                                                         const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchains) {
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkSwapchainCreateInfoKHR *local_pCreateInfos = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfos) {
            local_pCreateInfos = new safe_VkSwapchainCreateInfoKHR[swapchainCount];
            for (uint32_t i = 0; i < swapchainCount; ++i) {
                local_pCreateInfos[i].initialize(&pCreateInfos[i]);
                if (pCreateInfos[i].surface) {
                    // Surface is instance-level object
                    local_pCreateInfos[i].surface = Unwrap(dev_data->instance_data, pCreateInfos[i].surface);
                }
                if (pCreateInfos[i].oldSwapchain) {
                    local_pCreateInfos[i].oldSwapchain = Unwrap(dev_data, pCreateInfos[i].oldSwapchain);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateSharedSwapchainsKHR(
        device, swapchainCount, local_pCreateInfos->ptr(), pAllocator, pSwapchains);
    if (local_pCreateInfos) delete[] local_pCreateInfos;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t i = 0; i < swapchainCount; i++) {
            pSwapchains[i] = WrapNew(dev_data, pSwapchains[i]);
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t *pSwapchainImageCount,
                                                     VkImage *pSwapchainImages) {
    layer_data *my_device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    if (VK_NULL_HANDLE != swapchain) {
        std::lock_guard<std::mutex> lock(global_lock);
        swapchain = Unwrap(my_device_data, swapchain);
    }
    VkResult result =
        my_device_data->dispatch_table.GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
    // TODO : Need to add corresponding code to delete these images
    if (VK_SUCCESS == result) {
        if ((*pSwapchainImageCount > 0) && pSwapchainImages) {
            std::lock_guard<std::mutex> lock(global_lock);
            for (uint32_t i = 0; i < *pSwapchainImageCount; ++i) {
                pSwapchainImages[i] = WrapNew(my_device_data, pSwapchainImages[i]);
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    safe_VkPresentInfoKHR *local_pPresentInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pPresentInfo) {
            local_pPresentInfo = new safe_VkPresentInfoKHR(pPresentInfo);
            if (local_pPresentInfo->pWaitSemaphores) {
                for (uint32_t index1 = 0; index1 < local_pPresentInfo->waitSemaphoreCount; ++index1) {
                    local_pPresentInfo->pWaitSemaphores[index1] = Unwrap(dev_data, pPresentInfo->pWaitSemaphores[index1]);
                }
            }
            if (local_pPresentInfo->pSwapchains) {
                for (uint32_t index1 = 0; index1 < local_pPresentInfo->swapchainCount; ++index1) {
                    local_pPresentInfo->pSwapchains[index1] = Unwrap(dev_data, pPresentInfo->pSwapchains[index1]);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.QueuePresentKHR(queue, local_pPresentInfo->ptr());

    // pResults is an output array embedded in a structure. The code generator neglects to copy back from the safe_* version,
    // so handle it as a special case here:
    if (pPresentInfo && pPresentInfo->pResults) {
        for (uint32_t i = 0; i < pPresentInfo->swapchainCount; i++) {
            pPresentInfo->pResults[i] = local_pPresentInfo->pResults[i];
        }
    }

    if (local_pPresentInfo) delete local_pPresentInfo;
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorUpdateTemplateKHR(VkDevice device,
                                                                 const VkDescriptorUpdateTemplateCreateInfoKHR *pCreateInfo,
                                                                 const VkAllocationCallbacks *pAllocator,
                                                                 VkDescriptorUpdateTemplateKHR *pDescriptorUpdateTemplate) {
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkDescriptorUpdateTemplateCreateInfoKHR *local_create_info = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_create_info = new safe_VkDescriptorUpdateTemplateCreateInfoKHR(pCreateInfo);
            if (pCreateInfo->descriptorSetLayout) {
                local_create_info->descriptorSetLayout = Unwrap(dev_data, pCreateInfo->descriptorSetLayout);
            }
            if (pCreateInfo->pipelineLayout) {
                local_create_info->pipelineLayout = Unwrap(dev_data, pCreateInfo->pipelineLayout);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateDescriptorUpdateTemplateKHR(
        device, local_create_info->ptr(), pAllocator, pDescriptorUpdateTemplate);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pDescriptorUpdateTemplate = WrapNew(dev_data, *pDescriptorUpdateTemplate);

        // Shadow template createInfo for later updates
        std::unique_ptr<TEMPLATE_STATE> template_state(new TEMPLATE_STATE(*pDescriptorUpdateTemplate, local_create_info));
        dev_data->desc_template_map[(uint64_t)*pDescriptorUpdateTemplate] = std::move(template_state);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorUpdateTemplateKHR(VkDevice device,
                                                              VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate,
                                                              const VkAllocationCallbacks *pAllocator) {
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t descriptor_update_template_id = reinterpret_cast<uint64_t &>(descriptorUpdateTemplate);
    dev_data->desc_template_map.erase(descriptor_update_template_id);
    descriptorUpdateTemplate = (VkDescriptorUpdateTemplateKHR)dev_data->unique_id_mapping[descriptor_update_template_id];
    dev_data->unique_id_mapping.erase(descriptor_update_template_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyDescriptorUpdateTemplateKHR(device, descriptorUpdateTemplate, pAllocator);
}

void *BuildUnwrappedUpdateTemplateBuffer(layer_data *dev_data, uint64_t descriptorUpdateTemplate, const void *pData) {
    auto const template_map_entry = dev_data->desc_template_map.find(descriptorUpdateTemplate);
    if (template_map_entry == dev_data->desc_template_map.end()) {
        assert(0);
    }
    auto const &create_info = template_map_entry->second->create_info;
    size_t allocation_size = 0;
    std::vector<std::tuple<size_t, VulkanObjectType, void *>> template_entries;

    for (uint32_t i = 0; i < create_info.descriptorUpdateEntryCount; i++) {
        for (uint32_t j = 0; j < create_info.pDescriptorUpdateEntries[i].descriptorCount; j++) {
            size_t offset = create_info.pDescriptorUpdateEntries[i].offset + j * create_info.pDescriptorUpdateEntries[i].stride;
            char *update_entry = (char *)(pData) + offset;

            switch (create_info.pDescriptorUpdateEntries[i].descriptorType) {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
                    auto image_entry = reinterpret_cast<VkDescriptorImageInfo *>(update_entry);
                    allocation_size = std::max(allocation_size, offset + sizeof(VkDescriptorImageInfo));

                    VkDescriptorImageInfo *wrapped_entry = new VkDescriptorImageInfo(*image_entry);
                    wrapped_entry->sampler = Unwrap(dev_data, image_entry->sampler);
                    wrapped_entry->imageView = Unwrap(dev_data, image_entry->imageView);
                    template_entries.emplace_back(offset, kVulkanObjectTypeImage, reinterpret_cast<void *>(wrapped_entry));
                } break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
                    auto buffer_entry = reinterpret_cast<VkDescriptorBufferInfo *>(update_entry);
                    allocation_size = std::max(allocation_size, offset + sizeof(VkDescriptorBufferInfo));

                    VkDescriptorBufferInfo *wrapped_entry = new VkDescriptorBufferInfo(*buffer_entry);
                    wrapped_entry->buffer = Unwrap(dev_data, buffer_entry->buffer);
                    template_entries.emplace_back(offset, kVulkanObjectTypeBuffer, reinterpret_cast<void *>(wrapped_entry));
                } break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
                    auto buffer_view_handle = reinterpret_cast<VkBufferView *>(update_entry);
                    allocation_size = std::max(allocation_size, offset + sizeof(VkBufferView));

                    VkBufferView wrapped_entry = Unwrap(dev_data, *buffer_view_handle);
                    template_entries.emplace_back(offset, kVulkanObjectTypeBufferView, reinterpret_cast<void *>(wrapped_entry));
                } break;
                default:
                    assert(0);
                    break;
            }
        }
    }
    // Allocate required buffer size and populate with source/unwrapped data
    void *unwrapped_data = malloc(allocation_size);
    for (auto &this_entry : template_entries) {
        VulkanObjectType type = std::get<1>(this_entry);
        void *destination = (char *)unwrapped_data + std::get<0>(this_entry);
        void *source = (char *)std::get<2>(this_entry);

        switch (type) {
            case kVulkanObjectTypeImage:
                *(reinterpret_cast<VkDescriptorImageInfo *>(destination)) = *(reinterpret_cast<VkDescriptorImageInfo *>(source));
                delete reinterpret_cast<VkDescriptorImageInfo *>(source);
                break;
            case kVulkanObjectTypeBuffer:
                *(reinterpret_cast<VkDescriptorBufferInfo *>(destination)) = *(reinterpret_cast<VkDescriptorBufferInfo *>(source));
                delete reinterpret_cast<VkDescriptorBufferInfo *>(source);
                break;
            case kVulkanObjectTypeBufferView:
                *(reinterpret_cast<VkBufferView *>(destination)) = reinterpret_cast<VkBufferView>(source);
                break;
            default:
                assert(0);
                break;
        }
    }
    return (void *)unwrapped_data;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSetWithTemplateKHR(VkDevice device, VkDescriptorSet descriptorSet,
                                                              VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate,
                                                              const void *pData) {
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    uint64_t template_handle = reinterpret_cast<uint64_t &>(descriptorUpdateTemplate);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        descriptorSet = Unwrap(dev_data, descriptorSet);
        descriptorUpdateTemplate = (VkDescriptorUpdateTemplateKHR)dev_data->unique_id_mapping[template_handle];
    }
    void *unwrapped_buffer = BuildUnwrappedUpdateTemplateBuffer(dev_data, template_handle, pData);
    dev_data->dispatch_table.UpdateDescriptorSetWithTemplateKHR(device, descriptorSet, descriptorUpdateTemplate,
                                                                        unwrapped_buffer);
    free(unwrapped_buffer);
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer commandBuffer,
                                                               VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate,
                                                               VkPipelineLayout layout, uint32_t set, const void *pData) {
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    uint64_t template_handle = reinterpret_cast<uint64_t &>(descriptorUpdateTemplate);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        descriptorUpdateTemplate = Unwrap(dev_data, descriptorUpdateTemplate);
        layout = Unwrap(dev_data, layout);
    }
    void *unwrapped_buffer = BuildUnwrappedUpdateTemplateBuffer(dev_data, template_handle, pData);
    dev_data->dispatch_table.CmdPushDescriptorSetWithTemplateKHR(commandBuffer, descriptorUpdateTemplate, layout, set,
                                                                         unwrapped_buffer);
    free(unwrapped_buffer);
}

#ifndef __ANDROID__
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceDisplayPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                                     VkDisplayPropertiesKHR *pProperties) {
    instance_layer_data *my_map_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);

    VkResult result = my_map_data->dispatch_table.GetPhysicalDeviceDisplayPropertiesKHR(
        physicalDevice, pPropertyCount, pProperties);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && pProperties) {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t idx0 = 0; idx0 < *pPropertyCount; ++idx0) {
            pProperties[idx0].display = WrapNew(my_map_data, pProperties[idx0].display);
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneSupportedDisplaysKHR(VkPhysicalDevice physicalDevice, uint32_t planeIndex,
                                                                   uint32_t *pDisplayCount, VkDisplayKHR *pDisplays) {
    instance_layer_data *my_map_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    VkResult result = my_map_data->dispatch_table.GetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex,
                                                                                                pDisplayCount, pDisplays);
    if (VK_SUCCESS == result) {
        if ((*pDisplayCount > 0) && pDisplays) {
            std::lock_guard<std::mutex> lock(global_lock);
            for (uint32_t i = 0; i < *pDisplayCount; i++) {
                // TODO: this looks like it really wants a /reverse/ mapping. What's going on here?
                auto it = my_map_data->unique_id_mapping.find(reinterpret_cast<const uint64_t &>(pDisplays[i]));
                assert(it != my_map_data->unique_id_mapping.end());
                pDisplays[i] = reinterpret_cast<VkDisplayKHR &>(it->second);
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayModePropertiesKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                                                           uint32_t *pPropertyCount, VkDisplayModePropertiesKHR *pProperties) {
    instance_layer_data *my_map_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        display = Unwrap(my_map_data, display);
    }

    VkResult result = my_map_data->dispatch_table.GetDisplayModePropertiesKHR(
        physicalDevice, display, pPropertyCount, pProperties);
    if (result == VK_SUCCESS && pProperties) {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t idx0 = 0; idx0 < *pPropertyCount; ++idx0) {
            pProperties[idx0].displayMode = WrapNew(my_map_data, pProperties[idx0].displayMode);
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode,
                                                              uint32_t planeIndex, VkDisplayPlaneCapabilitiesKHR *pCapabilities) {
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        mode = Unwrap(dev_data, mode);
    }
    VkResult result =
        dev_data->dispatch_table.GetDisplayPlaneCapabilitiesKHR(physicalDevice, mode, planeIndex, pCapabilities);
    return result;
}
#endif

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectTagEXT(VkDevice device, const VkDebugMarkerObjectTagInfoEXT *pTagInfo) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    auto local_tag_info = new safe_VkDebugMarkerObjectTagInfoEXT(pTagInfo);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        auto it = device_data->unique_id_mapping.find(reinterpret_cast<uint64_t &>(local_tag_info->object));
        if (it != device_data->unique_id_mapping.end()) {
            local_tag_info->object = it->second;
        }
    }
    VkResult result = device_data->dispatch_table.DebugMarkerSetObjectTagEXT(
        device, reinterpret_cast<VkDebugMarkerObjectTagInfoEXT *>(local_tag_info));
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectNameEXT(VkDevice device, const VkDebugMarkerObjectNameInfoEXT *pNameInfo) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    auto local_name_info = new safe_VkDebugMarkerObjectNameInfoEXT(pNameInfo);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        auto it = device_data->unique_id_mapping.find(reinterpret_cast<uint64_t &>(local_name_info->object));
        if (it != device_data->unique_id_mapping.end()) {
            local_name_info->object = it->second;
        }
    }
    VkResult result = device_data->dispatch_table.DebugMarkerSetObjectNameEXT(
        device, reinterpret_cast<VkDebugMarkerObjectNameInfoEXT *>(local_name_info));
    return result;
}

}  // namespace unique_objects

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                                      VkExtensionProperties *pProperties) {
    return unique_objects::EnumerateInstanceExtensionProperties(pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pCount,
                                                                                  VkLayerProperties *pProperties) {
    return unique_objects::EnumerateInstanceLayerProperties(pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                                                VkLayerProperties *pProperties) {
    assert(physicalDevice == VK_NULL_HANDLE);
    return unique_objects::EnumerateDeviceLayerProperties(VK_NULL_HANDLE, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice dev, const char *funcName) {
    return unique_objects::GetDeviceProcAddr(dev, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *funcName) {
    return unique_objects::GetInstanceProcAddr(instance, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                                    const char *pLayerName, uint32_t *pCount,
                                                                                    VkExtensionProperties *pProperties) {
    assert(physicalDevice == VK_NULL_HANDLE);
    return unique_objects::EnumerateDeviceExtensionProperties(VK_NULL_HANDLE, pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance,
                                                                                           const char *funcName) {
    return unique_objects::GetPhysicalDeviceProcAddr(instance, funcName);
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
        unique_objects::loader_layer_if_version = pVersionStruct->loaderLayerInterfaceVersion;
    } else if (pVersionStruct->loaderLayerInterfaceVersion > CURRENT_LOADER_LAYER_INTERFACE_VERSION) {
        pVersionStruct->loaderLayerInterfaceVersion = CURRENT_LOADER_LAYER_INTERFACE_VERSION;
    }

    return VK_SUCCESS;
}

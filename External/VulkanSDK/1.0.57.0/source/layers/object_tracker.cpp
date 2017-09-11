// *** THIS FILE IS GENERATED - DO NOT EDIT ***
// See object_tracker_generator.py for modifications


/***************************************************************************
 *
 * Copyright (c) 2015-2017 The Khronos Group Inc.
 * Copyright (c) 2015-2017 Valve Corporation
 * Copyright (c) 2015-2017 LunarG, Inc.
 * Copyright (c) 2015-2017 Google Inc.
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
 * Author: Mark Lobodzinski <mark@lunarg.com>
 *
 ****************************************************************************/


#include "object_tracker.h"

namespace object_tracker {

// ObjectTracker undestroyed objects validation function
void ReportUndestroyedObjects(VkDevice device, enum UNIQUE_VALIDATION_ERROR_CODE error_code) {
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeCommandBuffer, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeSemaphore, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeFence, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDeviceMemory, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeBuffer, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeImage, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeEvent, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeQueryPool, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeBufferView, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeImageView, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeShaderModule, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypePipelineCache, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypePipelineLayout, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeRenderPass, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypePipeline, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDescriptorSetLayout, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeSampler, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDescriptorPool, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDescriptorSet, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeFramebuffer, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeCommandPool, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeSurfaceKHR, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeSwapchainKHR, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDisplayKHR, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDisplayModeKHR, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDescriptorUpdateTemplateKHR, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeDebugReportCallbackEXT, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeObjectTableNVX, error_code);
    DeviceReportUndestroyedObjects(device, kVulkanObjectTypeIndirectCommandsLayoutNVX, error_code);
}



// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(
    const VkInstanceCreateInfo*                 pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkInstance*                                 pInstance);

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroyInstance(
    VkInstance                                  instance,
    const VkAllocationCallbacks*                pAllocator);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(
    VkInstance                                  instance,
    uint32_t*                                   pPhysicalDeviceCount,
    VkPhysicalDevice*                           pPhysicalDevices);

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceFeatures*                   pFeatures)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2c027a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceFeatures(physicalDevice, pFeatures);

}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkFormatProperties*                         pFormatProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2c427a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkImageType                                 type,
    VkImageTiling                               tiling,
    VkImageUsageFlags                           usage,
    VkImageCreateFlags                          flags,
    VkImageFormatProperties*                    pImageFormatProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2ca27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceImageFormatProperties(physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceProperties*                 pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2d627a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceProperties(physicalDevice, pProperties);

}

// Declare only
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pQueueFamilyPropertyCount,
    VkQueueFamilyProperties*                    pQueueFamilyProperties);

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceMemoryProperties*           pMemoryProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2ce27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceMemoryProperties(physicalDevice, pMemoryProperties);

}

// Declare only
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(
    VkInstance                                  instance,
    const char*                                 pName);

// Declare only
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(
    VkDevice                                    device,
    const char*                                 pName);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(
    VkPhysicalDevice                            physicalDevice,
    const VkDeviceCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDevice*                                   pDevice);

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroyDevice(
    VkDevice                                    device,
    const VkAllocationCallbacks*                pAllocator);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(
    VkPhysicalDevice                            physicalDevice,
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceLayerProperties(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties);

// Declare only
VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(
    VkDevice                                    device,
    uint32_t                                    queueFamilyIndex,
    uint32_t                                    queueIndex,
    VkQueue*                                    pQueue);

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(
    VkQueue                                     queue,
    uint32_t                                    submitCount,
    const VkSubmitInfo*                         pSubmits,
    VkFence                                     fence)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(queue, queue, kVulkanObjectTypeQueue, false, VALIDATION_ERROR_31a29c01, VALIDATION_ERROR_31a00009);
        if (pSubmits) {
            for (uint32_t index0 = 0; index0 < submitCount; ++index0) {
                if (pSubmits[index0].pWaitSemaphores) {
                    for (uint32_t index1 = 0; index1 < pSubmits[index0].waitSemaphoreCount; ++index1) {
                        skip |= ValidateObject(queue, pSubmits[index0].pWaitSemaphores[index1], kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_13c27601, VALIDATION_ERROR_13c00009);
                    }
                }
                if (pSubmits[index0].pCommandBuffers) {
                    for (uint32_t index1 = 0; index1 < pSubmits[index0].commandBufferCount; ++index1) {
                        skip |= ValidateObject(queue, pSubmits[index0].pCommandBuffers[index1], kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_13c11401, VALIDATION_ERROR_13c00009);
                    }
                }
                if (pSubmits[index0].pSignalSemaphores) {
                    for (uint32_t index1 = 0; index1 < pSubmits[index0].signalSemaphoreCount; ++index1) {
                        skip |= ValidateObject(queue, pSubmits[index0].pSignalSemaphores[index1], kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_13c23401, VALIDATION_ERROR_13c00009);
                    }
                }
            }
        }
        skip |= ValidateObject(queue, fence, kVulkanObjectTypeFence, true, VALIDATION_ERROR_31a08801, VALIDATION_ERROR_31a00009);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, queue)->QueueSubmit(queue, submitCount, pSubmits, fence);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(
    VkQueue                                     queue)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(queue, queue, kVulkanObjectTypeQueue, false, VALIDATION_ERROR_31c29c01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, queue)->QueueWaitIdle(queue);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(
    VkDevice                                    device)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_27005601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->DeviceWaitIdle(device);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(
    VkDevice                                    device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_16c05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pMemory, kVulkanObjectTypeDeviceMemory, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, true, VALIDATION_ERROR_2880c601, VALIDATION_ERROR_2880c607);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, memory, kVulkanObjectTypeDeviceMemory, pAllocator, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->FreeMemory(device, memory, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize                                offset,
    VkDeviceSize                                size,
    VkMemoryMapFlags                            flags,
    void**                                      ppData)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_31205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_3120c601, VALIDATION_ERROR_3120c607);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->MapMemory(device, memory, offset, size, flags, ppData);

    return result;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33605601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_3360c601, VALIDATION_ERROR_3360c607);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->UnmapMemory(device, memory);

}

VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28205601, VALIDATION_ERROR_UNDEFINED);
        if (pMemoryRanges) {
            for (uint32_t index0 = 0; index0 < memoryRangeCount; ++index0) {
                skip |= ValidateObject(device, pMemoryRanges[index0].memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_0c20c601, VALIDATION_ERROR_UNDEFINED);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->FlushMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_31005601, VALIDATION_ERROR_UNDEFINED);
        if (pMemoryRanges) {
            for (uint32_t index0 = 0; index0 < memoryRangeCount; ++index0) {
                skip |= ValidateObject(device, pMemoryRanges[index0].memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_0c20c601, VALIDATION_ERROR_UNDEFINED);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->InvalidateMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceMemoryCommitment(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize*                               pCommittedMemoryInBytes)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_29205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_2920c601, VALIDATION_ERROR_2920c607);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes);

}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkDeviceMemory                              memory,
    VkDeviceSize                                memoryOffset)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_17005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_17001a01, VALIDATION_ERROR_17001a07);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_1700c601, VALIDATION_ERROR_1700c607);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->BindBufferMemory(device, buffer, memory, memoryOffset);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(
    VkDevice                                    device,
    VkImage                                     image,
    VkDeviceMemory                              memory,
    VkDeviceSize                                memoryOffset)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_17405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1740a001, VALIDATION_ERROR_1740a007);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_1740c601, VALIDATION_ERROR_1740c607);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->BindImageMemory(device, image, memory, memoryOffset);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkMemoryRequirements*                       pMemoryRequirements)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28a05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_28a01a01, VALIDATION_ERROR_28a01a07);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetBufferMemoryRequirements(device, buffer, pMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    VkMemoryRequirements*                       pMemoryRequirements)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2a205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_2a20a001, VALIDATION_ERROR_2a20a007);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetImageMemoryRequirements(device, image, pMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements*            pSparseMemoryRequirements)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2a405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_2a40a001, VALIDATION_ERROR_2a40a007);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkImageType                                 type,
    VkSampleCountFlagBits                       samples,
    VkImageUsageFlags                           usage,
    VkImageTiling                               tiling,
    uint32_t*                                   pPropertyCount,
    VkSparseImageFormatProperties*              pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2de27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties);

}

VKAPI_ATTR VkResult VKAPI_CALL QueueBindSparse(
    VkQueue                                     queue,
    uint32_t                                    bindInfoCount,
    const VkBindSparseInfo*                     pBindInfo,
    VkFence                                     fence)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(queue, queue, kVulkanObjectTypeQueue, false, VALIDATION_ERROR_31629c01, VALIDATION_ERROR_31600009);
        if (pBindInfo) {
            for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
                if (pBindInfo[index0].pWaitSemaphores) {
                    for (uint32_t index1 = 0; index1 < pBindInfo[index0].waitSemaphoreCount; ++index1) {
                        skip |= ValidateObject(queue, pBindInfo[index0].pWaitSemaphores[index1], kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_01227601, VALIDATION_ERROR_01200009);
                    }
                }
                if (pBindInfo[index0].pBufferBinds) {
                    for (uint32_t index1 = 0; index1 < pBindInfo[index0].bufferBindCount; ++index1) {
                        skip |= ValidateObject(queue, pBindInfo[index0].pBufferBinds[index1].buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_12c01a01, VALIDATION_ERROR_UNDEFINED);
                        if (pBindInfo[index0].pBufferBinds[index1].pBinds) {
                            for (uint32_t index2 = 0; index2 < pBindInfo[index0].pBufferBinds[index1].bindCount; ++index2) {
                                skip |= ValidateObject(queue, pBindInfo[index0].pBufferBinds[index1].pBinds[index2].memory, kVulkanObjectTypeDeviceMemory, true, VALIDATION_ERROR_1340c601, VALIDATION_ERROR_UNDEFINED);
                            }
                        }
                    }
                }
                if (pBindInfo[index0].pImageOpaqueBinds) {
                    for (uint32_t index1 = 0; index1 < pBindInfo[index0].imageOpaqueBindCount; ++index1) {
                        skip |= ValidateObject(queue, pBindInfo[index0].pImageOpaqueBinds[index1].image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1320a001, VALIDATION_ERROR_UNDEFINED);
                        if (pBindInfo[index0].pImageOpaqueBinds[index1].pBinds) {
                            for (uint32_t index2 = 0; index2 < pBindInfo[index0].pImageOpaqueBinds[index1].bindCount; ++index2) {
                                skip |= ValidateObject(queue, pBindInfo[index0].pImageOpaqueBinds[index1].pBinds[index2].memory, kVulkanObjectTypeDeviceMemory, true, VALIDATION_ERROR_1340c601, VALIDATION_ERROR_UNDEFINED);
                            }
                        }
                    }
                }
                if (pBindInfo[index0].pImageBinds) {
                    for (uint32_t index1 = 0; index1 < pBindInfo[index0].imageBindCount; ++index1) {
                        skip |= ValidateObject(queue, pBindInfo[index0].pImageBinds[index1].image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1300a001, VALIDATION_ERROR_UNDEFINED);
                        if (pBindInfo[index0].pImageBinds[index1].pBinds) {
                            for (uint32_t index2 = 0; index2 < pBindInfo[index0].pImageBinds[index1].bindCount; ++index2) {
                                skip |= ValidateObject(queue, pBindInfo[index0].pImageBinds[index1].pBinds[index2].memory, kVulkanObjectTypeDeviceMemory, true, VALIDATION_ERROR_12e0c601, VALIDATION_ERROR_UNDEFINED);
                            }
                        }
                    }
                }
                if (pBindInfo[index0].pSignalSemaphores) {
                    for (uint32_t index1 = 0; index1 < pBindInfo[index0].signalSemaphoreCount; ++index1) {
                        skip |= ValidateObject(queue, pBindInfo[index0].pSignalSemaphores[index1], kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_01223401, VALIDATION_ERROR_01200009);
                    }
                }
            }
        }
        skip |= ValidateObject(queue, fence, kVulkanObjectTypeFence, true, VALIDATION_ERROR_31608801, VALIDATION_ERROR_31600009);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, queue)->QueueBindSparse(queue, bindInfoCount, pBindInfo, fence);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(
    VkDevice                                    device,
    const VkFenceCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_20405601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateFence(device, pCreateInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pFence, kVulkanObjectTypeFence, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(
    VkDevice                                    device,
    VkFence                                     fence,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_24e05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, fence, kVulkanObjectTypeFence, true, VALIDATION_ERROR_24e08801, VALIDATION_ERROR_24e08807);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, fence, kVulkanObjectTypeFence, pAllocator, VALIDATION_ERROR_24e008c2, VALIDATION_ERROR_24e008c4);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyFence(device, fence, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_32e05601, VALIDATION_ERROR_UNDEFINED);
        if (pFences) {
            for (uint32_t index0 = 0; index0 < fenceCount; ++index0) {
                skip |= ValidateObject(device, pFences[index0], kVulkanObjectTypeFence, false, VALIDATION_ERROR_32e17201, VALIDATION_ERROR_32e17207);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ResetFences(device, fenceCount, pFences);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(
    VkDevice                                    device,
    VkFence                                     fence)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2a005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, fence, kVulkanObjectTypeFence, false, VALIDATION_ERROR_2a008801, VALIDATION_ERROR_2a008807);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetFenceStatus(device, fence);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences,
    VkBool32                                    waitAll,
    uint64_t                                    timeout)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33e05601, VALIDATION_ERROR_UNDEFINED);
        if (pFences) {
            for (uint32_t index0 = 0; index0 < fenceCount; ++index0) {
                skip |= ValidateObject(device, pFences[index0], kVulkanObjectTypeFence, false, VALIDATION_ERROR_33e17201, VALIDATION_ERROR_33e17207);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->WaitForFences(device, fenceCount, pFences, waitAll, timeout);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSemaphore(
    VkDevice                                    device,
    const VkSemaphoreCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSemaphore*                                pSemaphore)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_22405601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pSemaphore, kVulkanObjectTypeSemaphore, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySemaphore(
    VkDevice                                    device,
    VkSemaphore                                 semaphore,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_26805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, semaphore, kVulkanObjectTypeSemaphore, true, VALIDATION_ERROR_2682b801, VALIDATION_ERROR_2682b807);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, semaphore, kVulkanObjectTypeSemaphore, pAllocator, VALIDATION_ERROR_268008e4, VALIDATION_ERROR_268008e6);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroySemaphore(device, semaphore, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateEvent(
    VkDevice                                    device,
    const VkEventCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkEvent*                                    pEvent)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_20205601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateEvent(device, pCreateInfo, pAllocator, pEvent);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pEvent, kVulkanObjectTypeEvent, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyEvent(
    VkDevice                                    device,
    VkEvent                                     event,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_24c05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, event, kVulkanObjectTypeEvent, true, VALIDATION_ERROR_24c07e01, VALIDATION_ERROR_24c07e07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, event, kVulkanObjectTypeEvent, pAllocator, VALIDATION_ERROR_24c008f4, VALIDATION_ERROR_24c008f6);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyEvent(device, event, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetEventStatus(
    VkDevice                                    device,
    VkEvent                                     event)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_29e05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, event, kVulkanObjectTypeEvent, false, VALIDATION_ERROR_29e07e01, VALIDATION_ERROR_29e07e07);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetEventStatus(device, event);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL SetEvent(
    VkDevice                                    device,
    VkEvent                                     event)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, event, kVulkanObjectTypeEvent, false, VALIDATION_ERROR_33007e01, VALIDATION_ERROR_33007e07);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->SetEvent(device, event);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetEvent(
    VkDevice                                    device,
    VkEvent                                     event)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_32c05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, event, kVulkanObjectTypeEvent, false, VALIDATION_ERROR_32c07e01, VALIDATION_ERROR_32c07e07);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ResetEvent(device, event);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(
    VkDevice                                    device,
    const VkQueryPoolCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkQueryPool*                                pQueryPool)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_21e05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pQueryPool, kVulkanObjectTypeQueryPool, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_26205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, queryPool, kVulkanObjectTypeQueryPool, true, VALIDATION_ERROR_26229801, VALIDATION_ERROR_26229807);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, queryPool, kVulkanObjectTypeQueryPool, pAllocator, VALIDATION_ERROR_26200634, VALIDATION_ERROR_26200636);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyQueryPool(device, queryPool, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResults(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    size_t                                      dataSize,
    void*                                       pData,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2fa05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, queryPool, kVulkanObjectTypeQueryPool, false, VALIDATION_ERROR_2fa29801, VALIDATION_ERROR_2fa29807);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(
    VkDevice                                    device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1ec05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pBuffer, kVulkanObjectTypeBuffer, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_23c05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, buffer, kVulkanObjectTypeBuffer, true, VALIDATION_ERROR_23c01a01, VALIDATION_ERROR_23c01a07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, buffer, kVulkanObjectTypeBuffer, pAllocator, VALIDATION_ERROR_23c00736, VALIDATION_ERROR_23c00738);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyBuffer(device, buffer, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateBufferView(
    VkDevice                                    device,
    const VkBufferViewCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBufferView*                               pView)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1ee05601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            skip |= ValidateObject(device, pCreateInfo->buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_01a01a01, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateBufferView(device, pCreateInfo, pAllocator, pView);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pView, kVulkanObjectTypeBufferView, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBufferView(
    VkDevice                                    device,
    VkBufferView                                bufferView,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_23e05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, bufferView, kVulkanObjectTypeBufferView, true, VALIDATION_ERROR_23e01c01, VALIDATION_ERROR_23e01c07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, bufferView, kVulkanObjectTypeBufferView, pAllocator, VALIDATION_ERROR_23e00752, VALIDATION_ERROR_23e00754);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyBufferView(device, bufferView, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(
    VkDevice                                    device,
    const VkImageCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImage*                                    pImage)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_20c05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateImage(device, pCreateInfo, pAllocator, pImage);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pImage, kVulkanObjectTypeImage, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(
    VkDevice                                    device,
    VkImage                                     image,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, image, kVulkanObjectTypeImage, true, VALIDATION_ERROR_2520a001, VALIDATION_ERROR_2520a007);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, image, kVulkanObjectTypeImage, pAllocator, VALIDATION_ERROR_252007d2, VALIDATION_ERROR_252007d4);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyImage(device, image, pAllocator);

}

VKAPI_ATTR void VKAPI_CALL GetImageSubresourceLayout(
    VkDevice                                    device,
    VkImage                                     image,
    const VkImageSubresource*                   pSubresource,
    VkSubresourceLayout*                        pLayout)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2a605601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_2a60a001, VALIDATION_ERROR_2a60a007);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetImageSubresourceLayout(device, image, pSubresource, pLayout);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(
    VkDevice                                    device,
    const VkImageViewCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImageView*                                pView)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_20e05601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            skip |= ValidateObject(device, pCreateInfo->image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_0ac0a001, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateImageView(device, pCreateInfo, pAllocator, pView);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pView, kVulkanObjectTypeImageView, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(
    VkDevice                                    device,
    VkImageView                                 imageView,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, imageView, kVulkanObjectTypeImageView, true, VALIDATION_ERROR_2540b001, VALIDATION_ERROR_2540b007);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, imageView, kVulkanObjectTypeImageView, pAllocator, VALIDATION_ERROR_25400806, VALIDATION_ERROR_25400808);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyImageView(device, imageView, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(
    VkDevice                                    device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_22605601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pShaderModule, kVulkanObjectTypeShaderModule, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(
    VkDevice                                    device,
    VkShaderModule                              shaderModule,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_26a05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, shaderModule, kVulkanObjectTypeShaderModule, true, VALIDATION_ERROR_26a2be01, VALIDATION_ERROR_26a2be07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, shaderModule, kVulkanObjectTypeShaderModule, pAllocator, VALIDATION_ERROR_26a00888, VALIDATION_ERROR_26a0088a);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyShaderModule(device, shaderModule, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(
    VkDevice                                    device,
    const VkPipelineCacheCreateInfo*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineCache*                            pPipelineCache)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_21a05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pPipelineCache, kVulkanObjectTypePipelineCache, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25e05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, pipelineCache, kVulkanObjectTypePipelineCache, true, VALIDATION_ERROR_25e28001, VALIDATION_ERROR_25e28007);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, pipelineCache, kVulkanObjectTypePipelineCache, pAllocator, VALIDATION_ERROR_25e00606, VALIDATION_ERROR_25e00608);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyPipelineCache(device, pipelineCache, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    size_t*                                     pDataSize,
    void*                                       pData)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2f805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, pipelineCache, kVulkanObjectTypePipelineCache, false, VALIDATION_ERROR_2f828001, VALIDATION_ERROR_2f828007);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetPipelineCacheData(device, pipelineCache, pDataSize, pData);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(
    VkDevice                                    device,
    VkPipelineCache                             dstCache,
    uint32_t                                    srcCacheCount,
    const VkPipelineCache*                      pSrcCaches)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_31405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, dstCache, kVulkanObjectTypePipelineCache, false, VALIDATION_ERROR_31406e01, VALIDATION_ERROR_31406e07);
        if (pSrcCaches) {
            for (uint32_t index0 = 0; index0 < srcCacheCount; ++index0) {
                skip |= ValidateObject(device, pSrcCaches[index0], kVulkanObjectTypePipelineCache, false, VALIDATION_ERROR_31423c01, VALIDATION_ERROR_31423c07);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->MergePipelineCaches(device, dstCache, srcCacheCount, pSrcCaches);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkGraphicsPipelineCreateInfo*         pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_20805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, pipelineCache, kVulkanObjectTypePipelineCache, true, VALIDATION_ERROR_20828001, VALIDATION_ERROR_20828007);
        if (pCreateInfos) {
            for (uint32_t index0 = 0; index0 < createInfoCount; ++index0) {
                if (pCreateInfos[index0].pStages) {
                    for (uint32_t index1 = 0; index1 < pCreateInfos[index0].stageCount; ++index1) {
                        skip |= ValidateObject(device, pCreateInfos[index0].pStages[index1].module, kVulkanObjectTypeShaderModule, false, VALIDATION_ERROR_1060d201, VALIDATION_ERROR_UNDEFINED);
                    }
                }
                skip |= ValidateObject(device, pCreateInfos[index0].layout, kVulkanObjectTypePipelineLayout, false, VALIDATION_ERROR_0960be01, VALIDATION_ERROR_09600009);
                skip |= ValidateObject(device, pCreateInfos[index0].renderPass, kVulkanObjectTypeRenderPass, false, VALIDATION_ERROR_0962ae01, VALIDATION_ERROR_09600009);
                skip |= ValidateObject(device, pCreateInfos[index0].basePipelineHandle, kVulkanObjectTypePipeline, true, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_09600009);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t index = 0; index < createInfoCount; index++) {
            CreateObject(device, pPipelines[index], kVulkanObjectTypePipeline, pAllocator);
        }
    }
    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkComputePipelineCreateInfo*          pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines);

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25c05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, pipeline, kVulkanObjectTypePipeline, true, VALIDATION_ERROR_25c27c01, VALIDATION_ERROR_25c27c07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, pipeline, kVulkanObjectTypePipeline, pAllocator, VALIDATION_ERROR_25c005fc, VALIDATION_ERROR_25c005fe);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyPipeline(device, pipeline, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(
    VkDevice                                    device,
    const VkPipelineLayoutCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineLayout*                           pPipelineLayout)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_21c05601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            if (pCreateInfo->pSetLayouts) {
                for (uint32_t index1 = 0; index1 < pCreateInfo->setLayoutCount; ++index1) {
                    skip |= ValidateObject(device, pCreateInfo->pSetLayouts[index1], kVulkanObjectTypeDescriptorSetLayout, false, VALIDATION_ERROR_0fe22c01, VALIDATION_ERROR_UNDEFINED);
                }
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pPipelineLayout, kVulkanObjectTypePipelineLayout, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(
    VkDevice                                    device,
    VkPipelineLayout                            pipelineLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_26005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, pipelineLayout, kVulkanObjectTypePipelineLayout, true, VALIDATION_ERROR_26028201, VALIDATION_ERROR_26028207);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, pipelineLayout, kVulkanObjectTypePipelineLayout, pAllocator, VALIDATION_ERROR_26000256, VALIDATION_ERROR_26000258);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyPipelineLayout(device, pipelineLayout, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(
    VkDevice                                    device,
    const VkSamplerCreateInfo*                  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSampler*                                  pSampler)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_22205601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateSampler(device, pCreateInfo, pAllocator, pSampler);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pSampler, kVulkanObjectTypeSampler, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(
    VkDevice                                    device,
    VkSampler                                   sampler,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_26605601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, sampler, kVulkanObjectTypeSampler, true, VALIDATION_ERROR_2662b201, VALIDATION_ERROR_2662b207);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, sampler, kVulkanObjectTypeSampler, pAllocator, VALIDATION_ERROR_26600876, VALIDATION_ERROR_26600878);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroySampler(device, sampler, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorSetLayout*                      pSetLayout)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1f805601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            if (pCreateInfo->pBindings) {
                for (uint32_t index1 = 0; index1 < pCreateInfo->bindingCount; ++index1) {
                    if (pCreateInfo->pBindings[index1].pImmutableSamplers) {
                        for (uint32_t index2 = 0; index2 < pCreateInfo->pBindings[index1].descriptorCount; ++index2) {
                            skip |= ValidateObject(device, pCreateInfo->pBindings[index1].pImmutableSamplers[index2], kVulkanObjectTypeSampler, true, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
                        }
                    }
                }
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pSetLayout, kVulkanObjectTypeDescriptorSetLayout, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(
    VkDevice                                    device,
    VkDescriptorSetLayout                       descriptorSetLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_24605601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, descriptorSetLayout, kVulkanObjectTypeDescriptorSetLayout, true, VALIDATION_ERROR_24604c01, VALIDATION_ERROR_24604c07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, descriptorSetLayout, kVulkanObjectTypeDescriptorSetLayout, pAllocator, VALIDATION_ERROR_24600238, VALIDATION_ERROR_2460023a);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(
    VkDevice                                    device,
    const VkDescriptorPoolCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorPool*                           pDescriptorPool)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1f605601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pDescriptorPool, kVulkanObjectTypeDescriptorPool, pAllocator);
    }
    return result;
}

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    const VkAllocationCallbacks*                pAllocator);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    VkDescriptorPoolResetFlags                  flags);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(
    VkDevice                                    device,
    const VkDescriptorSetAllocateInfo*          pAllocateInfo,
    VkDescriptorSet*                            pDescriptorSets);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets);

// Declare only
VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(
    VkDevice                                    device,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites,
    uint32_t                                    descriptorCopyCount,
    const VkCopyDescriptorSet*                  pDescriptorCopies);

VKAPI_ATTR VkResult VKAPI_CALL CreateFramebuffer(
    VkDevice                                    device,
    const VkFramebufferCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFramebuffer*                              pFramebuffer)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_20605601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            skip |= ValidateObject(device, pCreateInfo->renderPass, kVulkanObjectTypeRenderPass, false, VALIDATION_ERROR_0942ae01, VALIDATION_ERROR_09400009);
            if (pCreateInfo->pAttachments) {
                for (uint32_t index1 = 0; index1 < pCreateInfo->attachmentCount; ++index1) {
                    skip |= ValidateObject(device, pCreateInfo->pAttachments[index1], kVulkanObjectTypeImageView, false, VALIDATION_ERROR_0940f201, VALIDATION_ERROR_09400009);
                }
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pFramebuffer, kVulkanObjectTypeFramebuffer, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFramebuffer(
    VkDevice                                    device,
    VkFramebuffer                               framebuffer,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, framebuffer, kVulkanObjectTypeFramebuffer, true, VALIDATION_ERROR_25009401, VALIDATION_ERROR_25009407);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, framebuffer, kVulkanObjectTypeFramebuffer, pAllocator, VALIDATION_ERROR_250006fa, VALIDATION_ERROR_250006fc);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyFramebuffer(device, framebuffer, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass(
    VkDevice                                    device,
    const VkRenderPassCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_22005601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pRenderPass, kVulkanObjectTypeRenderPass, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyRenderPass(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_26405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, renderPass, kVulkanObjectTypeRenderPass, true, VALIDATION_ERROR_2642ae01, VALIDATION_ERROR_2642ae07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, renderPass, kVulkanObjectTypeRenderPass, pAllocator, VALIDATION_ERROR_264006d4, VALIDATION_ERROR_264006d6);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyRenderPass(device, renderPass, pAllocator);

}

VKAPI_ATTR void VKAPI_CALL GetRenderAreaGranularity(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    VkExtent2D*                                 pGranularity)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, renderPass, kVulkanObjectTypeRenderPass, false, VALIDATION_ERROR_3002ae01, VALIDATION_ERROR_3002ae07);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetRenderAreaGranularity(device, renderPass, pGranularity);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(
    VkDevice                                    device,
    const VkCommandPoolCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkCommandPool*                              pCommandPool)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1f005601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pCommandPool, kVulkanObjectTypeCommandPool, pAllocator);
    }
    return result;
}

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    const VkAllocationCallbacks*                pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolResetFlags                     flags)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_32805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, commandPool, kVulkanObjectTypeCommandPool, false, VALIDATION_ERROR_32802801, VALIDATION_ERROR_32802807);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ResetCommandPool(device, commandPool, flags);

    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(
    VkDevice                                    device,
    const VkCommandBufferAllocateInfo*          pAllocateInfo,
    VkCommandBuffer*                            pCommandBuffers);

// Declare only
VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    const VkCommandBufferBeginInfo*             pBeginInfo);

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(
    VkCommandBuffer                             commandBuffer)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_27402401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, commandBuffer)->EndCommandBuffer(commandBuffer);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    VkCommandBufferResetFlags                   flags)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_32602401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, commandBuffer)->ResetCommandBuffer(commandBuffer, flags);

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipeline                                  pipeline)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18002401, VALIDATION_ERROR_18000009);
        skip |= ValidateObject(commandBuffer, pipeline, kVulkanObjectTypePipeline, false, VALIDATION_ERROR_18027c01, VALIDATION_ERROR_18000009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);

}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewport*                           pViewports)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1e002401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetViewport(commandBuffer, firstViewport, viewportCount, pViewports);

}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstScissor,
    uint32_t                                    scissorCount,
    const VkRect2D*                             pScissors)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1d802401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetScissor(commandBuffer, firstScissor, scissorCount, pScissors);

}

VKAPI_ATTR void VKAPI_CALL CmdSetLineWidth(
    VkCommandBuffer                             commandBuffer,
    float                                       lineWidth)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1d602401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetLineWidth(commandBuffer, lineWidth);

}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBias(
    VkCommandBuffer                             commandBuffer,
    float                                       depthBiasConstantFactor,
    float                                       depthBiasClamp,
    float                                       depthBiasSlopeFactor)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1cc02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetDepthBias(commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);

}

VKAPI_ATTR void VKAPI_CALL CmdSetBlendConstants(
    VkCommandBuffer                             commandBuffer,
    const float                                 blendConstants[4])
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1ca02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetBlendConstants(commandBuffer, blendConstants);

}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBounds(
    VkCommandBuffer                             commandBuffer,
    float                                       minDepthBounds,
    float                                       maxDepthBounds)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1ce02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetDepthBounds(commandBuffer, minDepthBounds, maxDepthBounds);

}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilCompareMask(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    compareMask)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1da02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetStencilCompareMask(commandBuffer, faceMask, compareMask);

}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilWriteMask(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    writeMask)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1de02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetStencilWriteMask(commandBuffer, faceMask, writeMask);

}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilReference(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    reference)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1dc02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetStencilReference(commandBuffer, faceMask, reference);

}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    firstSet,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets,
    uint32_t                                    dynamicOffsetCount,
    const uint32_t*                             pDynamicOffsets)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_17c02401, VALIDATION_ERROR_17c00009);
        skip |= ValidateObject(commandBuffer, layout, kVulkanObjectTypePipelineLayout, false, VALIDATION_ERROR_17c0be01, VALIDATION_ERROR_17c00009);
        if (pDescriptorSets) {
            for (uint32_t index0 = 0; index0 < descriptorSetCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pDescriptorSets[index0], kVulkanObjectTypeDescriptorSet, false, VALIDATION_ERROR_17c13001, VALIDATION_ERROR_17c00009);
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);

}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkIndexType                                 indexType)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_17e02401, VALIDATION_ERROR_17e00009);
        skip |= ValidateObject(commandBuffer, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_17e01a01, VALIDATION_ERROR_17e00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);

}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18202401, VALIDATION_ERROR_18200009);
        if (pBuffers) {
            for (uint32_t index0 = 0; index0 < bindingCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pBuffers[index0], kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_18210601, VALIDATION_ERROR_18200009);
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);

}

VKAPI_ATTR void VKAPI_CALL CmdDraw(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    vertexCount,
    uint32_t                                    instanceCount,
    uint32_t                                    firstVertex,
    uint32_t                                    firstInstance)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1a202401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);

}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    indexCount,
    uint32_t                                    instanceCount,
    uint32_t                                    firstIndex,
    int32_t                                     vertexOffset,
    uint32_t                                    firstInstance)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1a402401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDrawIndexed(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);

}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1aa02401, VALIDATION_ERROR_1aa00009);
        skip |= ValidateObject(commandBuffer, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1aa01a01, VALIDATION_ERROR_1aa00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDrawIndirect(commandBuffer, buffer, offset, drawCount, stride);

}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1a602401, VALIDATION_ERROR_1a600009);
        skip |= ValidateObject(commandBuffer, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1a601a01, VALIDATION_ERROR_1a600009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDrawIndexedIndirect(commandBuffer, buffer, offset, drawCount, stride);

}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    groupCountX,
    uint32_t                                    groupCountY,
    uint32_t                                    groupCountZ)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19c02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);

}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1a002401, VALIDATION_ERROR_1a000009);
        skip |= ValidateObject(commandBuffer, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1a001a01, VALIDATION_ERROR_1a000009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDispatchIndirect(commandBuffer, buffer, offset);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    srcBuffer,
    VkBuffer                                    dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferCopy*                         pRegions)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18c02401, VALIDATION_ERROR_18c00009);
        skip |= ValidateObject(commandBuffer, srcBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_18c2c801, VALIDATION_ERROR_18c00009);
        skip |= ValidateObject(commandBuffer, dstBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_18c06c01, VALIDATION_ERROR_18c00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkImageCopy*                          pRegions)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19002401, VALIDATION_ERROR_19000009);
        skip |= ValidateObject(commandBuffer, srcImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1902ce01, VALIDATION_ERROR_19000009);
        skip |= ValidateObject(commandBuffer, dstImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_19007201, VALIDATION_ERROR_19000009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkImageBlit*                          pRegions,
    VkFilter                                    filter)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18402401, VALIDATION_ERROR_18400009);
        skip |= ValidateObject(commandBuffer, srcImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1842ce01, VALIDATION_ERROR_18400009);
        skip |= ValidateObject(commandBuffer, dstImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_18407201, VALIDATION_ERROR_18400009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    srcBuffer,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18e02401, VALIDATION_ERROR_18e00009);
        skip |= ValidateObject(commandBuffer, srcBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_18e2c801, VALIDATION_ERROR_18e00009);
        skip |= ValidateObject(commandBuffer, dstImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_18e07201, VALIDATION_ERROR_18e00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkBuffer                                    dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19202401, VALIDATION_ERROR_19200009);
        skip |= ValidateObject(commandBuffer, srcImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1922ce01, VALIDATION_ERROR_19200009);
        skip |= ValidateObject(commandBuffer, dstBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_19206c01, VALIDATION_ERROR_19200009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                dataSize,
    const void*                                 pData)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1e402401, VALIDATION_ERROR_1e400009);
        skip |= ValidateObject(commandBuffer, dstBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1e406c01, VALIDATION_ERROR_1e400009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData);

}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                size,
    uint32_t                                    data)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1b402401, VALIDATION_ERROR_1b400009);
        skip |= ValidateObject(commandBuffer, dstBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1b406c01, VALIDATION_ERROR_1b400009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);

}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearColorValue*                    pColor,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18802401, VALIDATION_ERROR_18800009);
        skip |= ValidateObject(commandBuffer, image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1880a001, VALIDATION_ERROR_18800009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);

}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearDepthStencilValue*             pDepthStencil,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18a02401, VALIDATION_ERROR_18a00009);
        skip |= ValidateObject(commandBuffer, image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_18a0a001, VALIDATION_ERROR_18a00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);

}

VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    attachmentCount,
    const VkClearAttachment*                    pAttachments,
    uint32_t                                    rectCount,
    const VkClearRect*                          pRects)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_18602401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdClearAttachments(commandBuffer, attachmentCount, pAttachments, rectCount, pRects);

}

VKAPI_ATTR void VKAPI_CALL CmdResolveImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkImageResolve*                       pRegions)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1c802401, VALIDATION_ERROR_1c800009);
        skip |= ValidateObject(commandBuffer, srcImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1c82ce01, VALIDATION_ERROR_1c800009);
        skip |= ValidateObject(commandBuffer, dstImage, kVulkanObjectTypeImage, false, VALIDATION_ERROR_1c807201, VALIDATION_ERROR_1c800009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdSetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1d402401, VALIDATION_ERROR_1d400009);
        skip |= ValidateObject(commandBuffer, event, kVulkanObjectTypeEvent, false, VALIDATION_ERROR_1d407e01, VALIDATION_ERROR_1d400009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetEvent(commandBuffer, event, stageMask);

}

VKAPI_ATTR void VKAPI_CALL CmdResetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1c402401, VALIDATION_ERROR_1c400009);
        skip |= ValidateObject(commandBuffer, event, kVulkanObjectTypeEvent, false, VALIDATION_ERROR_1c407e01, VALIDATION_ERROR_1c400009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdResetEvent(commandBuffer, event, stageMask);

}

VKAPI_ATTR void VKAPI_CALL CmdWaitEvents(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    eventCount,
    const VkEvent*                              pEvents,
    VkPipelineStageFlags                        srcStageMask,
    VkPipelineStageFlags                        dstStageMask,
    uint32_t                                    memoryBarrierCount,
    const VkMemoryBarrier*                      pMemoryBarriers,
    uint32_t                                    bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier*                pBufferMemoryBarriers,
    uint32_t                                    imageMemoryBarrierCount,
    const VkImageMemoryBarrier*                 pImageMemoryBarriers)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1e602401, VALIDATION_ERROR_1e600009);
        if (pEvents) {
            for (uint32_t index0 = 0; index0 < eventCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pEvents[index0], kVulkanObjectTypeEvent, false, VALIDATION_ERROR_1e616001, VALIDATION_ERROR_1e600009);
            }
        }
        if (pBufferMemoryBarriers) {
            for (uint32_t index0 = 0; index0 < bufferMemoryBarrierCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pBufferMemoryBarriers[index0].buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_01801a01, VALIDATION_ERROR_UNDEFINED);
            }
        }
        if (pImageMemoryBarriers) {
            for (uint32_t index0 = 0; index0 < imageMemoryBarrierCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pImageMemoryBarriers[index0].image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_0a00a001, VALIDATION_ERROR_UNDEFINED);
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdWaitEvents(commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlags                        srcStageMask,
    VkPipelineStageFlags                        dstStageMask,
    VkDependencyFlags                           dependencyFlags,
    uint32_t                                    memoryBarrierCount,
    const VkMemoryBarrier*                      pMemoryBarriers,
    uint32_t                                    bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier*                pBufferMemoryBarriers,
    uint32_t                                    imageMemoryBarrierCount,
    const VkImageMemoryBarrier*                 pImageMemoryBarriers)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1b802401, VALIDATION_ERROR_UNDEFINED);
        if (pBufferMemoryBarriers) {
            for (uint32_t index0 = 0; index0 < bufferMemoryBarrierCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pBufferMemoryBarriers[index0].buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_01801a01, VALIDATION_ERROR_UNDEFINED);
            }
        }
        if (pImageMemoryBarriers) {
            for (uint32_t index0 = 0; index0 < imageMemoryBarrierCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pImageMemoryBarriers[index0].image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_0a00a001, VALIDATION_ERROR_UNDEFINED);
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_17802401, VALIDATION_ERROR_17800009);
        skip |= ValidateObject(commandBuffer, queryPool, kVulkanObjectTypeQueryPool, false, VALIDATION_ERROR_17829801, VALIDATION_ERROR_17800009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBeginQuery(commandBuffer, queryPool, query, flags);

}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1ae02401, VALIDATION_ERROR_1ae00009);
        skip |= ValidateObject(commandBuffer, queryPool, kVulkanObjectTypeQueryPool, false, VALIDATION_ERROR_1ae29801, VALIDATION_ERROR_1ae00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdEndQuery(commandBuffer, queryPool, query);

}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1c602401, VALIDATION_ERROR_1c600009);
        skip |= ValidateObject(commandBuffer, queryPool, kVulkanObjectTypeQueryPool, false, VALIDATION_ERROR_1c629801, VALIDATION_ERROR_1c600009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount);

}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1e802401, VALIDATION_ERROR_1e800009);
        skip |= ValidateObject(commandBuffer, queryPool, kVulkanObjectTypeQueryPool, false, VALIDATION_ERROR_1e829801, VALIDATION_ERROR_1e800009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, query);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyQueryPoolResults(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19402401, VALIDATION_ERROR_19400009);
        skip |= ValidateObject(commandBuffer, queryPool, kVulkanObjectTypeQueryPool, false, VALIDATION_ERROR_19429801, VALIDATION_ERROR_19400009);
        skip |= ValidateObject(commandBuffer, dstBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_19406c01, VALIDATION_ERROR_19400009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);

}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(
    VkCommandBuffer                             commandBuffer,
    VkPipelineLayout                            layout,
    VkShaderStageFlags                          stageFlags,
    uint32_t                                    offset,
    uint32_t                                    size,
    const void*                                 pValues)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1bc02401, VALIDATION_ERROR_1bc00009);
        skip |= ValidateObject(commandBuffer, layout, kVulkanObjectTypePipelineLayout, false, VALIDATION_ERROR_1bc0be01, VALIDATION_ERROR_1bc00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);

}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(
    VkCommandBuffer                             commandBuffer,
    const VkRenderPassBeginInfo*                pRenderPassBegin,
    VkSubpassContents                           contents)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_17a02401, VALIDATION_ERROR_UNDEFINED);
        if (pRenderPassBegin) {
            skip |= ValidateObject(commandBuffer, pRenderPassBegin->renderPass, kVulkanObjectTypeRenderPass, false, VALIDATION_ERROR_1202ae01, VALIDATION_ERROR_12000009);
            skip |= ValidateObject(commandBuffer, pRenderPassBegin->framebuffer, kVulkanObjectTypeFramebuffer, false, VALIDATION_ERROR_12009401, VALIDATION_ERROR_12000009);
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);

}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass(
    VkCommandBuffer                             commandBuffer,
    VkSubpassContents                           contents)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1b602401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdNextSubpass(commandBuffer, contents);

}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(
    VkCommandBuffer                             commandBuffer)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1b002401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdEndRenderPass(commandBuffer);

}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1b202401, VALIDATION_ERROR_1b200009);
        if (pCommandBuffers) {
            for (uint32_t index0 = 0; index0 < commandBufferCount; ++index0) {
                skip |= ValidateObject(commandBuffer, pCommandBuffers[index0], kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1b211401, VALIDATION_ERROR_1b200009);
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers);

}

VKAPI_ATTR void VKAPI_CALL DestroySurfaceKHR(
    VkInstance                                  instance,
    VkSurfaceKHR                                surface,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_26c0bc01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(instance, surface, kVulkanObjectTypeSurfaceKHR, true, VALIDATION_ERROR_26c2ec01, VALIDATION_ERROR_26c2ec07);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(instance, surface, kVulkanObjectTypeSurfaceKHR, pAllocator, VALIDATION_ERROR_26c009e6, VALIDATION_ERROR_26c009e8);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, instance)->DestroySurfaceKHR(instance, surface, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    VkSurfaceKHR                                surface,
    VkBool32*                                   pSupported)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2ee27a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2ee2ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    VkSurfaceCapabilitiesKHR*                   pSurfaceCapabilities)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2e627a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2e62ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pSurfaceFormatCount,
    VkSurfaceFormatKHR*                         pSurfaceFormats)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2ea27a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2ea2ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pPresentModeCount,
    VkPresentModeKHR*                           pPresentModes)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2ec27a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2ec2ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(
    VkDevice                                    device,
    const VkSwapchainCreateInfoKHR*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSwapchainKHR*                             pSwapchain)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_22a05601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            skip |= ValidateObject(device, pCreateInfo->surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_1462ec01, VALIDATION_ERROR_UNDEFINED);
            skip |= ValidateObject(device, pCreateInfo->oldSwapchain, kVulkanObjectTypeSwapchainKHR, true, VALIDATION_ERROR_1460de01, VALIDATION_ERROR_1460de07);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pSwapchain, kVulkanObjectTypeSwapchainKHR, pAllocator);
    }
    return result;
}

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    const VkAllocationCallbacks*                pAllocator);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint32_t*                                   pSwapchainImageCount,
    VkImage*                                    pSwapchainImages);

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImageKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint64_t                                    timeout,
    VkSemaphore                                 semaphore,
    VkFence                                     fence,
    uint32_t*                                   pImageIndex)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_16405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_1642f001, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, semaphore, kVulkanObjectTypeSemaphore, true, VALIDATION_ERROR_1642b801, VALIDATION_ERROR_1642b807);
        skip |= ValidateObject(device, fence, kVulkanObjectTypeFence, true, VALIDATION_ERROR_16408801, VALIDATION_ERROR_16408807);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(
    VkQueue                                     queue,
    const VkPresentInfoKHR*                     pPresentInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(queue, queue, kVulkanObjectTypeQueue, false, VALIDATION_ERROR_31829c01, VALIDATION_ERROR_UNDEFINED);
        if (pPresentInfo) {
            if (pPresentInfo->pWaitSemaphores) {
                for (uint32_t index1 = 0; index1 < pPresentInfo->waitSemaphoreCount; ++index1) {
                    skip |= ValidateObject(queue, pPresentInfo->pWaitSemaphores[index1], kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_11227601, VALIDATION_ERROR_UNDEFINED);
                }
            }
            if (pPresentInfo->pSwapchains) {
                for (uint32_t index1 = 0; index1 < pPresentInfo->swapchainCount; ++index1) {
                    skip |= ValidateObject(queue, pPresentInfo->pSwapchains[index1], kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_11225801, VALIDATION_ERROR_UNDEFINED);
                }
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, queue)->QueuePresentKHR(queue, pPresentInfo);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceDisplayPropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkDisplayPropertiesKHR*                     pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2b827a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, pPropertyCount, pProperties);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceDisplayPlanePropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkDisplayPlanePropertiesKHR*                pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2b627a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, pPropertyCount, pProperties);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneSupportedDisplaysKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    planeIndex,
    uint32_t*                                   pDisplayCount,
    VkDisplayKHR*                               pDisplays)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_29c27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, pDisplayCount, pDisplays);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayModePropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display,
    uint32_t*                                   pPropertyCount,
    VkDisplayModePropertiesKHR*                 pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_29827a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, display, kVulkanObjectTypeDisplayKHR, false, VALIDATION_ERROR_29806001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetDisplayModePropertiesKHR(physicalDevice, display, pPropertyCount, pProperties);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayModeKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display,
    const VkDisplayModeCreateInfoKHR*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDisplayModeKHR*                           pMode)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_1fe27a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, display, kVulkanObjectTypeDisplayKHR, false, VALIDATION_ERROR_1fe06001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->CreateDisplayModeKHR(physicalDevice, display, pCreateInfo, pAllocator, pMode);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(physicalDevice, *pMode, kVulkanObjectTypeDisplayModeKHR, pAllocator);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayModeKHR                            mode,
    uint32_t                                    planeIndex,
    VkDisplayPlaneCapabilitiesKHR*              pCapabilities)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_29a27a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, mode, kVulkanObjectTypeDisplayModeKHR, false, VALIDATION_ERROR_29a0ce01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetDisplayPlaneCapabilitiesKHR(physicalDevice, mode, planeIndex, pCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayPlaneSurfaceKHR(
    VkInstance                                  instance,
    const VkDisplaySurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2000bc01, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            skip |= ValidateObject(instance, pCreateInfo->displayMode, kVulkanObjectTypeDisplayModeKHR, false, VALIDATION_ERROR_07806401, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateDisplayPlaneSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSharedSwapchainsKHR(
    VkDevice                                    device,
    uint32_t                                    swapchainCount,
    const VkSwapchainCreateInfoKHR*             pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkSwapchainKHR*                             pSwapchains)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_22805601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfos) {
            for (uint32_t index0 = 0; index0 < swapchainCount; ++index0) {
                skip |= ValidateObject(device, pCreateInfos[index0].surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_1462ec01, VALIDATION_ERROR_UNDEFINED);
                skip |= ValidateObject(device, pCreateInfos[index0].oldSwapchain, kVulkanObjectTypeSwapchainKHR, true, VALIDATION_ERROR_1460de01, VALIDATION_ERROR_1460de07);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateSharedSwapchainsKHR(device, swapchainCount, pCreateInfos, pAllocator, pSwapchains);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t index = 0; index < swapchainCount; index++) {
            CreateObject(device, pSwapchains[index], kVulkanObjectTypeSwapchainKHR, pAllocator);
        }
    }
    return result;
}

#ifdef VK_USE_PLATFORM_XLIB_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateXlibSurfaceKHR(
    VkInstance                                  instance,
    const VkXlibSurfaceCreateInfoKHR*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2340bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateXlibSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_XLIB_KHR

#ifdef VK_USE_PLATFORM_XLIB_KHR

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceXlibPresentationSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    Display*                                    dpy,
    VisualID                                    visualID)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2f627a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkBool32 result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceXlibPresentationSupportKHR(physicalDevice, queueFamilyIndex, dpy, visualID);

    return result;
}
#endif // VK_USE_PLATFORM_XLIB_KHR

#ifdef VK_USE_PLATFORM_XCB_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateXcbSurfaceKHR(
    VkInstance                                  instance,
    const VkXcbSurfaceCreateInfoKHR*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2320bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateXcbSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_XCB_KHR

#ifdef VK_USE_PLATFORM_XCB_KHR

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceXcbPresentationSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    xcb_connection_t*                           connection,
    xcb_visualid_t                              visual_id)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2f427a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkBool32 result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceXcbPresentationSupportKHR(physicalDevice, queueFamilyIndex, connection, visual_id);

    return result;
}
#endif // VK_USE_PLATFORM_XCB_KHR

#ifdef VK_USE_PLATFORM_WAYLAND_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateWaylandSurfaceKHR(
    VkInstance                                  instance,
    const VkWaylandSurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_22e0bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateWaylandSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_WAYLAND_KHR

#ifdef VK_USE_PLATFORM_WAYLAND_KHR

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceWaylandPresentationSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    struct wl_display*                          display)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2f027a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkBool32 result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceWaylandPresentationSupportKHR(physicalDevice, queueFamilyIndex, display);

    return result;
}
#endif // VK_USE_PLATFORM_WAYLAND_KHR

#ifdef VK_USE_PLATFORM_MIR_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateMirSurfaceKHR(
    VkInstance                                  instance,
    const VkMirSurfaceCreateInfoKHR*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2160bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateMirSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_MIR_KHR

#ifdef VK_USE_PLATFORM_MIR_KHR

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceMirPresentationSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    MirConnection*                              connection)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2d227a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkBool32 result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceMirPresentationSupportKHR(physicalDevice, queueFamilyIndex, connection);

    return result;
}
#endif // VK_USE_PLATFORM_MIR_KHR

#ifdef VK_USE_PLATFORM_ANDROID_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateAndroidSurfaceKHR(
    VkInstance                                  instance,
    const VkAndroidSurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_1ea0bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateAndroidSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_ANDROID_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateWin32SurfaceKHR(
    VkInstance                                  instance,
    const VkWin32SurfaceCreateInfoKHR*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2300bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceWin32PresentationSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2f227a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkBool32 result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, queueFamilyIndex);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures2KHR(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceFeatures2KHR*               pFeatures)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2c227a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceFeatures2KHR(physicalDevice, pFeatures);

}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceProperties2KHR*             pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2d827a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceProperties2KHR(physicalDevice, pProperties);

}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkFormatProperties2KHR*                     pFormatProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2c627a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceFormatProperties2KHR(physicalDevice, format, pFormatProperties);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceImageFormatProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceImageFormatInfo2KHR*  pImageFormatInfo,
    VkImageFormatProperties2KHR*                pImageFormatProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2cc27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceImageFormatProperties2KHR(physicalDevice, pImageFormatInfo, pImageFormatProperties);

    return result;
}

// Declare only
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pQueueFamilyPropertyCount,
    VkQueueFamilyProperties2KHR*                pQueueFamilyProperties);

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceMemoryProperties2KHR*       pMemoryProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2d027a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceMemoryProperties2KHR(physicalDevice, pMemoryProperties);

}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceSparseImageFormatProperties2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSparseImageFormatInfo2KHR* pFormatInfo,
    uint32_t*                                   pPropertyCount,
    VkSparseImageFormatProperties2KHR*          pProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2e027a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSparseImageFormatProperties2KHR(physicalDevice, pFormatInfo, pPropertyCount, pProperties);

}

VKAPI_ATTR void VKAPI_CALL TrimCommandPoolKHR(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolTrimFlagsKHR                   flags)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, commandPool, kVulkanObjectTypeCommandPool, false, VALIDATION_ERROR_33402801, VALIDATION_ERROR_33402807);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->TrimCommandPoolKHR(device, commandPool, flags);

}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceExternalBufferPropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceExternalBufferInfoKHR* pExternalBufferInfo,
    VkExternalBufferPropertiesKHR*              pExternalBufferProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2ba27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceExternalBufferPropertiesKHR(physicalDevice, pExternalBufferInfo, pExternalBufferProperties);

}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleKHR(
    VkDevice                                    device,
    const VkMemoryGetWin32HandleInfoKHR*        pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2ae05601, VALIDATION_ERROR_UNDEFINED);
        if (pGetWin32HandleInfo) {
            skip |= ValidateObject(device, pGetWin32HandleInfo->memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_3b40c601, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetMemoryWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandlePropertiesKHR(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBitsKHR       handleType,
    HANDLE                                      handle,
    VkMemoryWin32HandlePropertiesKHR*           pMemoryWin32HandleProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2b205601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetMemoryWin32HandlePropertiesKHR(device, handleType, handle, pMemoryWin32HandleProperties);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdKHR(
    VkDevice                                    device,
    const VkMemoryGetFdInfoKHR*                 pGetFdInfo,
    int*                                        pFd)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2aa05601, VALIDATION_ERROR_UNDEFINED);
        if (pGetFdInfo) {
            skip |= ValidateObject(device, pGetFdInfo->memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_3b20c601, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetMemoryFdKHR(device, pGetFdInfo, pFd);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdPropertiesKHR(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBitsKHR       handleType,
    int                                         fd,
    VkMemoryFdPropertiesKHR*                    pMemoryFdProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2ac05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetMemoryFdPropertiesKHR(device, handleType, fd, pMemoryFdProperties);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceExternalSemaphorePropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceExternalSemaphoreInfoKHR* pExternalSemaphoreInfo,
    VkExternalSemaphorePropertiesKHR*           pExternalSemaphoreProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2be27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceExternalSemaphorePropertiesKHR(physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties);

}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkImportSemaphoreWin32HandleInfoKHR*  pImportSemaphoreWin32HandleInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30e05601, VALIDATION_ERROR_UNDEFINED);
        if (pImportSemaphoreWin32HandleInfo) {
            skip |= ValidateObject(device, pImportSemaphoreWin32HandleInfo->semaphore, kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_0b62b801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ImportSemaphoreWin32HandleKHR(device, pImportSemaphoreWin32HandleInfo);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkSemaphoreGetWin32HandleInfoKHR*     pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30405601, VALIDATION_ERROR_UNDEFINED);
        if (pGetWin32HandleInfo) {
            skip |= ValidateObject(device, pGetWin32HandleInfo->semaphore, kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_3b02b801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetSemaphoreWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreFdKHR(
    VkDevice                                    device,
    const VkImportSemaphoreFdInfoKHR*           pImportSemaphoreFdInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30c05601, VALIDATION_ERROR_UNDEFINED);
        if (pImportSemaphoreFdInfo) {
            skip |= ValidateObject(device, pImportSemaphoreFdInfo->semaphore, kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_0b42b801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ImportSemaphoreFdKHR(device, pImportSemaphoreFdInfo);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreFdKHR(
    VkDevice                                    device,
    const VkSemaphoreGetFdInfoKHR*              pGetFdInfo,
    int*                                        pFd)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30205601, VALIDATION_ERROR_UNDEFINED);
        if (pGetFdInfo) {
            skip |= ValidateObject(device, pGetFdInfo->semaphore, kVulkanObjectTypeSemaphore, false, VALIDATION_ERROR_3ae2b801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetSemaphoreFdKHR(device, pGetFdInfo, pFd);

    return result;
}

// Declare only
VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites);

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorUpdateTemplateKHR*              pDescriptorUpdateTemplate)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1fa05601, VALIDATION_ERROR_UNDEFINED);
        if (pCreateInfo) {
            skip |= ValidateObject(device, pCreateInfo->descriptorSetLayout, kVulkanObjectTypeDescriptorSetLayout, true, VALIDATION_ERROR_05204c01, VALIDATION_ERROR_05200009);
            skip |= ValidateObject(device, pCreateInfo->pipelineLayout, kVulkanObjectTypePipelineLayout, true, VALIDATION_ERROR_05228201, VALIDATION_ERROR_05200009);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateDescriptorUpdateTemplateKHR(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pDescriptorUpdateTemplate, kVulkanObjectTypeDescriptorUpdateTemplateKHR, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_24805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, descriptorUpdateTemplate, kVulkanObjectTypeDescriptorUpdateTemplateKHR, true, VALIDATION_ERROR_24805201, VALIDATION_ERROR_24805207);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, descriptorUpdateTemplate, kVulkanObjectTypeDescriptorUpdateTemplateKHR, pAllocator, VALIDATION_ERROR_248002c8, VALIDATION_ERROR_248002ca);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyDescriptorUpdateTemplateKHR(device, descriptorUpdateTemplate, pAllocator);

}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSetWithTemplateKHR(
    VkDevice                                    device,
    VkDescriptorSet                             descriptorSet,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    const void*                                 pData)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33a05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, descriptorSet, kVulkanObjectTypeDescriptorSet, false, VALIDATION_ERROR_33a04801, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, descriptorUpdateTemplate, kVulkanObjectTypeDescriptorUpdateTemplateKHR, false, VALIDATION_ERROR_33a05201, VALIDATION_ERROR_33a05207);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->UpdateDescriptorSetWithTemplateKHR(device, descriptorSet, descriptorUpdateTemplate, pData);

}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer                             commandBuffer,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    const void*                                 pData)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1c002401, VALIDATION_ERROR_1c000009);
        skip |= ValidateObject(commandBuffer, descriptorUpdateTemplate, kVulkanObjectTypeDescriptorUpdateTemplateKHR, false, VALIDATION_ERROR_1c005201, VALIDATION_ERROR_1c000009);
        skip |= ValidateObject(commandBuffer, layout, kVulkanObjectTypePipelineLayout, false, VALIDATION_ERROR_1c00be01, VALIDATION_ERROR_1c000009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdPushDescriptorSetWithTemplateKHR(commandBuffer, descriptorUpdateTemplate, layout, set, pData);

}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainStatusKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30a05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_30a2f001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetSwapchainStatusKHR(device, swapchain);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceExternalFencePropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceExternalFenceInfoKHR* pExternalFenceInfo,
    VkExternalFencePropertiesKHR*               pExternalFenceProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_39a27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceExternalFencePropertiesKHR(physicalDevice, pExternalFenceInfo, pExternalFenceProperties);

}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkImportFenceWin32HandleInfoKHR*      pImportFenceWin32HandleInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_3aa05601, VALIDATION_ERROR_UNDEFINED);
        if (pImportFenceWin32HandleInfo) {
            skip |= ValidateObject(device, pImportFenceWin32HandleInfo->fence, kVulkanObjectTypeFence, false, VALIDATION_ERROR_3a408801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ImportFenceWin32HandleKHR(device, pImportFenceWin32HandleInfo);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkFenceGetWin32HandleInfoKHR*         pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_3ac05601, VALIDATION_ERROR_UNDEFINED);
        if (pGetWin32HandleInfo) {
            skip |= ValidateObject(device, pGetWin32HandleInfo->fence, kVulkanObjectTypeFence, false, VALIDATION_ERROR_3a808801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetFenceWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceFdKHR(
    VkDevice                                    device,
    const VkImportFenceFdInfoKHR*               pImportFenceFdInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_3a005601, VALIDATION_ERROR_UNDEFINED);
        if (pImportFenceFdInfo) {
            skip |= ValidateObject(device, pImportFenceFdInfo->fence, kVulkanObjectTypeFence, false, VALIDATION_ERROR_39c08801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ImportFenceFdKHR(device, pImportFenceFdInfo);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceFdKHR(
    VkDevice                                    device,
    const VkFenceGetFdInfoKHR*                  pGetFdInfo,
    int*                                        pFd)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_3a205601, VALIDATION_ERROR_UNDEFINED);
        if (pGetFdInfo) {
            skip |= ValidateObject(device, pGetFdInfo->fence, kVulkanObjectTypeFence, false, VALIDATION_ERROR_39e08801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetFenceFdKHR(device, pGetFdInfo, pFd);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    VkSurfaceCapabilities2KHR*                  pSurfaceCapabilities)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2e427a01, VALIDATION_ERROR_UNDEFINED);
        if (pSurfaceInfo) {
            skip |= ValidateObject(physicalDevice, pSurfaceInfo->surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_0ee2ec01, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice, pSurfaceInfo, pSurfaceCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormats2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    uint32_t*                                   pSurfaceFormatCount,
    VkSurfaceFormat2KHR*                        pSurfaceFormats)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2e827a01, VALIDATION_ERROR_UNDEFINED);
        if (pSurfaceInfo) {
            skip |= ValidateObject(physicalDevice, pSurfaceInfo->surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_0ee2ec01, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceFormats2KHR(physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageMemoryRequirementsInfo2KHR*    pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_3c405601, VALIDATION_ERROR_UNDEFINED);
        if (pInfo) {
            skip |= ValidateObject(device, pInfo->image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_3bc0a001, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetImageMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkBufferMemoryRequirementsInfo2KHR*   pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_3c605601, VALIDATION_ERROR_UNDEFINED);
        if (pInfo) {
            skip |= ValidateObject(device, pInfo->buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_3ba01a01, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetBufferMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageSparseMemoryRequirementsInfo2KHR* pInfo,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements2KHR*        pSparseMemoryRequirements)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
        if (pInfo) {
            skip |= ValidateObject(device, pInfo->image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetImageSparseMemoryRequirements2KHR(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);

}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateDebugReportCallbackEXT(
    VkInstance                                  instance,
    const VkDebugReportCallbackCreateInfoEXT*   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDebugReportCallbackEXT*                   pCallback);

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroyDebugReportCallbackEXT(
    VkInstance                                  instance,
    VkDebugReportCallbackEXT                    callback,
    const VkAllocationCallbacks*                pAllocator);

// Declare only
VKAPI_ATTR void VKAPI_CALL DebugReportMessageEXT(
    VkInstance                                  instance,
    VkDebugReportFlagsEXT                       flags,
    VkDebugReportObjectTypeEXT                  objectType,
    uint64_t                                    object,
    size_t                                      location,
    int32_t                                     messageCode,
    const char*                                 pLayerPrefix,
    const char*                                 pMessage);

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectTagEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectTagInfoEXT*        pTagInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_23805601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->DebugMarkerSetObjectTagEXT(device, pTagInfo);

    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectNameEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectNameInfoEXT*       pNameInfo);

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerBeginEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugMarkerMarkerInfoEXT*           pMarkerInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19602401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDebugMarkerBeginEXT(commandBuffer, pMarkerInfo);

}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerEndEXT(
    VkCommandBuffer                             commandBuffer)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19802401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDebugMarkerEndEXT(commandBuffer);

}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerInsertEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugMarkerMarkerInfoEXT*           pMarkerInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19a02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDebugMarkerInsertEXT(commandBuffer, pMarkerInfo);

}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountAMD(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1ac02401, VALIDATION_ERROR_1ac00009);
        skip |= ValidateObject(commandBuffer, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1ac01a01, VALIDATION_ERROR_1ac00009);
        skip |= ValidateObject(commandBuffer, countBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1ac03401, VALIDATION_ERROR_1ac00009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDrawIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);

}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCountAMD(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1a802401, VALIDATION_ERROR_1a800009);
        skip |= ValidateObject(commandBuffer, buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1a801a01, VALIDATION_ERROR_1a800009);
        skip |= ValidateObject(commandBuffer, countBuffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_1a803401, VALIDATION_ERROR_1a800009);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDrawIndexedIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceExternalImageFormatPropertiesNV(
    VkPhysicalDevice                            physicalDevice,
    VkFormat                                    format,
    VkImageType                                 type,
    VkImageTiling                               tiling,
    VkImageUsageFlags                           usage,
    VkImageCreateFlags                          flags,
    VkExternalMemoryHandleTypeFlagsNV           externalHandleType,
    VkExternalImageFormatPropertiesNV*          pExternalImageFormatProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2bc27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceExternalImageFormatPropertiesNV(physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);

    return result;
}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleNV(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkExternalMemoryHandleTypeFlagsNV           handleType,
    HANDLE*                                     pHandle)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2b005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_2b00c601, VALIDATION_ERROR_2b00c607);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetMemoryWin32HandleNV(device, memory, handleType, pHandle);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR void VKAPI_CALL GetDeviceGroupPeerMemoryFeaturesKHX(
    VkDevice                                    device,
    uint32_t                                    heapIndex,
    uint32_t                                    localDeviceIndex,
    uint32_t                                    remoteDeviceIndex,
    VkPeerMemoryFeatureFlagsKHX*                pPeerMemoryFeatures)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28c05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->GetDeviceGroupPeerMemoryFeaturesKHX(device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);

}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory2KHX(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindBufferMemoryInfoKHX*            pBindInfos)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_17205601, VALIDATION_ERROR_UNDEFINED);
        if (pBindInfos) {
            for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
                skip |= ValidateObject(device, pBindInfos[index0].buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_00c01a01, VALIDATION_ERROR_00c00009);
                skip |= ValidateObject(device, pBindInfos[index0].memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_00c0c601, VALIDATION_ERROR_00c00009);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->BindBufferMemory2KHX(device, bindInfoCount, pBindInfos);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2KHX(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindImageMemoryInfoKHX*             pBindInfos)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_17605601, VALIDATION_ERROR_UNDEFINED);
        if (pBindInfos) {
            for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
                skip |= ValidateObject(device, pBindInfos[index0].image, kVulkanObjectTypeImage, false, VALIDATION_ERROR_00e0a001, VALIDATION_ERROR_00e00009);
                skip |= ValidateObject(device, pBindInfos[index0].memory, kVulkanObjectTypeDeviceMemory, false, VALIDATION_ERROR_00e0c601, VALIDATION_ERROR_00e00009);
            }
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->BindImageMemory2KHX(device, bindInfoCount, pBindInfos);

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdSetDeviceMaskKHX(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    deviceMask)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1d002401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetDeviceMaskKHX(commandBuffer, deviceMask);

}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupPresentCapabilitiesKHX(
    VkDevice                                    device,
    VkDeviceGroupPresentCapabilitiesKHX*        pDeviceGroupPresentCapabilities)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28e05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetDeviceGroupPresentCapabilitiesKHX(device, pDeviceGroupPresentCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupSurfacePresentModesKHX(
    VkDevice                                    device,
    VkSurfaceKHR                                surface,
    VkDeviceGroupPresentModeFlagsKHX*           pModes)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_29005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2902ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetDeviceGroupSurfacePresentModesKHX(device, surface, pModes);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImage2KHX(
    VkDevice                                    device,
    const VkAcquireNextImageInfoKHX*            pAcquireInfo,
    uint32_t*                                   pImageIndex)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_16205601, VALIDATION_ERROR_UNDEFINED);
        if (pAcquireInfo) {
            skip |= ValidateObject(device, pAcquireInfo->swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_0002f001, VALIDATION_ERROR_UNDEFINED);
            skip |= ValidateObject(device, pAcquireInfo->semaphore, kVulkanObjectTypeSemaphore, true, VALIDATION_ERROR_0002b801, VALIDATION_ERROR_UNDEFINED);
            skip |= ValidateObject(device, pAcquireInfo->fence, kVulkanObjectTypeFence, true, VALIDATION_ERROR_00008801, VALIDATION_ERROR_UNDEFINED);
        }
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->AcquireNextImage2KHX(device, pAcquireInfo, pImageIndex);

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchBaseKHX(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    baseGroupX,
    uint32_t                                    baseGroupY,
    uint32_t                                    baseGroupZ,
    uint32_t                                    groupCountX,
    uint32_t                                    groupCountY,
    uint32_t                                    groupCountZ)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_19e02401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdDispatchBaseKHX(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDevicePresentRectanglesKHX(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pRectCount,
    VkRect2D*                                   pRects)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2d427a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2d42ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDevicePresentRectanglesKHX(physicalDevice, surface, pRectCount, pRects);

    return result;
}

#ifdef VK_USE_PLATFORM_VI_NN

VKAPI_ATTR VkResult VKAPI_CALL CreateViSurfaceNN(
    VkInstance                                  instance,
    const VkViSurfaceCreateInfoNN*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_22c0bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateViSurfaceNN(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_VI_NN

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDeviceGroupsKHX(
    VkInstance                                  instance,
    uint32_t*                                   pPhysicalDeviceGroupCount,
    VkPhysicalDeviceGroupPropertiesKHX*         pPhysicalDeviceGroupProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_27e0bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->EnumeratePhysicalDeviceGroupsKHX(instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties);

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdProcessCommandsNVX(
    VkCommandBuffer                             commandBuffer,
    const VkCmdProcessCommandsInfoNVX*          pProcessCommandsInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1ba02401, VALIDATION_ERROR_UNDEFINED);
        if (pProcessCommandsInfo) {
            skip |= ValidateObject(commandBuffer, pProcessCommandsInfo->objectTable, kVulkanObjectTypeObjectTableNVX, false, VALIDATION_ERROR_0220d801, VALIDATION_ERROR_02200009);
            skip |= ValidateObject(commandBuffer, pProcessCommandsInfo->indirectCommandsLayout, kVulkanObjectTypeIndirectCommandsLayoutNVX, false, VALIDATION_ERROR_0220b401, VALIDATION_ERROR_02200009);
            if (pProcessCommandsInfo->pIndirectCommandsTokens) {
                for (uint32_t index1 = 0; index1 < pProcessCommandsInfo->indirectCommandsTokenCount; ++index1) {
                    skip |= ValidateObject(commandBuffer, pProcessCommandsInfo->pIndirectCommandsTokens[index1].buffer, kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_0bc01a01, VALIDATION_ERROR_UNDEFINED);
                }
            }
            skip |= ValidateObject(commandBuffer, pProcessCommandsInfo->targetCommandBuffer, kVulkanObjectTypeCommandBuffer, true, VALIDATION_ERROR_0222f601, VALIDATION_ERROR_02200009);
            skip |= ValidateObject(commandBuffer, pProcessCommandsInfo->sequencesCountBuffer, kVulkanObjectTypeBuffer, true, VALIDATION_ERROR_0222ba01, VALIDATION_ERROR_02200009);
            skip |= ValidateObject(commandBuffer, pProcessCommandsInfo->sequencesIndexBuffer, kVulkanObjectTypeBuffer, true, VALIDATION_ERROR_0222bc01, VALIDATION_ERROR_02200009);
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdProcessCommandsNVX(commandBuffer, pProcessCommandsInfo);

}

VKAPI_ATTR void VKAPI_CALL CmdReserveSpaceForCommandsNVX(
    VkCommandBuffer                             commandBuffer,
    const VkCmdReserveSpaceForCommandsInfoNVX*  pReserveSpaceInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1c202401, VALIDATION_ERROR_UNDEFINED);
        if (pReserveSpaceInfo) {
            skip |= ValidateObject(commandBuffer, pReserveSpaceInfo->objectTable, kVulkanObjectTypeObjectTableNVX, false, VALIDATION_ERROR_0240d801, VALIDATION_ERROR_02400009);
            skip |= ValidateObject(commandBuffer, pReserveSpaceInfo->indirectCommandsLayout, kVulkanObjectTypeIndirectCommandsLayoutNVX, false, VALIDATION_ERROR_0240b401, VALIDATION_ERROR_02400009);
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdReserveSpaceForCommandsNVX(commandBuffer, pReserveSpaceInfo);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateIndirectCommandsLayoutNVX(
    VkDevice                                    device,
    const VkIndirectCommandsLayoutCreateInfoNVX* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkIndirectCommandsLayoutNVX*                pIndirectCommandsLayout)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_21005601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateIndirectCommandsLayoutNVX(device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pIndirectCommandsLayout, kVulkanObjectTypeIndirectCommandsLayoutNVX, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyIndirectCommandsLayoutNVX(
    VkDevice                                    device,
    VkIndirectCommandsLayoutNVX                 indirectCommandsLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25605601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, indirectCommandsLayout, kVulkanObjectTypeIndirectCommandsLayoutNVX, false, VALIDATION_ERROR_2560b401, VALIDATION_ERROR_2560b407);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, indirectCommandsLayout, kVulkanObjectTypeIndirectCommandsLayoutNVX, pAllocator, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyIndirectCommandsLayoutNVX(device, indirectCommandsLayout, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateObjectTableNVX(
    VkDevice                                    device,
    const VkObjectTableCreateInfoNVX*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkObjectTableNVX*                           pObjectTable)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_21805601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->CreateObjectTableNVX(device, pCreateInfo, pAllocator, pObjectTable);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pObjectTable, kVulkanObjectTypeObjectTableNVX, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyObjectTableNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    const VkAllocationCallbacks*                pAllocator)
{
    bool skip = false;
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_25a05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, objectTable, kVulkanObjectTypeObjectTableNVX, false, VALIDATION_ERROR_25a0d801, VALIDATION_ERROR_25a0d807);
    if (skip) return;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        DestroyObject(device, objectTable, kVulkanObjectTypeObjectTableNVX, pAllocator, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->DestroyObjectTableNVX(device, objectTable, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL RegisterObjectsNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    uint32_t                                    objectCount,
    const VkObjectTableEntryNVX* const*         ppObjectTableEntries,
    const uint32_t*                             pObjectIndices)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_32205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, objectTable, kVulkanObjectTypeObjectTableNVX, false, VALIDATION_ERROR_3220d801, VALIDATION_ERROR_3220d807);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->RegisterObjectsNVX(device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL UnregisterObjectsNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    uint32_t                                    objectCount,
    const VkObjectEntryTypeNVX*                 pObjectEntryTypes,
    const uint32_t*                             pObjectIndices)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33805601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, objectTable, kVulkanObjectTypeObjectTableNVX, false, VALIDATION_ERROR_3380d801, VALIDATION_ERROR_3380d807);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->UnregisterObjectsNVX(device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceGeneratedCommandsPropertiesNVX(
    VkPhysicalDevice                            physicalDevice,
    VkDeviceGeneratedCommandsFeaturesNVX*       pFeatures,
    VkDeviceGeneratedCommandsLimitsNVX*         pLimits)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2c827a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceGeneratedCommandsPropertiesNVX(physicalDevice, pFeatures, pLimits);

}

VKAPI_ATTR void VKAPI_CALL CmdSetViewportWScalingNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewportWScalingNV*                 pViewportWScalings)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1e202401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetViewportWScalingNV(commandBuffer, firstViewport, viewportCount, pViewportWScalings);

}

VKAPI_ATTR VkResult VKAPI_CALL ReleaseDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_32427a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, display, kVulkanObjectTypeDisplayKHR, false, VALIDATION_ERROR_32406001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->ReleaseDisplayEXT(physicalDevice, display);

    return result;
}

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT

VKAPI_ATTR VkResult VKAPI_CALL AcquireXlibDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    Display*                                    dpy,
    VkDisplayKHR                                display)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_16627a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, display, kVulkanObjectTypeDisplayKHR, false, VALIDATION_ERROR_16606001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->AcquireXlibDisplayEXT(physicalDevice, dpy, display);

    return result;
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT

VKAPI_ATTR VkResult VKAPI_CALL GetRandROutputDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    Display*                                    dpy,
    RROutput                                    rrOutput,
    VkDisplayKHR*                               pDisplay)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2fc27a01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetRandROutputDisplayEXT(physicalDevice, dpy, rrOutput, pDisplay);

    return result;
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2EXT(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    VkSurfaceCapabilities2EXT*                  pSurfaceCapabilities)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2e227a01, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(physicalDevice, surface, kVulkanObjectTypeSurfaceKHR, false, VALIDATION_ERROR_2e22ec01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceCapabilities2EXT(physicalDevice, surface, pSurfaceCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DisplayPowerControlEXT(
    VkDevice                                    device,
    VkDisplayKHR                                display,
    const VkDisplayPowerInfoEXT*                pDisplayPowerInfo)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_27205601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, display, kVulkanObjectTypeDisplayKHR, false, VALIDATION_ERROR_27206001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->DisplayPowerControlEXT(device, display, pDisplayPowerInfo);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDeviceEventEXT(
    VkDevice                                    device,
    const VkDeviceEventInfoEXT*                 pDeviceEventInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_31e05601, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->RegisterDeviceEventEXT(device, pDeviceEventInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pFence, kVulkanObjectTypeFence, pAllocator);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDisplayEventEXT(
    VkDevice                                    device,
    VkDisplayKHR                                display,
    const VkDisplayEventInfoEXT*                pDisplayEventInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_32005601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, display, kVulkanObjectTypeDisplayKHR, false, VALIDATION_ERROR_32006001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->RegisterDisplayEventEXT(device, display, pDisplayEventInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(device, *pFence, kVulkanObjectTypeFence, pAllocator);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainCounterEXT(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkSurfaceCounterFlagBitsEXT                 counter,
    uint64_t*                                   pCounterValue)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30605601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_3062f001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetSwapchainCounterEXT(device, swapchain, counter, pCounterValue);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetRefreshCycleDurationGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkRefreshCycleDurationGOOGLE*               pDisplayTimingProperties)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2fe05601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_2fe2f001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetRefreshCycleDurationGOOGLE(device, swapchain, pDisplayTimingProperties);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPastPresentationTimingGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint32_t*                                   pPresentationTimingCount,
    VkPastPresentationTimingGOOGLE*             pPresentationTimings)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_2b405601, VALIDATION_ERROR_UNDEFINED);
        skip |= ValidateObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_2b42f001, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_device_table_map, device)->GetPastPresentationTimingGOOGLE(device, swapchain, pPresentationTimingCount, pPresentationTimings);

    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdSetDiscardRectangleEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstDiscardRectangle,
    uint32_t                                    discardRectangleCount,
    const VkRect2D*                             pDiscardRectangles)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1d202401, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)->CmdSetDiscardRectangleEXT(commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles);

}

VKAPI_ATTR void VKAPI_CALL SetHdrMetadataEXT(
    VkDevice                                    device,
    uint32_t                                    swapchainCount,
    const VkSwapchainKHR*                       pSwapchains,
    const VkHdrMetadataEXT*                     pMetadata)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33205601, VALIDATION_ERROR_UNDEFINED);
        if (pSwapchains) {
            for (uint32_t index0 = 0; index0 < swapchainCount; ++index0) {
                skip |= ValidateObject(device, pSwapchains[index0], kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_33225801, VALIDATION_ERROR_UNDEFINED);
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, device)->SetHdrMetadataEXT(device, swapchainCount, pSwapchains, pMetadata);

}

#ifdef VK_USE_PLATFORM_IOS_MVK

VKAPI_ATTR VkResult VKAPI_CALL CreateIOSSurfaceMVK(
    VkInstance                                  instance,
    const VkIOSSurfaceCreateInfoMVK*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_20a0bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateIOSSurfaceMVK(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_IOS_MVK

#ifdef VK_USE_PLATFORM_MACOS_MVK

VKAPI_ATTR VkResult VKAPI_CALL CreateMacOSSurfaceMVK(
    VkInstance                                  instance,
    const VkMacOSSurfaceCreateInfoMVK*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2140bc01, VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)->CreateMacOSSurfaceMVK(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        CreateObject(instance, *pSurface, kVulkanObjectTypeSurfaceKHR, pAllocator);
    }
    return result;
}
#endif // VK_USE_PLATFORM_MACOS_MVK
// Map of all APIs to be intercepted by this layer
const std::unordered_map<std::string, void*> name_to_funcptr_map = {
    {"vkCreateInstance", (void *)CreateInstance},
    {"vkDestroyInstance", (void *)DestroyInstance},
    {"vkEnumeratePhysicalDevices", (void *)EnumeratePhysicalDevices},
    {"vkGetPhysicalDeviceFeatures", (void*)GetPhysicalDeviceFeatures},
    {"vkGetPhysicalDeviceFormatProperties", (void*)GetPhysicalDeviceFormatProperties},
    {"vkGetPhysicalDeviceImageFormatProperties", (void*)GetPhysicalDeviceImageFormatProperties},
    {"vkGetPhysicalDeviceProperties", (void*)GetPhysicalDeviceProperties},
    {"vkGetPhysicalDeviceQueueFamilyProperties", (void *)GetPhysicalDeviceQueueFamilyProperties},
    {"vkGetPhysicalDeviceMemoryProperties", (void*)GetPhysicalDeviceMemoryProperties},
    {"vkGetInstanceProcAddr", (void *)GetInstanceProcAddr},
    {"vkGetDeviceProcAddr", (void *)GetDeviceProcAddr},
    {"vkCreateDevice", (void *)CreateDevice},
    {"vkDestroyDevice", (void *)DestroyDevice},
    {"vkEnumerateInstanceExtensionProperties", (void *)EnumerateInstanceExtensionProperties},
    {"vkEnumerateDeviceExtensionProperties", (void *)EnumerateDeviceExtensionProperties},
    {"vkEnumerateInstanceLayerProperties", (void *)EnumerateInstanceLayerProperties},
    {"vkEnumerateDeviceLayerProperties", (void *)EnumerateDeviceLayerProperties},
    {"vkGetDeviceQueue", (void *)GetDeviceQueue},
    {"vkQueueSubmit", (void*)QueueSubmit},
    {"vkQueueWaitIdle", (void*)QueueWaitIdle},
    {"vkDeviceWaitIdle", (void*)DeviceWaitIdle},
    {"vkAllocateMemory", (void*)AllocateMemory},
    {"vkFreeMemory", (void*)FreeMemory},
    {"vkMapMemory", (void*)MapMemory},
    {"vkUnmapMemory", (void*)UnmapMemory},
    {"vkFlushMappedMemoryRanges", (void*)FlushMappedMemoryRanges},
    {"vkInvalidateMappedMemoryRanges", (void*)InvalidateMappedMemoryRanges},
    {"vkGetDeviceMemoryCommitment", (void*)GetDeviceMemoryCommitment},
    {"vkBindBufferMemory", (void*)BindBufferMemory},
    {"vkBindImageMemory", (void*)BindImageMemory},
    {"vkGetBufferMemoryRequirements", (void*)GetBufferMemoryRequirements},
    {"vkGetImageMemoryRequirements", (void*)GetImageMemoryRequirements},
    {"vkGetImageSparseMemoryRequirements", (void*)GetImageSparseMemoryRequirements},
    {"vkGetPhysicalDeviceSparseImageFormatProperties", (void*)GetPhysicalDeviceSparseImageFormatProperties},
    {"vkQueueBindSparse", (void*)QueueBindSparse},
    {"vkCreateFence", (void*)CreateFence},
    {"vkDestroyFence", (void*)DestroyFence},
    {"vkResetFences", (void*)ResetFences},
    {"vkGetFenceStatus", (void*)GetFenceStatus},
    {"vkWaitForFences", (void*)WaitForFences},
    {"vkCreateSemaphore", (void*)CreateSemaphore},
    {"vkDestroySemaphore", (void*)DestroySemaphore},
    {"vkCreateEvent", (void*)CreateEvent},
    {"vkDestroyEvent", (void*)DestroyEvent},
    {"vkGetEventStatus", (void*)GetEventStatus},
    {"vkSetEvent", (void*)SetEvent},
    {"vkResetEvent", (void*)ResetEvent},
    {"vkCreateQueryPool", (void*)CreateQueryPool},
    {"vkDestroyQueryPool", (void*)DestroyQueryPool},
    {"vkGetQueryPoolResults", (void*)GetQueryPoolResults},
    {"vkCreateBuffer", (void*)CreateBuffer},
    {"vkDestroyBuffer", (void*)DestroyBuffer},
    {"vkCreateBufferView", (void*)CreateBufferView},
    {"vkDestroyBufferView", (void*)DestroyBufferView},
    {"vkCreateImage", (void*)CreateImage},
    {"vkDestroyImage", (void*)DestroyImage},
    {"vkGetImageSubresourceLayout", (void*)GetImageSubresourceLayout},
    {"vkCreateImageView", (void*)CreateImageView},
    {"vkDestroyImageView", (void*)DestroyImageView},
    {"vkCreateShaderModule", (void*)CreateShaderModule},
    {"vkDestroyShaderModule", (void*)DestroyShaderModule},
    {"vkCreatePipelineCache", (void*)CreatePipelineCache},
    {"vkDestroyPipelineCache", (void*)DestroyPipelineCache},
    {"vkGetPipelineCacheData", (void*)GetPipelineCacheData},
    {"vkMergePipelineCaches", (void*)MergePipelineCaches},
    {"vkCreateGraphicsPipelines", (void*)CreateGraphicsPipelines},
    {"vkCreateComputePipelines", (void *)CreateComputePipelines},
    {"vkDestroyPipeline", (void*)DestroyPipeline},
    {"vkCreatePipelineLayout", (void*)CreatePipelineLayout},
    {"vkDestroyPipelineLayout", (void*)DestroyPipelineLayout},
    {"vkCreateSampler", (void*)CreateSampler},
    {"vkDestroySampler", (void*)DestroySampler},
    {"vkCreateDescriptorSetLayout", (void*)CreateDescriptorSetLayout},
    {"vkDestroyDescriptorSetLayout", (void*)DestroyDescriptorSetLayout},
    {"vkCreateDescriptorPool", (void*)CreateDescriptorPool},
    {"vkDestroyDescriptorPool", (void *)DestroyDescriptorPool},
    {"vkResetDescriptorPool", (void *)ResetDescriptorPool},
    {"vkAllocateDescriptorSets", (void *)AllocateDescriptorSets},
    {"vkFreeDescriptorSets", (void *)FreeDescriptorSets},
    {"vkUpdateDescriptorSets", (void *)UpdateDescriptorSets},
    {"vkCreateFramebuffer", (void*)CreateFramebuffer},
    {"vkDestroyFramebuffer", (void*)DestroyFramebuffer},
    {"vkCreateRenderPass", (void*)CreateRenderPass},
    {"vkDestroyRenderPass", (void*)DestroyRenderPass},
    {"vkGetRenderAreaGranularity", (void*)GetRenderAreaGranularity},
    {"vkCreateCommandPool", (void*)CreateCommandPool},
    {"vkDestroyCommandPool", (void *)DestroyCommandPool},
    {"vkResetCommandPool", (void*)ResetCommandPool},
    {"vkAllocateCommandBuffers", (void *)AllocateCommandBuffers},
    {"vkFreeCommandBuffers", (void *)FreeCommandBuffers},
    {"vkBeginCommandBuffer", (void *)BeginCommandBuffer},
    {"vkEndCommandBuffer", (void*)EndCommandBuffer},
    {"vkResetCommandBuffer", (void*)ResetCommandBuffer},
    {"vkCmdBindPipeline", (void*)CmdBindPipeline},
    {"vkCmdSetViewport", (void*)CmdSetViewport},
    {"vkCmdSetScissor", (void*)CmdSetScissor},
    {"vkCmdSetLineWidth", (void*)CmdSetLineWidth},
    {"vkCmdSetDepthBias", (void*)CmdSetDepthBias},
    {"vkCmdSetBlendConstants", (void*)CmdSetBlendConstants},
    {"vkCmdSetDepthBounds", (void*)CmdSetDepthBounds},
    {"vkCmdSetStencilCompareMask", (void*)CmdSetStencilCompareMask},
    {"vkCmdSetStencilWriteMask", (void*)CmdSetStencilWriteMask},
    {"vkCmdSetStencilReference", (void*)CmdSetStencilReference},
    {"vkCmdBindDescriptorSets", (void*)CmdBindDescriptorSets},
    {"vkCmdBindIndexBuffer", (void*)CmdBindIndexBuffer},
    {"vkCmdBindVertexBuffers", (void*)CmdBindVertexBuffers},
    {"vkCmdDraw", (void*)CmdDraw},
    {"vkCmdDrawIndexed", (void*)CmdDrawIndexed},
    {"vkCmdDrawIndirect", (void*)CmdDrawIndirect},
    {"vkCmdDrawIndexedIndirect", (void*)CmdDrawIndexedIndirect},
    {"vkCmdDispatch", (void*)CmdDispatch},
    {"vkCmdDispatchIndirect", (void*)CmdDispatchIndirect},
    {"vkCmdCopyBuffer", (void*)CmdCopyBuffer},
    {"vkCmdCopyImage", (void*)CmdCopyImage},
    {"vkCmdBlitImage", (void*)CmdBlitImage},
    {"vkCmdCopyBufferToImage", (void*)CmdCopyBufferToImage},
    {"vkCmdCopyImageToBuffer", (void*)CmdCopyImageToBuffer},
    {"vkCmdUpdateBuffer", (void*)CmdUpdateBuffer},
    {"vkCmdFillBuffer", (void*)CmdFillBuffer},
    {"vkCmdClearColorImage", (void*)CmdClearColorImage},
    {"vkCmdClearDepthStencilImage", (void*)CmdClearDepthStencilImage},
    {"vkCmdClearAttachments", (void*)CmdClearAttachments},
    {"vkCmdResolveImage", (void*)CmdResolveImage},
    {"vkCmdSetEvent", (void*)CmdSetEvent},
    {"vkCmdResetEvent", (void*)CmdResetEvent},
    {"vkCmdWaitEvents", (void*)CmdWaitEvents},
    {"vkCmdPipelineBarrier", (void*)CmdPipelineBarrier},
    {"vkCmdBeginQuery", (void*)CmdBeginQuery},
    {"vkCmdEndQuery", (void*)CmdEndQuery},
    {"vkCmdResetQueryPool", (void*)CmdResetQueryPool},
    {"vkCmdWriteTimestamp", (void*)CmdWriteTimestamp},
    {"vkCmdCopyQueryPoolResults", (void*)CmdCopyQueryPoolResults},
    {"vkCmdPushConstants", (void*)CmdPushConstants},
    {"vkCmdBeginRenderPass", (void*)CmdBeginRenderPass},
    {"vkCmdNextSubpass", (void*)CmdNextSubpass},
    {"vkCmdEndRenderPass", (void*)CmdEndRenderPass},
    {"vkCmdExecuteCommands", (void*)CmdExecuteCommands},
    {"vkDestroySurfaceKHR", (void*)DestroySurfaceKHR},
    {"vkGetPhysicalDeviceSurfaceSupportKHR", (void*)GetPhysicalDeviceSurfaceSupportKHR},
    {"vkGetPhysicalDeviceSurfaceCapabilitiesKHR", (void*)GetPhysicalDeviceSurfaceCapabilitiesKHR},
    {"vkGetPhysicalDeviceSurfaceFormatsKHR", (void*)GetPhysicalDeviceSurfaceFormatsKHR},
    {"vkGetPhysicalDeviceSurfacePresentModesKHR", (void*)GetPhysicalDeviceSurfacePresentModesKHR},
    {"vkCreateSwapchainKHR", (void*)CreateSwapchainKHR},
    {"vkDestroySwapchainKHR", (void *)DestroySwapchainKHR},
    {"vkGetSwapchainImagesKHR", (void *)GetSwapchainImagesKHR},
    {"vkAcquireNextImageKHR", (void*)AcquireNextImageKHR},
    {"vkQueuePresentKHR", (void*)QueuePresentKHR},
    {"vkGetPhysicalDeviceDisplayPropertiesKHR", (void*)GetPhysicalDeviceDisplayPropertiesKHR},
    {"vkGetPhysicalDeviceDisplayPlanePropertiesKHR", (void*)GetPhysicalDeviceDisplayPlanePropertiesKHR},
    {"vkGetDisplayPlaneSupportedDisplaysKHR", (void*)GetDisplayPlaneSupportedDisplaysKHR},
    {"vkGetDisplayModePropertiesKHR", (void*)GetDisplayModePropertiesKHR},
    {"vkCreateDisplayModeKHR", (void*)CreateDisplayModeKHR},
    {"vkGetDisplayPlaneCapabilitiesKHR", (void*)GetDisplayPlaneCapabilitiesKHR},
    {"vkCreateDisplayPlaneSurfaceKHR", (void*)CreateDisplayPlaneSurfaceKHR},
    {"vkCreateSharedSwapchainsKHR", (void*)CreateSharedSwapchainsKHR},
#ifdef VK_USE_PLATFORM_XLIB_KHR
    {"vkCreateXlibSurfaceKHR", (void*)CreateXlibSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    {"vkGetPhysicalDeviceXlibPresentationSupportKHR", (void*)GetPhysicalDeviceXlibPresentationSupportKHR},
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
    {"vkCreateXcbSurfaceKHR", (void*)CreateXcbSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
    {"vkGetPhysicalDeviceXcbPresentationSupportKHR", (void*)GetPhysicalDeviceXcbPresentationSupportKHR},
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    {"vkCreateWaylandSurfaceKHR", (void*)CreateWaylandSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    {"vkGetPhysicalDeviceWaylandPresentationSupportKHR", (void*)GetPhysicalDeviceWaylandPresentationSupportKHR},
#endif
#ifdef VK_USE_PLATFORM_MIR_KHR
    {"vkCreateMirSurfaceKHR", (void*)CreateMirSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_MIR_KHR
    {"vkGetPhysicalDeviceMirPresentationSupportKHR", (void*)GetPhysicalDeviceMirPresentationSupportKHR},
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    {"vkCreateAndroidSurfaceKHR", (void*)CreateAndroidSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkCreateWin32SurfaceKHR", (void*)CreateWin32SurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetPhysicalDeviceWin32PresentationSupportKHR", (void*)GetPhysicalDeviceWin32PresentationSupportKHR},
#endif
    {"vkGetPhysicalDeviceFeatures2KHR", (void*)GetPhysicalDeviceFeatures2KHR},
    {"vkGetPhysicalDeviceProperties2KHR", (void*)GetPhysicalDeviceProperties2KHR},
    {"vkGetPhysicalDeviceFormatProperties2KHR", (void*)GetPhysicalDeviceFormatProperties2KHR},
    {"vkGetPhysicalDeviceImageFormatProperties2KHR", (void*)GetPhysicalDeviceImageFormatProperties2KHR},
    {"vkGetPhysicalDeviceQueueFamilyProperties2KHR", (void *)GetPhysicalDeviceQueueFamilyProperties2KHR},
    {"vkGetPhysicalDeviceMemoryProperties2KHR", (void*)GetPhysicalDeviceMemoryProperties2KHR},
    {"vkGetPhysicalDeviceSparseImageFormatProperties2KHR", (void*)GetPhysicalDeviceSparseImageFormatProperties2KHR},
    {"vkTrimCommandPoolKHR", (void*)TrimCommandPoolKHR},
    {"vkGetPhysicalDeviceExternalBufferPropertiesKHR", (void*)GetPhysicalDeviceExternalBufferPropertiesKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetMemoryWin32HandleKHR", (void*)GetMemoryWin32HandleKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetMemoryWin32HandlePropertiesKHR", (void*)GetMemoryWin32HandlePropertiesKHR},
#endif
    {"vkGetMemoryFdKHR", (void*)GetMemoryFdKHR},
    {"vkGetMemoryFdPropertiesKHR", (void*)GetMemoryFdPropertiesKHR},
    {"vkGetPhysicalDeviceExternalSemaphorePropertiesKHR", (void*)GetPhysicalDeviceExternalSemaphorePropertiesKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkImportSemaphoreWin32HandleKHR", (void*)ImportSemaphoreWin32HandleKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetSemaphoreWin32HandleKHR", (void*)GetSemaphoreWin32HandleKHR},
#endif
    {"vkImportSemaphoreFdKHR", (void*)ImportSemaphoreFdKHR},
    {"vkGetSemaphoreFdKHR", (void*)GetSemaphoreFdKHR},
    {"vkCmdPushDescriptorSetKHR", (void *)CmdPushDescriptorSetKHR},
    {"vkCreateDescriptorUpdateTemplateKHR", (void*)CreateDescriptorUpdateTemplateKHR},
    {"vkDestroyDescriptorUpdateTemplateKHR", (void*)DestroyDescriptorUpdateTemplateKHR},
    {"vkUpdateDescriptorSetWithTemplateKHR", (void*)UpdateDescriptorSetWithTemplateKHR},
    {"vkCmdPushDescriptorSetWithTemplateKHR", (void*)CmdPushDescriptorSetWithTemplateKHR},
    {"vkGetSwapchainStatusKHR", (void*)GetSwapchainStatusKHR},
    {"vkGetPhysicalDeviceExternalFencePropertiesKHR", (void*)GetPhysicalDeviceExternalFencePropertiesKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkImportFenceWin32HandleKHR", (void*)ImportFenceWin32HandleKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetFenceWin32HandleKHR", (void*)GetFenceWin32HandleKHR},
#endif
    {"vkImportFenceFdKHR", (void*)ImportFenceFdKHR},
    {"vkGetFenceFdKHR", (void*)GetFenceFdKHR},
    {"vkGetPhysicalDeviceSurfaceCapabilities2KHR", (void*)GetPhysicalDeviceSurfaceCapabilities2KHR},
    {"vkGetPhysicalDeviceSurfaceFormats2KHR", (void*)GetPhysicalDeviceSurfaceFormats2KHR},
    {"vkGetImageMemoryRequirements2KHR", (void*)GetImageMemoryRequirements2KHR},
    {"vkGetBufferMemoryRequirements2KHR", (void*)GetBufferMemoryRequirements2KHR},
    {"vkGetImageSparseMemoryRequirements2KHR", (void*)GetImageSparseMemoryRequirements2KHR},
    {"vkCreateDebugReportCallbackEXT", (void *)CreateDebugReportCallbackEXT},
    {"vkDestroyDebugReportCallbackEXT", (void *)DestroyDebugReportCallbackEXT},
    {"vkDebugReportMessageEXT", (void *)DebugReportMessageEXT},
    {"vkDebugMarkerSetObjectTagEXT", (void*)DebugMarkerSetObjectTagEXT},
    {"vkDebugMarkerSetObjectNameEXT", (void *)DebugMarkerSetObjectNameEXT},
    {"vkCmdDebugMarkerBeginEXT", (void*)CmdDebugMarkerBeginEXT},
    {"vkCmdDebugMarkerEndEXT", (void*)CmdDebugMarkerEndEXT},
    {"vkCmdDebugMarkerInsertEXT", (void*)CmdDebugMarkerInsertEXT},
    {"vkCmdDrawIndirectCountAMD", (void*)CmdDrawIndirectCountAMD},
    {"vkCmdDrawIndexedIndirectCountAMD", (void*)CmdDrawIndexedIndirectCountAMD},
    {"vkGetPhysicalDeviceExternalImageFormatPropertiesNV", (void*)GetPhysicalDeviceExternalImageFormatPropertiesNV},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetMemoryWin32HandleNV", (void*)GetMemoryWin32HandleNV},
#endif
    {"vkGetDeviceGroupPeerMemoryFeaturesKHX", (void*)GetDeviceGroupPeerMemoryFeaturesKHX},
    {"vkBindBufferMemory2KHX", (void*)BindBufferMemory2KHX},
    {"vkBindImageMemory2KHX", (void*)BindImageMemory2KHX},
    {"vkCmdSetDeviceMaskKHX", (void*)CmdSetDeviceMaskKHX},
    {"vkGetDeviceGroupPresentCapabilitiesKHX", (void*)GetDeviceGroupPresentCapabilitiesKHX},
    {"vkGetDeviceGroupSurfacePresentModesKHX", (void*)GetDeviceGroupSurfacePresentModesKHX},
    {"vkAcquireNextImage2KHX", (void*)AcquireNextImage2KHX},
    {"vkCmdDispatchBaseKHX", (void*)CmdDispatchBaseKHX},
    {"vkGetPhysicalDevicePresentRectanglesKHX", (void*)GetPhysicalDevicePresentRectanglesKHX},
#ifdef VK_USE_PLATFORM_VI_NN
    {"vkCreateViSurfaceNN", (void*)CreateViSurfaceNN},
#endif
    {"vkEnumeratePhysicalDeviceGroupsKHX", (void*)EnumeratePhysicalDeviceGroupsKHX},
    {"vkCmdProcessCommandsNVX", (void*)CmdProcessCommandsNVX},
    {"vkCmdReserveSpaceForCommandsNVX", (void*)CmdReserveSpaceForCommandsNVX},
    {"vkCreateIndirectCommandsLayoutNVX", (void*)CreateIndirectCommandsLayoutNVX},
    {"vkDestroyIndirectCommandsLayoutNVX", (void*)DestroyIndirectCommandsLayoutNVX},
    {"vkCreateObjectTableNVX", (void*)CreateObjectTableNVX},
    {"vkDestroyObjectTableNVX", (void*)DestroyObjectTableNVX},
    {"vkRegisterObjectsNVX", (void*)RegisterObjectsNVX},
    {"vkUnregisterObjectsNVX", (void*)UnregisterObjectsNVX},
    {"vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX", (void*)GetPhysicalDeviceGeneratedCommandsPropertiesNVX},
    {"vkCmdSetViewportWScalingNV", (void*)CmdSetViewportWScalingNV},
    {"vkReleaseDisplayEXT", (void*)ReleaseDisplayEXT},
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    {"vkAcquireXlibDisplayEXT", (void*)AcquireXlibDisplayEXT},
#endif
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    {"vkGetRandROutputDisplayEXT", (void*)GetRandROutputDisplayEXT},
#endif
    {"vkGetPhysicalDeviceSurfaceCapabilities2EXT", (void*)GetPhysicalDeviceSurfaceCapabilities2EXT},
    {"vkDisplayPowerControlEXT", (void*)DisplayPowerControlEXT},
    {"vkRegisterDeviceEventEXT", (void*)RegisterDeviceEventEXT},
    {"vkRegisterDisplayEventEXT", (void*)RegisterDisplayEventEXT},
    {"vkGetSwapchainCounterEXT", (void*)GetSwapchainCounterEXT},
    {"vkGetRefreshCycleDurationGOOGLE", (void*)GetRefreshCycleDurationGOOGLE},
    {"vkGetPastPresentationTimingGOOGLE", (void*)GetPastPresentationTimingGOOGLE},
    {"vkCmdSetDiscardRectangleEXT", (void*)CmdSetDiscardRectangleEXT},
    {"vkSetHdrMetadataEXT", (void*)SetHdrMetadataEXT},
#ifdef VK_USE_PLATFORM_IOS_MVK
    {"vkCreateIOSSurfaceMVK", (void*)CreateIOSSurfaceMVK},
#endif
#ifdef VK_USE_PLATFORM_MACOS_MVK
    {"vkCreateMacOSSurfaceMVK", (void*)CreateMacOSSurfaceMVK},
#endif
};


} // namespace object_tracker


/* Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (c) 2015-2016 Google Inc.
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
 * Author: Lenny Komow <lenny@lunarg.com>
 */

/*
 * This file is generated from the Khronos Vulkan XML API Registry.
 */

#include "api_dump_text.h"
#include "api_dump_html.h"

//============================= Dump Functions ==============================//

inline void dump_vkAllocateCommandBuffers(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkAllocateCommandBuffers(dump_inst, result, device, pAllocateInfo, pCommandBuffers);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkAllocateCommandBuffers(dump_inst, result, device, pAllocateInfo, pCommandBuffers);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkEnumerateDeviceExtensionProperties(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEnumerateDeviceExtensionProperties(dump_inst, result, physicalDevice, pLayerName, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEnumerateDeviceExtensionProperties(dump_inst, result, physicalDevice, pLayerName, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkGetSemaphoreWin32HandleKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkSemaphoreGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetSemaphoreWin32HandleKHR(dump_inst, result, device, pGetWin32HandleInfo, pHandle);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetSemaphoreWin32HandleKHR(dump_inst, result, device, pGetWin32HandleInfo, pHandle);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
inline void dump_vkGetQueryPoolResults(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetQueryPoolResults(dump_inst, result, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetQueryPoolResults(dump_inst, result, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_MIR_KHR)
inline void dump_vkCreateMirSurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkMirSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateMirSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateMirSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_MIR_KHR
inline void dump_vkEnumerateInstanceLayerProperties(ApiDumpInstance& dump_inst, VkResult result, uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEnumerateInstanceLayerProperties(dump_inst, result, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEnumerateInstanceLayerProperties(dump_inst, result, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateEvent(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkEventCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkEvent* pEvent)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateEvent(dump_inst, result, device, pCreateInfo, pAllocator, pEvent);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateEvent(dump_inst, result, device, pCreateInfo, pAllocator, pEvent);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateBuffer(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateBuffer(dump_inst, result, device, pCreateInfo, pAllocator, pBuffer);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateBuffer(dump_inst, result, device, pCreateInfo, pAllocator, pBuffer);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkBeginCommandBuffer(ApiDumpInstance& dump_inst, VkResult result, VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkBeginCommandBuffer(dump_inst, result, commandBuffer, pBeginInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkBeginCommandBuffer(dump_inst, result, commandBuffer, pBeginInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkEnumerateDeviceLayerProperties(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEnumerateDeviceLayerProperties(dump_inst, result, physicalDevice, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEnumerateDeviceLayerProperties(dump_inst, result, physicalDevice, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateDescriptorSetLayout(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDescriptorSetLayout(dump_inst, result, device, pCreateInfo, pAllocator, pSetLayout);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDescriptorSetLayout(dump_inst, result, device, pCreateInfo, pAllocator, pSetLayout);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateRenderPass(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkRenderPassCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateRenderPass(dump_inst, result, device, pCreateInfo, pAllocator, pRenderPass);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateRenderPass(dump_inst, result, device, pCreateInfo, pAllocator, pRenderPass);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfaceCapabilities2KHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, VkSurfaceCapabilities2KHR* pSurfaceCapabilities)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfaceCapabilities2KHR(dump_inst, result, physicalDevice, pSurfaceInfo, pSurfaceCapabilities);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfaceCapabilities2KHR(dump_inst, result, physicalDevice, pSurfaceInfo, pSurfaceCapabilities);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkReleaseDisplayEXT(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkDisplayKHR display)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkReleaseDisplayEXT(dump_inst, result, physicalDevice, display);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkReleaseDisplayEXT(dump_inst, result, physicalDevice, display);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetInstanceProcAddr(ApiDumpInstance& dump_inst, PFN_vkVoidFunction result, VkInstance instance, const char* pName)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetInstanceProcAddr(dump_inst, result, instance, pName);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetInstanceProcAddr(dump_inst, result, instance, pName);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkGetMemoryWin32HandleNV(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagsNV handleType, HANDLE* pHandle)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetMemoryWin32HandleNV(dump_inst, result, device, memory, handleType, pHandle);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetMemoryWin32HandleNV(dump_inst, result, device, memory, handleType, pHandle);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
#if defined(VK_USE_PLATFORM_MIR_KHR)
inline void dump_vkGetPhysicalDeviceMirPresentationSupportKHR(ApiDumpInstance& dump_inst, VkBool32 result, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, MirConnection* connection)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceMirPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, connection);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceMirPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, connection);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_MIR_KHR
inline void dump_vkDisplayPowerControlEXT(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDisplayKHR display, const VkDisplayPowerInfoEXT* pDisplayPowerInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDisplayPowerControlEXT(dump_inst, result, device, display, pDisplayPowerInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDisplayPowerControlEXT(dump_inst, result, device, display, pDisplayPowerInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateInstance(ApiDumpInstance& dump_inst, VkResult result, const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateInstance(dump_inst, result, pCreateInfo, pAllocator, pInstance);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateInstance(dump_inst, result, pCreateInfo, pAllocator, pInstance);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfaceFormats2KHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, uint32_t* pSurfaceFormatCount, VkSurfaceFormat2KHR* pSurfaceFormats)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfaceFormats2KHR(dump_inst, result, physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfaceFormats2KHR(dump_inst, result, physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkRegisterDeviceEventEXT(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDeviceEventInfoEXT* pDeviceEventInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkRegisterDeviceEventEXT(dump_inst, result, device, pDeviceEventInfo, pAllocator, pFence);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkRegisterDeviceEventEXT(dump_inst, result, device, pDeviceEventInfo, pAllocator, pFence);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
inline void dump_vkCreateAndroidSurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkAndroidSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateAndroidSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateAndroidSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_ANDROID_KHR
inline void dump_vkSetEvent(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkEvent event)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkSetEvent(dump_inst, result, device, event);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkSetEvent(dump_inst, result, device, event);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetEventStatus(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkEvent event)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetEventStatus(dump_inst, result, device, event);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetEventStatus(dump_inst, result, device, event);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkRegisterDisplayEventEXT(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDisplayKHR display, const VkDisplayEventInfoEXT* pDisplayEventInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkRegisterDisplayEventEXT(dump_inst, result, device, display, pDisplayEventInfo, pAllocator, pFence);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkRegisterDisplayEventEXT(dump_inst, result, device, display, pDisplayEventInfo, pAllocator, pFence);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkEndCommandBuffer(ApiDumpInstance& dump_inst, VkResult result, VkCommandBuffer commandBuffer)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEndCommandBuffer(dump_inst, result, commandBuffer);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEndCommandBuffer(dump_inst, result, commandBuffer);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkQueueSubmit(ApiDumpInstance& dump_inst, VkResult result, VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkQueueSubmit(dump_inst, result, queue, submitCount, pSubmits, fence);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkQueueSubmit(dump_inst, result, queue, submitCount, pSubmits, fence);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDeviceProcAddr(ApiDumpInstance& dump_inst, PFN_vkVoidFunction result, VkDevice device, const char* pName)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDeviceProcAddr(dump_inst, result, device, pName);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDeviceProcAddr(dump_inst, result, device, pName);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetSwapchainCounterEXT(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSwapchainKHR swapchain, VkSurfaceCounterFlagBitsEXT counter, uint64_t* pCounterValue)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetSwapchainCounterEXT(dump_inst, result, device, swapchain, counter, pCounterValue);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetSwapchainCounterEXT(dump_inst, result, device, swapchain, counter, pCounterValue);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_XLIB_XRANDR_EXT)
inline void dump_vkAcquireXlibDisplayEXT(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, Display* dpy, VkDisplayKHR display)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkAcquireXlibDisplayEXT(dump_inst, result, physicalDevice, dpy, display);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkAcquireXlibDisplayEXT(dump_inst, result, physicalDevice, dpy, display);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
inline void dump_vkCreateDevice(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDevice(dump_inst, result, physicalDevice, pCreateInfo, pAllocator, pDevice);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDevice(dump_inst, result, physicalDevice, pCreateInfo, pAllocator, pDevice);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateShaderModule(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateShaderModule(dump_inst, result, device, pCreateInfo, pAllocator, pShaderModule);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateShaderModule(dump_inst, result, device, pCreateInfo, pAllocator, pShaderModule);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_IOS_MVK)
inline void dump_vkCreateIOSSurfaceMVK(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkIOSSurfaceCreateInfoMVK* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateIOSSurfaceMVK(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateIOSSurfaceMVK(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_IOS_MVK
inline void dump_vkImportSemaphoreFdKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkImportSemaphoreFdInfoKHR* pImportSemaphoreFdInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkImportSemaphoreFdKHR(dump_inst, result, device, pImportSemaphoreFdInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkImportSemaphoreFdKHR(dump_inst, result, device, pImportSemaphoreFdInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkResetEvent(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkEvent event)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkResetEvent(dump_inst, result, device, event);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkResetEvent(dump_inst, result, device, event);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateComputePipelines(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateComputePipelines(dump_inst, result, device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateComputePipelines(dump_inst, result, device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_XLIB_XRANDR_EXT)
inline void dump_vkGetRandROutputDisplayEXT(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, Display* dpy, RROutput rrOutput, VkDisplayKHR* pDisplay)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetRandROutputDisplayEXT(dump_inst, result, physicalDevice, dpy, rrOutput, pDisplay);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetRandROutputDisplayEXT(dump_inst, result, physicalDevice, dpy, rrOutput, pDisplay);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
inline void dump_vkCreateQueryPool(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateQueryPool(dump_inst, result, device, pCreateInfo, pAllocator, pQueryPool);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateQueryPool(dump_inst, result, device, pCreateInfo, pAllocator, pQueryPool);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_XCB_KHR)
inline void dump_vkCreateXcbSurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkXcbSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateXcbSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateXcbSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_XCB_KHR
inline void dump_vkGetSemaphoreFdKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkSemaphoreGetFdInfoKHR* pGetFdInfo, int* pFd)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetSemaphoreFdKHR(dump_inst, result, device, pGetFdInfo, pFd);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetSemaphoreFdKHR(dump_inst, result, device, pGetFdInfo, pFd);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfaceSupportKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkBool32* pSupported)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfaceSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, surface, pSupported);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfaceSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, surface, pSupported);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateDebugReportCallbackEXT(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDebugReportCallbackEXT(dump_inst, result, instance, pCreateInfo, pAllocator, pCallback);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDebugReportCallbackEXT(dump_inst, result, instance, pCreateInfo, pAllocator, pCallback);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkBindBufferMemory2KHX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfoKHX* pBindInfos)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkBindBufferMemory2KHX(dump_inst, result, device, bindInfoCount, pBindInfos);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkBindBufferMemory2KHX(dump_inst, result, device, bindInfoCount, pBindInfos);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* pSurfaceCapabilities)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dump_inst, result, physicalDevice, surface, pSurfaceCapabilities);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dump_inst, result, physicalDevice, surface, pSurfaceCapabilities);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfaceCapabilities2EXT(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilities2EXT* pSurfaceCapabilities)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfaceCapabilities2EXT(dump_inst, result, physicalDevice, surface, pSurfaceCapabilities);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfaceCapabilities2EXT(dump_inst, result, physicalDevice, surface, pSurfaceCapabilities);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkResetCommandBuffer(ApiDumpInstance& dump_inst, VkResult result, VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkResetCommandBuffer(dump_inst, result, commandBuffer, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkResetCommandBuffer(dump_inst, result, commandBuffer, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_MACOS_MVK)
inline void dump_vkCreateMacOSSurfaceMVK(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkMacOSSurfaceCreateInfoMVK* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateMacOSSurfaceMVK(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateMacOSSurfaceMVK(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_MACOS_MVK
inline void dump_vkGetRefreshCycleDurationGOOGLE(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSwapchainKHR swapchain, VkRefreshCycleDurationGOOGLE* pDisplayTimingProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetRefreshCycleDurationGOOGLE(dump_inst, result, device, swapchain, pDisplayTimingProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetRefreshCycleDurationGOOGLE(dump_inst, result, device, swapchain, pDisplayTimingProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkBindImageMemory2KHX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfoKHX* pBindInfos)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkBindImageMemory2KHX(dump_inst, result, device, bindInfoCount, pBindInfos);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkBindImageMemory2KHX(dump_inst, result, device, bindInfoCount, pBindInfos);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateDescriptorPool(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDescriptorPool(dump_inst, result, device, pCreateInfo, pAllocator, pDescriptorPool);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDescriptorPool(dump_inst, result, device, pCreateInfo, pAllocator, pDescriptorPool);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_XCB_KHR)
inline void dump_vkGetPhysicalDeviceXcbPresentationSupportKHR(ApiDumpInstance& dump_inst, VkBool32 result, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, xcb_connection_t* connection, xcb_visualid_t visual_id)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceXcbPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, connection, visual_id);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceXcbPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, connection, visual_id);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_XCB_KHR
inline void dump_vkCreatePipelineCache(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreatePipelineCache(dump_inst, result, device, pCreateInfo, pAllocator, pPipelineCache);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreatePipelineCache(dump_inst, result, device, pCreateInfo, pAllocator, pPipelineCache);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceImageFormatProperties2KHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2KHR* pImageFormatInfo, VkImageFormatProperties2KHR* pImageFormatProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceImageFormatProperties2KHR(dump_inst, result, physicalDevice, pImageFormatInfo, pImageFormatProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceImageFormatProperties2KHR(dump_inst, result, physicalDevice, pImageFormatInfo, pImageFormatProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDeviceGroupPresentCapabilitiesKHX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDeviceGroupPresentCapabilitiesKHX* pDeviceGroupPresentCapabilities)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDeviceGroupPresentCapabilitiesKHX(dump_inst, result, device, pDeviceGroupPresentCapabilities);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDeviceGroupPresentCapabilitiesKHX(dump_inst, result, device, pDeviceGroupPresentCapabilities);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDeviceGroupSurfacePresentModesKHX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSurfaceKHR surface, VkDeviceGroupPresentModeFlagsKHX* pModes)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDeviceGroupSurfacePresentModesKHX(dump_inst, result, device, surface, pModes);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDeviceGroupSurfacePresentModesKHX(dump_inst, result, device, surface, pModes);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPastPresentationTimingGOOGLE(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSwapchainKHR swapchain, uint32_t* pPresentationTimingCount, VkPastPresentationTimingGOOGLE* pPresentationTimings)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPastPresentationTimingGOOGLE(dump_inst, result, device, swapchain, pPresentationTimingCount, pPresentationTimings);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPastPresentationTimingGOOGLE(dump_inst, result, device, swapchain, pPresentationTimingCount, pPresentationTimings);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateBufferView(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkBufferViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBufferView* pView)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateBufferView(dump_inst, result, device, pCreateInfo, pAllocator, pView);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateBufferView(dump_inst, result, device, pCreateInfo, pAllocator, pView);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkAcquireNextImage2KHX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkAcquireNextImageInfoKHX* pAcquireInfo, uint32_t* pImageIndex)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkAcquireNextImage2KHX(dump_inst, result, device, pAcquireInfo, pImageIndex);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkAcquireNextImage2KHX(dump_inst, result, device, pAcquireInfo, pImageIndex);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkEnumerateInstanceExtensionProperties(ApiDumpInstance& dump_inst, VkResult result, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEnumerateInstanceExtensionProperties(dump_inst, result, pLayerName, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEnumerateInstanceExtensionProperties(dump_inst, result, pLayerName, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkImportSemaphoreWin32HandleKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkImportSemaphoreWin32HandleInfoKHR* pImportSemaphoreWin32HandleInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkImportSemaphoreWin32HandleKHR(dump_inst, result, device, pImportSemaphoreWin32HandleInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkImportSemaphoreWin32HandleKHR(dump_inst, result, device, pImportSemaphoreWin32HandleInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
inline void dump_vkQueueWaitIdle(ApiDumpInstance& dump_inst, VkResult result, VkQueue queue)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkQueueWaitIdle(dump_inst, result, queue);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkQueueWaitIdle(dump_inst, result, queue);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfaceFormatsKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfaceFormatsKHR(dump_inst, result, physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfaceFormatsKHR(dump_inst, result, physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkAcquireNextImageKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkAcquireNextImageKHR(dump_inst, result, device, swapchain, timeout, semaphore, fence, pImageIndex);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkAcquireNextImageKHR(dump_inst, result, device, swapchain, timeout, semaphore, fence, pImageIndex);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDevicePresentRectanglesKHX(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pRectCount, VkRect2D* pRects)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDevicePresentRectanglesKHX(dump_inst, result, physicalDevice, surface, pRectCount, pRects);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDevicePresentRectanglesKHX(dump_inst, result, physicalDevice, surface, pRectCount, pRects);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSurfacePresentModesKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSurfacePresentModesKHR(dump_inst, result, physicalDevice, surface, pPresentModeCount, pPresentModes);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSurfacePresentModesKHR(dump_inst, result, physicalDevice, surface, pPresentModeCount, pPresentModes);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPipelineCacheData(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkPipelineCache pipelineCache, size_t* pDataSize, void* pData)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPipelineCacheData(dump_inst, result, device, pipelineCache, pDataSize, pData);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPipelineCacheData(dump_inst, result, device, pipelineCache, pDataSize, pData);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkAllocateDescriptorSets(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkAllocateDescriptorSets(dump_inst, result, device, pAllocateInfo, pDescriptorSets);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkAllocateDescriptorSets(dump_inst, result, device, pAllocateInfo, pDescriptorSets);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkResetDescriptorPool(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkResetDescriptorPool(dump_inst, result, device, descriptorPool, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkResetDescriptorPool(dump_inst, result, device, descriptorPool, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDeviceWaitIdle(ApiDumpInstance& dump_inst, VkResult result, VkDevice device)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDeviceWaitIdle(dump_inst, result, device);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDeviceWaitIdle(dump_inst, result, device);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateFence(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateFence(dump_inst, result, device, pCreateInfo, pAllocator, pFence);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateFence(dump_inst, result, device, pCreateInfo, pAllocator, pFence);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetSwapchainImagesKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetSwapchainImagesKHR(dump_inst, result, device, swapchain, pSwapchainImageCount, pSwapchainImages);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetSwapchainImagesKHR(dump_inst, result, device, swapchain, pSwapchainImageCount, pSwapchainImages);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkEnumeratePhysicalDevices(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEnumeratePhysicalDevices(dump_inst, result, instance, pPhysicalDeviceCount, pPhysicalDevices);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEnumeratePhysicalDevices(dump_inst, result, instance, pPhysicalDeviceCount, pPhysicalDevices);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkMergePipelineCaches(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount, const VkPipelineCache* pSrcCaches)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkMergePipelineCaches(dump_inst, result, device, dstCache, srcCacheCount, pSrcCaches);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkMergePipelineCaches(dump_inst, result, device, dstCache, srcCacheCount, pSrcCaches);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_XLIB_KHR)
inline void dump_vkCreateXlibSurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkXlibSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateXlibSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateXlibSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_XLIB_KHR
inline void dump_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPlanePropertiesKHR* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(dump_inst, result, physicalDevice, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(dump_inst, result, physicalDevice, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkAllocateMemory(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkAllocateMemory(dump_inst, result, device, pAllocateInfo, pAllocator, pMemory);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkAllocateMemory(dump_inst, result, device, pAllocateInfo, pAllocator, pMemory);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateImage(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateImage(dump_inst, result, device, pCreateInfo, pAllocator, pImage);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateImage(dump_inst, result, device, pCreateInfo, pAllocator, pImage);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateGraphicsPipelines(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateGraphicsPipelines(dump_inst, result, device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateGraphicsPipelines(dump_inst, result, device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreatePipelineLayout(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreatePipelineLayout(dump_inst, result, device, pCreateInfo, pAllocator, pPipelineLayout);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreatePipelineLayout(dump_inst, result, device, pCreateInfo, pAllocator, pPipelineLayout);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDisplayPlaneSupportedDisplaysKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, uint32_t planeIndex, uint32_t* pDisplayCount, VkDisplayKHR* pDisplays)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDisplayPlaneSupportedDisplaysKHR(dump_inst, result, physicalDevice, planeIndex, pDisplayCount, pDisplays);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDisplayPlaneSupportedDisplaysKHR(dump_inst, result, physicalDevice, planeIndex, pDisplayCount, pDisplays);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkQueuePresentKHR(ApiDumpInstance& dump_inst, VkResult result, VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkQueuePresentKHR(dump_inst, result, queue, pPresentInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkQueuePresentKHR(dump_inst, result, queue, pPresentInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_XLIB_KHR)
inline void dump_vkGetPhysicalDeviceXlibPresentationSupportKHR(ApiDumpInstance& dump_inst, VkBool32 result, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, Display* dpy, VisualID visualID)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceXlibPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, dpy, visualID);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceXlibPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, dpy, visualID);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_XLIB_KHR
inline void dump_vkGetDisplayModePropertiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkDisplayKHR display, uint32_t* pPropertyCount, VkDisplayModePropertiesKHR* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDisplayModePropertiesKHR(dump_inst, result, physicalDevice, display, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDisplayModePropertiesKHR(dump_inst, result, physicalDevice, display, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkGetMemoryWin32HandleKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkMemoryGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetMemoryWin32HandleKHR(dump_inst, result, device, pGetWin32HandleInfo, pHandle);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetMemoryWin32HandleKHR(dump_inst, result, device, pGetWin32HandleInfo, pHandle);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
#if defined(VK_USE_PLATFORM_VI_NN)
inline void dump_vkCreateViSurfaceNN(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkViSurfaceCreateInfoNN* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateViSurfaceNN(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateViSurfaceNN(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_VI_NN
inline void dump_vkFreeDescriptorSets(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkFreeDescriptorSets(dump_inst, result, device, descriptorPool, descriptorSetCount, pDescriptorSets);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkFreeDescriptorSets(dump_inst, result, device, descriptorPool, descriptorSetCount, pDescriptorSets);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateDisplayModeKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkDisplayKHR display, const VkDisplayModeCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDisplayModeKHR* pMode)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDisplayModeKHR(dump_inst, result, physicalDevice, display, pCreateInfo, pAllocator, pMode);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDisplayModeKHR(dump_inst, result, physicalDevice, display, pCreateInfo, pAllocator, pMode);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkGetMemoryWin32HandlePropertiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkExternalMemoryHandleTypeFlagBitsKHR handleType, HANDLE handle, VkMemoryWin32HandlePropertiesKHR* pMemoryWin32HandleProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetMemoryWin32HandlePropertiesKHR(dump_inst, result, device, handleType, handle, pMemoryWin32HandleProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetMemoryWin32HandlePropertiesKHR(dump_inst, result, device, handleType, handle, pMemoryWin32HandleProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
inline void dump_vkMapMemory(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkMapMemory(dump_inst, result, device, memory, offset, size, flags, ppData);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkMapMemory(dump_inst, result, device, memory, offset, size, flags, ppData);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDisplayPlaneCapabilitiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode, uint32_t planeIndex, VkDisplayPlaneCapabilitiesKHR* pCapabilities)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDisplayPlaneCapabilitiesKHR(dump_inst, result, physicalDevice, mode, planeIndex, pCapabilities);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDisplayPlaneCapabilitiesKHR(dump_inst, result, physicalDevice, mode, planeIndex, pCapabilities);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateDescriptorUpdateTemplateKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorUpdateTemplateKHR* pDescriptorUpdateTemplate)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDescriptorUpdateTemplateKHR(dump_inst, result, device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDescriptorUpdateTemplateKHR(dump_inst, result, device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetSwapchainStatusKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkSwapchainKHR swapchain)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetSwapchainStatusKHR(dump_inst, result, device, swapchain);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetSwapchainStatusKHR(dump_inst, result, device, swapchain);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkCreateWin32SurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateWin32SurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateWin32SurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
inline void dump_vkFlushMappedMemoryRanges(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkFlushMappedMemoryRanges(dump_inst, result, device, memoryRangeCount, pMemoryRanges);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkFlushMappedMemoryRanges(dump_inst, result, device, memoryRangeCount, pMemoryRanges);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateCommandPool(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateCommandPool(dump_inst, result, device, pCreateInfo, pAllocator, pCommandPool);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateCommandPool(dump_inst, result, device, pCreateInfo, pAllocator, pCommandPool);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkImportFenceWin32HandleKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkImportFenceWin32HandleInfoKHR* pImportFenceWin32HandleInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkImportFenceWin32HandleKHR(dump_inst, result, device, pImportFenceWin32HandleInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkImportFenceWin32HandleKHR(dump_inst, result, device, pImportFenceWin32HandleInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
inline void dump_vkCreateDisplayPlaneSurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkDisplaySurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateDisplayPlaneSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateDisplayPlaneSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateSampler(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateSampler(dump_inst, result, device, pCreateInfo, pAllocator, pSampler);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateSampler(dump_inst, result, device, pCreateInfo, pAllocator, pSampler);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
inline void dump_vkCreateWaylandSurfaceKHR(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, const VkWaylandSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateWaylandSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateWaylandSurfaceKHR(dump_inst, result, instance, pCreateInfo, pAllocator, pSurface);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkGetPhysicalDeviceWin32PresentationSupportKHR(ApiDumpInstance& dump_inst, VkBool32 result, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceWin32PresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceWin32PresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
#if defined(VK_USE_PLATFORM_WIN32_KHR)
inline void dump_vkGetFenceWin32HandleKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkFenceGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetFenceWin32HandleKHR(dump_inst, result, device, pGetWin32HandleInfo, pHandle);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetFenceWin32HandleKHR(dump_inst, result, device, pGetWin32HandleInfo, pHandle);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WIN32_KHR
inline void dump_vkQueueBindSparse(ApiDumpInstance& dump_inst, VkResult result, VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo* pBindInfo, VkFence fence)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkQueueBindSparse(dump_inst, result, queue, bindInfoCount, pBindInfo, fence);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkQueueBindSparse(dump_inst, result, queue, bindInfoCount, pBindInfo, fence);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateIndirectCommandsLayoutNVX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkIndirectCommandsLayoutCreateInfoNVX* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkIndirectCommandsLayoutNVX* pIndirectCommandsLayout)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateIndirectCommandsLayoutNVX(dump_inst, result, device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateIndirectCommandsLayoutNVX(dump_inst, result, device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkInvalidateMappedMemoryRanges(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkInvalidateMappedMemoryRanges(dump_inst, result, device, memoryRangeCount, pMemoryRanges);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkInvalidateMappedMemoryRanges(dump_inst, result, device, memoryRangeCount, pMemoryRanges);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
inline void dump_vkGetPhysicalDeviceWaylandPresentationSupportKHR(ApiDumpInstance& dump_inst, VkBool32 result, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, struct wl_display* display)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceWaylandPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, display);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceWaylandPresentationSupportKHR(dump_inst, result, physicalDevice, queueFamilyIndex, display);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
#endif // VK_USE_PLATFORM_WAYLAND_KHR
inline void dump_vkCreateSwapchainKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateSwapchainKHR(dump_inst, result, device, pCreateInfo, pAllocator, pSwapchain);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateSwapchainKHR(dump_inst, result, device, pCreateInfo, pAllocator, pSwapchain);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetMemoryFdKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetMemoryFdKHR(dump_inst, result, device, pGetFdInfo, pFd);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetMemoryFdKHR(dump_inst, result, device, pGetFdInfo, pFd);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateObjectTableNVX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkObjectTableCreateInfoNVX* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkObjectTableNVX* pObjectTable)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateObjectTableNVX(dump_inst, result, device, pCreateInfo, pAllocator, pObjectTable);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateObjectTableNVX(dump_inst, result, device, pCreateInfo, pAllocator, pObjectTable);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateImageView(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateImageView(dump_inst, result, device, pCreateInfo, pAllocator, pView);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateImageView(dump_inst, result, device, pCreateInfo, pAllocator, pView);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetMemoryFdPropertiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkExternalMemoryHandleTypeFlagBitsKHR handleType, int fd, VkMemoryFdPropertiesKHR* pMemoryFdProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetMemoryFdPropertiesKHR(dump_inst, result, device, handleType, fd, pMemoryFdProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetMemoryFdPropertiesKHR(dump_inst, result, device, handleType, fd, pMemoryFdProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateFramebuffer(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateFramebuffer(dump_inst, result, device, pCreateInfo, pAllocator, pFramebuffer);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateFramebuffer(dump_inst, result, device, pCreateInfo, pAllocator, pFramebuffer);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkResetFences(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t fenceCount, const VkFence* pFences)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkResetFences(dump_inst, result, device, fenceCount, pFences);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkResetFences(dump_inst, result, device, fenceCount, pFences);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkUnregisterObjectsNVX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkObjectTableNVX objectTable, uint32_t objectCount, const VkObjectEntryTypeNVX* pObjectEntryTypes, const uint32_t* pObjectIndices)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkUnregisterObjectsNVX(dump_inst, result, device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkUnregisterObjectsNVX(dump_inst, result, device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkRegisterObjectsNVX(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkObjectTableNVX objectTable, uint32_t objectCount, const VkObjectTableEntryNVX* const*    ppObjectTableEntries, const uint32_t* pObjectIndices)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkRegisterObjectsNVX(dump_inst, result, device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkRegisterObjectsNVX(dump_inst, result, device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkBindBufferMemory(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkBindBufferMemory(dump_inst, result, device, buffer, memory, memoryOffset);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkBindBufferMemory(dump_inst, result, device, buffer, memory, memoryOffset);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetFenceStatus(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkFence fence)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetFenceStatus(dump_inst, result, device, fence);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetFenceStatus(dump_inst, result, device, fence);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDebugMarkerSetObjectTagEXT(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDebugMarkerObjectTagInfoEXT* pTagInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDebugMarkerSetObjectTagEXT(dump_inst, result, device, pTagInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDebugMarkerSetObjectTagEXT(dump_inst, result, device, pTagInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateSharedSwapchainsKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t swapchainCount, const VkSwapchainCreateInfoKHR* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchains)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateSharedSwapchainsKHR(dump_inst, result, device, swapchainCount, pCreateInfos, pAllocator, pSwapchains);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateSharedSwapchainsKHR(dump_inst, result, device, swapchainCount, pCreateInfos, pAllocator, pSwapchains);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkImportFenceFdKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkImportFenceFdInfoKHR* pImportFenceFdInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkImportFenceFdKHR(dump_inst, result, device, pImportFenceFdInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkImportFenceFdKHR(dump_inst, result, device, pImportFenceFdInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkExternalMemoryHandleTypeFlagsNV externalHandleType, VkExternalImageFormatPropertiesNV* pExternalImageFormatProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(dump_inst, result, physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(dump_inst, result, physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkWaitForFences(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkWaitForFences(dump_inst, result, device, fenceCount, pFences, waitAll, timeout);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkWaitForFences(dump_inst, result, device, fenceCount, pFences, waitAll, timeout);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceImageFormatProperties(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* pImageFormatProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceImageFormatProperties(dump_inst, result, physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceImageFormatProperties(dump_inst, result, physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkBindImageMemory(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkBindImageMemory(dump_inst, result, device, image, memory, memoryOffset);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkBindImageMemory(dump_inst, result, device, image, memory, memoryOffset);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCreateSemaphore(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCreateSemaphore(dump_inst, result, device, pCreateInfo, pAllocator, pSemaphore);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCreateSemaphore(dump_inst, result, device, pCreateInfo, pAllocator, pSemaphore);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkResetCommandPool(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkResetCommandPool(dump_inst, result, device, commandPool, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkResetCommandPool(dump_inst, result, device, commandPool, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetFenceFdKHR(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkFenceGetFdInfoKHR* pGetFdInfo, int* pFd)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetFenceFdKHR(dump_inst, result, device, pGetFdInfo, pFd);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetFenceFdKHR(dump_inst, result, device, pGetFdInfo, pFd);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDebugMarkerSetObjectNameEXT(ApiDumpInstance& dump_inst, VkResult result, VkDevice device, const VkDebugMarkerObjectNameInfoEXT* pNameInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDebugMarkerSetObjectNameEXT(dump_inst, result, device, pNameInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDebugMarkerSetObjectNameEXT(dump_inst, result, device, pNameInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceDisplayPropertiesKHR(ApiDumpInstance& dump_inst, VkResult result, VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPropertiesKHR* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceDisplayPropertiesKHR(dump_inst, result, physicalDevice, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceDisplayPropertiesKHR(dump_inst, result, physicalDevice, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkEnumeratePhysicalDeviceGroupsKHX(ApiDumpInstance& dump_inst, VkResult result, VkInstance instance, uint32_t* pPhysicalDeviceGroupCount, VkPhysicalDeviceGroupPropertiesKHX* pPhysicalDeviceGroupProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkEnumeratePhysicalDeviceGroupsKHX(dump_inst, result, instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkEnumeratePhysicalDeviceGroupsKHX(dump_inst, result, instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}


inline void dump_vkCmdSetDepthBias(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetDepthBias(dump_inst, commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetDepthBias(dump_inst, commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdCopyImageToBuffer(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdCopyImageToBuffer(dump_inst, commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdCopyImageToBuffer(dump_inst, commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroySemaphore(ApiDumpInstance& dump_inst, VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroySemaphore(dump_inst, device, semaphore, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroySemaphore(dump_inst, device, semaphore, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDrawIndexedIndirect(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDrawIndexedIndirect(dump_inst, commandBuffer, buffer, offset, drawCount, stride);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDrawIndexedIndirect(dump_inst, commandBuffer, buffer, offset, drawCount, stride);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyQueryPool(ApiDumpInstance& dump_inst, VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyQueryPool(dump_inst, device, queryPool, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyQueryPool(dump_inst, device, queryPool, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetBlendConstants(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, const float blendConstants[4])
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetBlendConstants(dump_inst, commandBuffer, blendConstants);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetBlendConstants(dump_inst, commandBuffer, blendConstants);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetEvent(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetEvent(dump_inst, commandBuffer, event, stageMask);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetEvent(dump_inst, commandBuffer, event, stageMask);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkFreeCommandBuffers(ApiDumpInstance& dump_inst, VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkFreeCommandBuffers(dump_inst, device, commandPool, commandBufferCount, pCommandBuffers);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkFreeCommandBuffers(dump_inst, device, commandPool, commandBufferCount, pCommandBuffers);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetDepthBounds(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, float minDepthBounds, float maxDepthBounds)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetDepthBounds(dump_inst, commandBuffer, minDepthBounds, maxDepthBounds);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetDepthBounds(dump_inst, commandBuffer, minDepthBounds, maxDepthBounds);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDispatch(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDispatch(dump_inst, commandBuffer, groupCountX, groupCountY, groupCountZ);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDispatch(dump_inst, commandBuffer, groupCountX, groupCountY, groupCountZ);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetViewportWScalingNV(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewportWScalingNV* pViewportWScalings)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetViewportWScalingNV(dump_inst, commandBuffer, firstViewport, viewportCount, pViewportWScalings);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetViewportWScalingNV(dump_inst, commandBuffer, firstViewport, viewportCount, pViewportWScalings);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyFramebuffer(ApiDumpInstance& dump_inst, VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyFramebuffer(dump_inst, device, framebuffer, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyFramebuffer(dump_inst, device, framebuffer, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDispatchIndirect(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDispatchIndirect(dump_inst, commandBuffer, buffer, offset);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDispatchIndirect(dump_inst, commandBuffer, buffer, offset);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroySampler(ApiDumpInstance& dump_inst, VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroySampler(dump_inst, device, sampler, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroySampler(dump_inst, device, sampler, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdUpdateBuffer(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const void* pData)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdUpdateBuffer(dump_inst, commandBuffer, dstBuffer, dstOffset, dataSize, pData);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdUpdateBuffer(dump_inst, commandBuffer, dstBuffer, dstOffset, dataSize, pData);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetStencilCompareMask(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t compareMask)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetStencilCompareMask(dump_inst, commandBuffer, faceMask, compareMask);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetStencilCompareMask(dump_inst, commandBuffer, faceMask, compareMask);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroySurfaceKHR(ApiDumpInstance& dump_inst, VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroySurfaceKHR(dump_inst, instance, surface, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroySurfaceKHR(dump_inst, instance, surface, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalSemaphoreInfoKHR* pExternalSemaphoreInfo, VkExternalSemaphorePropertiesKHR* pExternalSemaphoreProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(dump_inst, physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(dump_inst, physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdCopyBuffer(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdCopyBuffer(dump_inst, commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdCopyBuffer(dump_inst, commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdFillBuffer(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdFillBuffer(dump_inst, commandBuffer, dstBuffer, dstOffset, size, data);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdFillBuffer(dump_inst, commandBuffer, dstBuffer, dstOffset, size, data);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdResetEvent(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdResetEvent(dump_inst, commandBuffer, event, stageMask);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdResetEvent(dump_inst, commandBuffer, event, stageMask);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyEvent(ApiDumpInstance& dump_inst, VkDevice device, VkEvent event, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyEvent(dump_inst, device, event, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyEvent(dump_inst, device, event, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdCopyImage(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdCopyImage(dump_inst, commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdCopyImage(dump_inst, commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetStencilWriteMask(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t writeMask)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetStencilWriteMask(dump_inst, commandBuffer, faceMask, writeMask);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetStencilWriteMask(dump_inst, commandBuffer, faceMask, writeMask);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdWaitEvents(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdWaitEvents(dump_inst, commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdWaitEvents(dump_inst, commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdClearColorImage(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearColorValue* pColor, uint32_t rangeCount, const VkImageSubresourceRange* pRanges)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdClearColorImage(dump_inst, commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdClearColorImage(dump_inst, commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyDescriptorSetLayout(ApiDumpInstance& dump_inst, VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyDescriptorSetLayout(dump_inst, device, descriptorSetLayout, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyDescriptorSetLayout(dump_inst, device, descriptorSetLayout, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDeviceQueue(ApiDumpInstance& dump_inst, VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDeviceQueue(dump_inst, device, queueFamilyIndex, queueIndex, pQueue);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDeviceQueue(dump_inst, device, queueFamilyIndex, queueIndex, pQueue);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetStencilReference(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t reference)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetStencilReference(dump_inst, commandBuffer, faceMask, reference);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetStencilReference(dump_inst, commandBuffer, faceMask, reference);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyImageView(ApiDumpInstance& dump_inst, VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyImageView(dump_inst, device, imageView, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyImageView(dump_inst, device, imageView, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBindDescriptorSets(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBindDescriptorSets(dump_inst, commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBindDescriptorSets(dump_inst, commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBlitImage(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, VkFilter filter)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBlitImage(dump_inst, commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBlitImage(dump_inst, commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdClearDepthStencilImage(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const VkImageSubresourceRange* pRanges)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdClearDepthStencilImage(dump_inst, commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdClearDepthStencilImage(dump_inst, commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBindIndexBuffer(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBindIndexBuffer(dump_inst, commandBuffer, buffer, offset, indexType);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBindIndexBuffer(dump_inst, commandBuffer, buffer, offset, indexType);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceExternalBufferPropertiesKHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfoKHR* pExternalBufferInfo, VkExternalBufferPropertiesKHR* pExternalBufferProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceExternalBufferPropertiesKHR(dump_inst, physicalDevice, pExternalBufferInfo, pExternalBufferProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceExternalBufferPropertiesKHR(dump_inst, physicalDevice, pExternalBufferInfo, pExternalBufferProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDeviceGroupPeerMemoryFeaturesKHX(ApiDumpInstance& dump_inst, VkDevice device, uint32_t heapIndex, uint32_t localDeviceIndex, uint32_t remoteDeviceIndex, VkPeerMemoryFeatureFlagsKHX* pPeerMemoryFeatures)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDeviceGroupPeerMemoryFeaturesKHX(dump_inst, device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDeviceGroupPeerMemoryFeaturesKHX(dump_inst, device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyDevice(ApiDumpInstance& dump_inst, VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyDevice(dump_inst, device, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyDevice(dump_inst, device, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceFeatures2KHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2KHR* pFeatures)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceFeatures2KHR(dump_inst, physicalDevice, pFeatures);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceFeatures2KHR(dump_inst, physicalDevice, pFeatures);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBindVertexBuffers(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBindVertexBuffers(dump_inst, commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBindVertexBuffers(dump_inst, commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdPushDescriptorSetKHR(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdPushDescriptorSetKHR(dump_inst, commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdPushDescriptorSetKHR(dump_inst, commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyBuffer(ApiDumpInstance& dump_inst, VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyBuffer(dump_inst, device, buffer, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyBuffer(dump_inst, device, buffer, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdClearAttachments(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t rectCount, const VkClearRect* pRects)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdClearAttachments(dump_inst, commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdClearAttachments(dump_inst, commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdPipelineBarrier(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdPipelineBarrier(dump_inst, commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdPipelineBarrier(dump_inst, commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetDeviceMaskKHX(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t deviceMask)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetDeviceMaskKHX(dump_inst, commandBuffer, deviceMask);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetDeviceMaskKHX(dump_inst, commandBuffer, deviceMask);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBindPipeline(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBindPipeline(dump_inst, commandBuffer, pipelineBindPoint, pipeline);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBindPipeline(dump_inst, commandBuffer, pipelineBindPoint, pipeline);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceProperties2KHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2KHR* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceProperties2KHR(dump_inst, physicalDevice, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceProperties2KHR(dump_inst, physicalDevice, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyShaderModule(ApiDumpInstance& dump_inst, VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyShaderModule(dump_inst, device, shaderModule, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyShaderModule(dump_inst, device, shaderModule, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDraw(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDraw(dump_inst, commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDraw(dump_inst, commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetViewport(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewport* pViewports)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetViewport(dump_inst, commandBuffer, firstViewport, viewportCount, pViewports);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetViewport(dump_inst, commandBuffer, firstViewport, viewportCount, pViewports);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdCopyBufferToImage(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdCopyBufferToImage(dump_inst, commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdCopyBufferToImage(dump_inst, commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyDebugReportCallbackEXT(ApiDumpInstance& dump_inst, VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyDebugReportCallbackEXT(dump_inst, instance, callback, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyDebugReportCallbackEXT(dump_inst, instance, callback, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDrawIndexed(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDrawIndexed(dump_inst, commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDrawIndexed(dump_inst, commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceFormatProperties2KHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2KHR* pFormatProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceFormatProperties2KHR(dump_inst, physicalDevice, format, pFormatProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceFormatProperties2KHR(dump_inst, physicalDevice, format, pFormatProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetLineWidth(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, float lineWidth)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetLineWidth(dump_inst, commandBuffer, lineWidth);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetLineWidth(dump_inst, commandBuffer, lineWidth);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDebugReportMessageEXT(ApiDumpInstance& dump_inst, VkInstance instance, VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDebugReportMessageEXT(dump_inst, instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDebugReportMessageEXT(dump_inst, instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetScissor(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetScissor(dump_inst, commandBuffer, firstScissor, scissorCount, pScissors);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetScissor(dump_inst, commandBuffer, firstScissor, scissorCount, pScissors);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdResolveImage(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageResolve* pRegions)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdResolveImage(dump_inst, commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdResolveImage(dump_inst, commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceQueueFamilyProperties2KHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties2KHR* pQueueFamilyProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceQueueFamilyProperties2KHR(dump_inst, physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceQueueFamilyProperties2KHR(dump_inst, physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceMemoryProperties2KHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2KHR* pMemoryProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceMemoryProperties2KHR(dump_inst, physicalDevice, pMemoryProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceMemoryProperties2KHR(dump_inst, physicalDevice, pMemoryProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyPipeline(ApiDumpInstance& dump_inst, VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyPipeline(dump_inst, device, pipeline, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyPipeline(dump_inst, device, pipeline, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDrawIndirect(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDrawIndirect(dump_inst, commandBuffer, buffer, offset, drawCount, stride);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDrawIndirect(dump_inst, commandBuffer, buffer, offset, drawCount, stride);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDispatchBaseKHX(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDispatchBaseKHX(dump_inst, commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDispatchBaseKHX(dump_inst, commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyPipelineCache(ApiDumpInstance& dump_inst, VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyPipelineCache(dump_inst, device, pipelineCache, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyPipelineCache(dump_inst, device, pipelineCache, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetImageMemoryRequirements(ApiDumpInstance& dump_inst, VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetImageMemoryRequirements(dump_inst, device, image, pMemoryRequirements);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetImageMemoryRequirements(dump_inst, device, image, pMemoryRequirements);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyDescriptorPool(ApiDumpInstance& dump_inst, VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyDescriptorPool(dump_inst, device, descriptorPool, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyDescriptorPool(dump_inst, device, descriptorPool, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBeginQuery(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBeginQuery(dump_inst, commandBuffer, queryPool, query, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBeginQuery(dump_inst, commandBuffer, queryPool, query, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2KHR* pFormatInfo, uint32_t* pPropertyCount, VkSparseImageFormatProperties2KHR* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(dump_inst, physicalDevice, pFormatInfo, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(dump_inst, physicalDevice, pFormatInfo, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDebugMarkerEndEXT(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDebugMarkerEndEXT(dump_inst, commandBuffer);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDebugMarkerEndEXT(dump_inst, commandBuffer);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkSetHdrMetadataEXT(ApiDumpInstance& dump_inst, VkDevice device, uint32_t swapchainCount, const VkSwapchainKHR* pSwapchains, const VkHdrMetadataEXT* pMetadata)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkSetHdrMetadataEXT(dump_inst, device, swapchainCount, pSwapchains, pMetadata);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkSetHdrMetadataEXT(dump_inst, device, swapchainCount, pSwapchains, pMetadata);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDebugMarkerInsertEXT(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT* pMarkerInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDebugMarkerInsertEXT(dump_inst, commandBuffer, pMarkerInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDebugMarkerInsertEXT(dump_inst, commandBuffer, pMarkerInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyInstance(ApiDumpInstance& dump_inst, VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyInstance(dump_inst, instance, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyInstance(dump_inst, instance, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyBufferView(ApiDumpInstance& dump_inst, VkDevice device, VkBufferView bufferView, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyBufferView(dump_inst, device, bufferView, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyBufferView(dump_inst, device, bufferView, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceFeatures(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures* pFeatures)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceFeatures(dump_inst, physicalDevice, pFeatures);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceFeatures(dump_inst, physicalDevice, pFeatures);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetImageSparseMemoryRequirements(ApiDumpInstance& dump_inst, VkDevice device, VkImage image, uint32_t* pSparseMemoryRequirementCount, VkSparseImageMemoryRequirements* pSparseMemoryRequirements)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetImageSparseMemoryRequirements(dump_inst, device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetImageSparseMemoryRequirements(dump_inst, device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdEndQuery(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdEndQuery(dump_inst, commandBuffer, queryPool, query);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdEndQuery(dump_inst, commandBuffer, queryPool, query);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdResetQueryPool(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdResetQueryPool(dump_inst, commandBuffer, queryPool, firstQuery, queryCount);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdResetQueryPool(dump_inst, commandBuffer, queryPool, firstQuery, queryCount);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceFormatProperties(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties* pFormatProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceFormatProperties(dump_inst, physicalDevice, format, pFormatProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceFormatProperties(dump_inst, physicalDevice, format, pFormatProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyImage(ApiDumpInstance& dump_inst, VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyImage(dump_inst, device, image, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyImage(dump_inst, device, image, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyRenderPass(ApiDumpInstance& dump_inst, VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyRenderPass(dump_inst, device, renderPass, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyRenderPass(dump_inst, device, renderPass, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyPipelineLayout(ApiDumpInstance& dump_inst, VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyPipelineLayout(dump_inst, device, pipelineLayout, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyPipelineLayout(dump_inst, device, pipelineLayout, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdWriteTimestamp(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage, VkQueryPool queryPool, uint32_t query)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdWriteTimestamp(dump_inst, commandBuffer, pipelineStage, queryPool, query);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdWriteTimestamp(dump_inst, commandBuffer, pipelineStage, queryPool, query);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkUnmapMemory(ApiDumpInstance& dump_inst, VkDevice device, VkDeviceMemory memory)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkUnmapMemory(dump_inst, device, memory);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkUnmapMemory(dump_inst, device, memory);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkFreeMemory(ApiDumpInstance& dump_inst, VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkFreeMemory(dump_inst, device, memory, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkFreeMemory(dump_inst, device, memory, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkUpdateDescriptorSets(ApiDumpInstance& dump_inst, VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkUpdateDescriptorSets(dump_inst, device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkUpdateDescriptorSets(dump_inst, device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdPushConstants(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdPushConstants(dump_inst, commandBuffer, layout, stageFlags, offset, size, pValues);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdPushConstants(dump_inst, commandBuffer, layout, stageFlags, offset, size, pValues);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceSparseImageFormatProperties(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageTiling tiling, uint32_t* pPropertyCount, VkSparseImageFormatProperties* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceSparseImageFormatProperties(dump_inst, physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceSparseImageFormatProperties(dump_inst, physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdCopyQueryPoolResults(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdCopyQueryPoolResults(dump_inst, commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdCopyQueryPoolResults(dump_inst, commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdProcessCommandsNVX(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, const VkCmdProcessCommandsInfoNVX* pProcessCommandsInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdProcessCommandsNVX(dump_inst, commandBuffer, pProcessCommandsInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdProcessCommandsNVX(dump_inst, commandBuffer, pProcessCommandsInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDrawIndirectCountAMD(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDrawIndirectCountAMD(dump_inst, commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDrawIndirectCountAMD(dump_inst, commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdReserveSpaceForCommandsNVX(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, const VkCmdReserveSpaceForCommandsInfoNVX* pReserveSpaceInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdReserveSpaceForCommandsNVX(dump_inst, commandBuffer, pReserveSpaceInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdReserveSpaceForCommandsNVX(dump_inst, commandBuffer, pReserveSpaceInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdBeginRenderPass(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin, VkSubpassContents contents)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdBeginRenderPass(dump_inst, commandBuffer, pRenderPassBegin, contents);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdBeginRenderPass(dump_inst, commandBuffer, pRenderPassBegin, contents);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceProperties(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceProperties(dump_inst, physicalDevice, pProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceProperties(dump_inst, physicalDevice, pProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetRenderAreaGranularity(ApiDumpInstance& dump_inst, VkDevice device, VkRenderPass renderPass, VkExtent2D* pGranularity)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetRenderAreaGranularity(dump_inst, device, renderPass, pGranularity);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetRenderAreaGranularity(dump_inst, device, renderPass, pGranularity);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDrawIndexedIndirectCountAMD(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDrawIndexedIndirectCountAMD(dump_inst, commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDrawIndexedIndirectCountAMD(dump_inst, commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyDescriptorUpdateTemplateKHR(ApiDumpInstance& dump_inst, VkDevice device, VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyDescriptorUpdateTemplateKHR(dump_inst, device, descriptorUpdateTemplate, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyDescriptorUpdateTemplateKHR(dump_inst, device, descriptorUpdateTemplate, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceQueueFamilyProperties(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceQueueFamilyProperties(dump_inst, physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceQueueFamilyProperties(dump_inst, physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyIndirectCommandsLayoutNVX(ApiDumpInstance& dump_inst, VkDevice device, VkIndirectCommandsLayoutNVX indirectCommandsLayout, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyIndirectCommandsLayoutNVX(dump_inst, device, indirectCommandsLayout, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyIndirectCommandsLayoutNVX(dump_inst, device, indirectCommandsLayout, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyFence(ApiDumpInstance& dump_inst, VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyFence(dump_inst, device, fence, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyFence(dump_inst, device, fence, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetImageMemoryRequirements2KHR(ApiDumpInstance& dump_inst, VkDevice device, const VkImageMemoryRequirementsInfo2KHR* pInfo, VkMemoryRequirements2KHR* pMemoryRequirements)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetImageMemoryRequirements2KHR(dump_inst, device, pInfo, pMemoryRequirements);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetImageMemoryRequirements2KHR(dump_inst, device, pInfo, pMemoryRequirements);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyObjectTableNVX(ApiDumpInstance& dump_inst, VkDevice device, VkObjectTableNVX objectTable, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyObjectTableNVX(dump_inst, device, objectTable, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyObjectTableNVX(dump_inst, device, objectTable, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetImageSubresourceLayout(ApiDumpInstance& dump_inst, VkDevice device, VkImage image, const VkImageSubresource* pSubresource, VkSubresourceLayout* pLayout)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetImageSubresourceLayout(dump_inst, device, image, pSubresource, pLayout);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetImageSubresourceLayout(dump_inst, device, image, pSubresource, pLayout);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkUpdateDescriptorSetWithTemplateKHR(ApiDumpInstance& dump_inst, VkDevice device, VkDescriptorSet descriptorSet, VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, const void* pData)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkUpdateDescriptorSetWithTemplateKHR(dump_inst, device, descriptorSet, descriptorUpdateTemplate, pData);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkUpdateDescriptorSetWithTemplateKHR(dump_inst, device, descriptorSet, descriptorUpdateTemplate, pData);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkTrimCommandPoolKHR(ApiDumpInstance& dump_inst, VkDevice device, VkCommandPool commandPool, VkCommandPoolTrimFlagsKHR flags)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkTrimCommandPoolKHR(dump_inst, device, commandPool, flags);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkTrimCommandPoolKHR(dump_inst, device, commandPool, flags);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetBufferMemoryRequirements2KHR(ApiDumpInstance& dump_inst, VkDevice device, const VkBufferMemoryRequirementsInfo2KHR* pInfo, VkMemoryRequirements2KHR* pMemoryRequirements)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetBufferMemoryRequirements2KHR(dump_inst, device, pInfo, pMemoryRequirements);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetBufferMemoryRequirements2KHR(dump_inst, device, pInfo, pMemoryRequirements);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroySwapchainKHR(ApiDumpInstance& dump_inst, VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroySwapchainKHR(dump_inst, device, swapchain, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroySwapchainKHR(dump_inst, device, swapchain, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdPushDescriptorSetWithTemplateKHR(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, VkPipelineLayout layout, uint32_t set, const void* pData)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdPushDescriptorSetWithTemplateKHR(dump_inst, commandBuffer, descriptorUpdateTemplate, layout, set, pData);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdPushDescriptorSetWithTemplateKHR(dump_inst, commandBuffer, descriptorUpdateTemplate, layout, set, pData);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetDeviceMemoryCommitment(ApiDumpInstance& dump_inst, VkDevice device, VkDeviceMemory memory, VkDeviceSize* pCommittedMemoryInBytes)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetDeviceMemoryCommitment(dump_inst, device, memory, pCommittedMemoryInBytes);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetDeviceMemoryCommitment(dump_inst, device, memory, pCommittedMemoryInBytes);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdNextSubpass(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, VkSubpassContents contents)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdNextSubpass(dump_inst, commandBuffer, contents);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdNextSubpass(dump_inst, commandBuffer, contents);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdSetDiscardRectangleEXT(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t firstDiscardRectangle, uint32_t discardRectangleCount, const VkRect2D* pDiscardRectangles)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdSetDiscardRectangleEXT(dump_inst, commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdSetDiscardRectangleEXT(dump_inst, commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetImageSparseMemoryRequirements2KHR(ApiDumpInstance& dump_inst, VkDevice device, const VkImageSparseMemoryRequirementsInfo2KHR* pInfo, uint32_t* pSparseMemoryRequirementCount, VkSparseImageMemoryRequirements2KHR* pSparseMemoryRequirements)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetImageSparseMemoryRequirements2KHR(dump_inst, device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetImageSparseMemoryRequirements2KHR(dump_inst, device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkDestroyCommandPool(ApiDumpInstance& dump_inst, VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkDestroyCommandPool(dump_inst, device, commandPool, pAllocator);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkDestroyCommandPool(dump_inst, device, commandPool, pAllocator);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceExternalFencePropertiesKHR(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalFenceInfoKHR* pExternalFenceInfo, VkExternalFencePropertiesKHR* pExternalFenceProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceExternalFencePropertiesKHR(dump_inst, physicalDevice, pExternalFenceInfo, pExternalFenceProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceExternalFencePropertiesKHR(dump_inst, physicalDevice, pExternalFenceInfo, pExternalFenceProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdEndRenderPass(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdEndRenderPass(dump_inst, commandBuffer);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdEndRenderPass(dump_inst, commandBuffer);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdExecuteCommands(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdExecuteCommands(dump_inst, commandBuffer, commandBufferCount, pCommandBuffers);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdExecuteCommands(dump_inst, commandBuffer, commandBufferCount, pCommandBuffers);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkDeviceGeneratedCommandsFeaturesNVX* pFeatures, VkDeviceGeneratedCommandsLimitsNVX* pLimits)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX(dump_inst, physicalDevice, pFeatures, pLimits);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX(dump_inst, physicalDevice, pFeatures, pLimits);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetPhysicalDeviceMemoryProperties(ApiDumpInstance& dump_inst, VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetPhysicalDeviceMemoryProperties(dump_inst, physicalDevice, pMemoryProperties);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetPhysicalDeviceMemoryProperties(dump_inst, physicalDevice, pMemoryProperties);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkGetBufferMemoryRequirements(ApiDumpInstance& dump_inst, VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkGetBufferMemoryRequirements(dump_inst, device, buffer, pMemoryRequirements);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkGetBufferMemoryRequirements(dump_inst, device, buffer, pMemoryRequirements);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}
inline void dump_vkCmdDebugMarkerBeginEXT(ApiDumpInstance& dump_inst, VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT* pMarkerInfo)
{
    loader_platform_thread_lock_mutex(dump_inst.outputMutex());
    switch(dump_inst.settings().format())
    {
    case ApiDumpFormat::Text:
        dump_text_vkCmdDebugMarkerBeginEXT(dump_inst, commandBuffer, pMarkerInfo);
        break;
    case ApiDumpFormat::Html:
        dump_html_vkCmdDebugMarkerBeginEXT(dump_inst, commandBuffer, pMarkerInfo);
        break;
    }
    loader_platform_thread_unlock_mutex(dump_inst.outputMutex());
}

//============================= API EntryPoints =============================//

// Specifically implemented functions

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
    // Get the function pointer
    VkLayerInstanceCreateInfo* chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    assert(chain_info->u.pLayerInfo != 0);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    assert(fpGetInstanceProcAddr != 0);
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance) fpGetInstanceProcAddr(NULL, "vkCreateInstance");
    if(fpCreateInstance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Call the function and create the dispatch table
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;
    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if(result == VK_SUCCESS) {
        initInstanceTable(*pInstance, fpGetInstanceProcAddr);
    }

    

    // Output the API dump
    dump_vkCreateInstance(ApiDumpInstance::current(), result, pCreateInfo, pAllocator, pInstance);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    // Destroy the dispatch table
    dispatch_key key = get_dispatch_key(instance);
    instance_dispatch_table(instance)->DestroyInstance(instance, pAllocator);
    destroy_instance_dispatch_table(key);

    

    // Output the API dump
    dump_vkDestroyInstance(ApiDumpInstance::current(), instance, pAllocator);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice)
{
    // Get the function pointer
    VkLayerDeviceCreateInfo* chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    assert(chain_info->u.pLayerInfo != 0);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice) fpGetInstanceProcAddr(NULL, "vkCreateDevice");
    if(fpCreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Call the function and create the dispatch table
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;
    VkResult result = fpCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if(result == VK_SUCCESS) {
        initDeviceTable(*pDevice, fpGetDeviceProcAddr);
    }

    

    // Output the API dump
    dump_vkCreateDevice(ApiDumpInstance::current(), result, physicalDevice, pCreateInfo, pAllocator, pDevice);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    // Destroy the dispatch table
    dispatch_key key = get_dispatch_key(device);
    device_dispatch_table(device)->DestroyDevice(device, pAllocator);
    destroy_device_dispatch_table(key);

    

    // Output the API dump
    dump_vkDestroyDevice(ApiDumpInstance::current(), device, pAllocator);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties)
{
    return util_GetExtensionProperties(0, NULL, pPropertyCount, pProperties);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    static const VkLayerProperties layerProperties[] = {
        {
            "VK_LAYER_LUNARG_api_dump",
            VK_MAKE_VERSION(1, 0, VK_HEADER_VERSION), // specVersion
            VK_MAKE_VERSION(0, 2, 0), // implementationVersion
            "layer: api_dump",
        }
    };

    return util_GetLayerProperties(ARRAY_SIZE(layerProperties), layerProperties, pPropertyCount, pProperties);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkLayerProperties* pProperties)
{
    static const VkLayerProperties layerProperties[] = {
        {
            "VK_LAYER_LUNARG_api_dump",
            VK_MAKE_VERSION(1, 0, VK_HEADER_VERSION),
            VK_MAKE_VERSION(0, 2, 0),
            "layer: api_dump",
        }
    };

    return util_GetLayerProperties(ARRAY_SIZE(layerProperties), layerProperties, pPropertyCount, pProperties);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    VkResult result = device_dispatch_table(queue)->QueuePresentKHR(queue, pPresentInfo);
    
    dump_vkQueuePresentKHR(ApiDumpInstance::current(), result, queue, pPresentInfo);
    ApiDumpInstance::current().nextFrame();
    return result;
}

// Autogen instance functions

#if defined(VK_USE_PLATFORM_MIR_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateMirSurfaceKHR(VkInstance instance, const VkMirSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateMirSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateMirSurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_MIR_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilities2KHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, VkSurfaceCapabilities2KHR* pSurfaceCapabilities)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice, pSurfaceInfo, pSurfaceCapabilities);
    
    dump_vkGetPhysicalDeviceSurfaceCapabilities2KHR(ApiDumpInstance::current(), result, physicalDevice, pSurfaceInfo, pSurfaceCapabilities);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkReleaseDisplayEXT(VkPhysicalDevice physicalDevice, VkDisplayKHR display)
{
    VkResult result = instance_dispatch_table(physicalDevice)->ReleaseDisplayEXT(physicalDevice, display);
    
    dump_vkReleaseDisplayEXT(ApiDumpInstance::current(), result, physicalDevice, display);
    return result;
}
#if defined(VK_USE_PLATFORM_MIR_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceMirPresentationSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, MirConnection* connection)
{
    VkBool32 result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceMirPresentationSupportKHR(physicalDevice, queueFamilyIndex, connection);
    
    dump_vkGetPhysicalDeviceMirPresentationSupportKHR(ApiDumpInstance::current(), result, physicalDevice, queueFamilyIndex, connection);
    return result;
}
#endif // VK_USE_PLATFORM_MIR_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormats2KHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, uint32_t* pSurfaceFormatCount, VkSurfaceFormat2KHR* pSurfaceFormats)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfaceFormats2KHR(physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats);
    
    dump_vkGetPhysicalDeviceSurfaceFormats2KHR(ApiDumpInstance::current(), result, physicalDevice, pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats);
    return result;
}
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateAndroidSurfaceKHR(VkInstance instance, const VkAndroidSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateAndroidSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateAndroidSurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_ANDROID_KHR
#if defined(VK_USE_PLATFORM_XLIB_XRANDR_EXT)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAcquireXlibDisplayEXT(VkPhysicalDevice physicalDevice, Display* dpy, VkDisplayKHR display)
{
    VkResult result = instance_dispatch_table(physicalDevice)->AcquireXlibDisplayEXT(physicalDevice, dpy, display);
    
    dump_vkAcquireXlibDisplayEXT(ApiDumpInstance::current(), result, physicalDevice, dpy, display);
    return result;
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#if defined(VK_USE_PLATFORM_IOS_MVK)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateIOSSurfaceMVK(VkInstance instance, const VkIOSSurfaceCreateInfoMVK* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateIOSSurfaceMVK(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateIOSSurfaceMVK(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_IOS_MVK
#if defined(VK_USE_PLATFORM_XLIB_XRANDR_EXT)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetRandROutputDisplayEXT(VkPhysicalDevice physicalDevice, Display* dpy, RROutput rrOutput, VkDisplayKHR* pDisplay)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetRandROutputDisplayEXT(physicalDevice, dpy, rrOutput, pDisplay);
    
    dump_vkGetRandROutputDisplayEXT(ApiDumpInstance::current(), result, physicalDevice, dpy, rrOutput, pDisplay);
    return result;
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#if defined(VK_USE_PLATFORM_XCB_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateXcbSurfaceKHR(VkInstance instance, const VkXcbSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateXcbSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateXcbSurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_XCB_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkBool32* pSupported)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);
    
    dump_vkGetPhysicalDeviceSurfaceSupportKHR(ApiDumpInstance::current(), result, physicalDevice, queueFamilyIndex, surface, pSupported);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback)
{
    VkResult result = instance_dispatch_table(instance)->CreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pCallback);
    
    dump_vkCreateDebugReportCallbackEXT(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pCallback);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* pSurfaceCapabilities)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);
    
    dump_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ApiDumpInstance::current(), result, physicalDevice, surface, pSurfaceCapabilities);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilities2EXT(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilities2EXT* pSurfaceCapabilities)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfaceCapabilities2EXT(physicalDevice, surface, pSurfaceCapabilities);
    
    dump_vkGetPhysicalDeviceSurfaceCapabilities2EXT(ApiDumpInstance::current(), result, physicalDevice, surface, pSurfaceCapabilities);
    return result;
}
#if defined(VK_USE_PLATFORM_MACOS_MVK)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateMacOSSurfaceMVK(VkInstance instance, const VkMacOSSurfaceCreateInfoMVK* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateMacOSSurfaceMVK(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateMacOSSurfaceMVK(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_MACOS_MVK
#if defined(VK_USE_PLATFORM_XCB_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceXcbPresentationSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, xcb_connection_t* connection, xcb_visualid_t visual_id)
{
    VkBool32 result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceXcbPresentationSupportKHR(physicalDevice, queueFamilyIndex, connection, visual_id);
    
    dump_vkGetPhysicalDeviceXcbPresentationSupportKHR(ApiDumpInstance::current(), result, physicalDevice, queueFamilyIndex, connection, visual_id);
    return result;
}
#endif // VK_USE_PLATFORM_XCB_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2KHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2KHR* pImageFormatInfo, VkImageFormatProperties2KHR* pImageFormatProperties)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceImageFormatProperties2KHR(physicalDevice, pImageFormatInfo, pImageFormatProperties);
    
    dump_vkGetPhysicalDeviceImageFormatProperties2KHR(ApiDumpInstance::current(), result, physicalDevice, pImageFormatInfo, pImageFormatProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
    
    dump_vkGetPhysicalDeviceSurfaceFormatsKHR(ApiDumpInstance::current(), result, physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDevicePresentRectanglesKHX(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pRectCount, VkRect2D* pRects)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDevicePresentRectanglesKHX(physicalDevice, surface, pRectCount, pRects);
    
    dump_vkGetPhysicalDevicePresentRectanglesKHX(ApiDumpInstance::current(), result, physicalDevice, surface, pRectCount, pRects);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes);
    
    dump_vkGetPhysicalDeviceSurfacePresentModesKHR(ApiDumpInstance::current(), result, physicalDevice, surface, pPresentModeCount, pPresentModes);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices)
{
    VkResult result = instance_dispatch_table(instance)->EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    
    dump_vkEnumeratePhysicalDevices(ApiDumpInstance::current(), result, instance, pPhysicalDeviceCount, pPhysicalDevices);
    return result;
}
#if defined(VK_USE_PLATFORM_XLIB_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateXlibSurfaceKHR(VkInstance instance, const VkXlibSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateXlibSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateXlibSurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_XLIB_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceDisplayPlanePropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPlanePropertiesKHR* pProperties)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, pPropertyCount, pProperties);
    
    dump_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(ApiDumpInstance::current(), result, physicalDevice, pPropertyCount, pProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetDisplayPlaneSupportedDisplaysKHR(VkPhysicalDevice physicalDevice, uint32_t planeIndex, uint32_t* pDisplayCount, VkDisplayKHR* pDisplays)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, pDisplayCount, pDisplays);
    
    dump_vkGetDisplayPlaneSupportedDisplaysKHR(ApiDumpInstance::current(), result, physicalDevice, planeIndex, pDisplayCount, pDisplays);
    return result;
}
#if defined(VK_USE_PLATFORM_XLIB_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceXlibPresentationSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, Display* dpy, VisualID visualID)
{
    VkBool32 result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceXlibPresentationSupportKHR(physicalDevice, queueFamilyIndex, dpy, visualID);
    
    dump_vkGetPhysicalDeviceXlibPresentationSupportKHR(ApiDumpInstance::current(), result, physicalDevice, queueFamilyIndex, dpy, visualID);
    return result;
}
#endif // VK_USE_PLATFORM_XLIB_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetDisplayModePropertiesKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display, uint32_t* pPropertyCount, VkDisplayModePropertiesKHR* pProperties)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetDisplayModePropertiesKHR(physicalDevice, display, pPropertyCount, pProperties);
    
    dump_vkGetDisplayModePropertiesKHR(ApiDumpInstance::current(), result, physicalDevice, display, pPropertyCount, pProperties);
    return result;
}
#if defined(VK_USE_PLATFORM_VI_NN)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateViSurfaceNN(VkInstance instance, const VkViSurfaceCreateInfoNN* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateViSurfaceNN(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateViSurfaceNN(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_VI_NN
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDisplayModeKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display, const VkDisplayModeCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDisplayModeKHR* pMode)
{
    VkResult result = instance_dispatch_table(physicalDevice)->CreateDisplayModeKHR(physicalDevice, display, pCreateInfo, pAllocator, pMode);
    
    dump_vkCreateDisplayModeKHR(ApiDumpInstance::current(), result, physicalDevice, display, pCreateInfo, pAllocator, pMode);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetDisplayPlaneCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode, uint32_t planeIndex, VkDisplayPlaneCapabilitiesKHR* pCapabilities)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetDisplayPlaneCapabilitiesKHR(physicalDevice, mode, planeIndex, pCapabilities);
    
    dump_vkGetDisplayPlaneCapabilitiesKHR(ApiDumpInstance::current(), result, physicalDevice, mode, planeIndex, pCapabilities);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateWin32SurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDisplayPlaneSurfaceKHR(VkInstance instance, const VkDisplaySurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateDisplayPlaneSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateDisplayPlaneSurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateWaylandSurfaceKHR(VkInstance instance, const VkWaylandSurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
    VkResult result = instance_dispatch_table(instance)->CreateWaylandSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    
    dump_vkCreateWaylandSurfaceKHR(ApiDumpInstance::current(), result, instance, pCreateInfo, pAllocator, pSurface);
    return result;
}
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceWin32PresentationSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex)
{
    VkBool32 result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, queueFamilyIndex);
    
    dump_vkGetPhysicalDeviceWin32PresentationSupportKHR(ApiDumpInstance::current(), result, physicalDevice, queueFamilyIndex);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceWaylandPresentationSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, struct wl_display* display)
{
    VkBool32 result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceWaylandPresentationSupportKHR(physicalDevice, queueFamilyIndex, display);
    
    dump_vkGetPhysicalDeviceWaylandPresentationSupportKHR(ApiDumpInstance::current(), result, physicalDevice, queueFamilyIndex, display);
    return result;
}
#endif // VK_USE_PLATFORM_WAYLAND_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceExternalImageFormatPropertiesNV(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkExternalMemoryHandleTypeFlagsNV externalHandleType, VkExternalImageFormatPropertiesNV* pExternalImageFormatProperties)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceExternalImageFormatPropertiesNV(physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);
    
    dump_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(ApiDumpInstance::current(), result, physicalDevice, format, type, tiling, usage, flags, externalHandleType, pExternalImageFormatProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, VkImageFormatProperties* pImageFormatProperties)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceImageFormatProperties(physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
    
    dump_vkGetPhysicalDeviceImageFormatProperties(ApiDumpInstance::current(), result, physicalDevice, format, type, tiling, usage, flags, pImageFormatProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceDisplayPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkDisplayPropertiesKHR* pProperties)
{
    VkResult result = instance_dispatch_table(physicalDevice)->GetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, pPropertyCount, pProperties);
    
    dump_vkGetPhysicalDeviceDisplayPropertiesKHR(ApiDumpInstance::current(), result, physicalDevice, pPropertyCount, pProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDeviceGroupsKHX(VkInstance instance, uint32_t* pPhysicalDeviceGroupCount, VkPhysicalDeviceGroupPropertiesKHX* pPhysicalDeviceGroupProperties)
{
    VkResult result = instance_dispatch_table(instance)->EnumeratePhysicalDeviceGroupsKHX(instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties);
    
    dump_vkEnumeratePhysicalDeviceGroupsKHX(ApiDumpInstance::current(), result, instance, pPhysicalDeviceGroupCount, pPhysicalDeviceGroupProperties);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks* pAllocator)
{
    instance_dispatch_table(instance)->DestroySurfaceKHR(instance, surface, pAllocator);
    
    dump_vkDestroySurfaceKHR(ApiDumpInstance::current(), instance, surface, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalSemaphoreInfoKHR* pExternalSemaphoreInfo, VkExternalSemaphorePropertiesKHR* pExternalSemaphoreProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceExternalSemaphorePropertiesKHR(physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties);
    
    dump_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(ApiDumpInstance::current(), physicalDevice, pExternalSemaphoreInfo, pExternalSemaphoreProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalBufferPropertiesKHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfoKHR* pExternalBufferInfo, VkExternalBufferPropertiesKHR* pExternalBufferProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceExternalBufferPropertiesKHR(physicalDevice, pExternalBufferInfo, pExternalBufferProperties);
    
    dump_vkGetPhysicalDeviceExternalBufferPropertiesKHR(ApiDumpInstance::current(), physicalDevice, pExternalBufferInfo, pExternalBufferProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2KHR* pFeatures)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceFeatures2KHR(physicalDevice, pFeatures);
    
    dump_vkGetPhysicalDeviceFeatures2KHR(ApiDumpInstance::current(), physicalDevice, pFeatures);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2KHR* pProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceProperties2KHR(physicalDevice, pProperties);
    
    dump_vkGetPhysicalDeviceProperties2KHR(ApiDumpInstance::current(), physicalDevice, pProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator)
{
    instance_dispatch_table(instance)->DestroyDebugReportCallbackEXT(instance, callback, pAllocator);
    
    dump_vkDestroyDebugReportCallbackEXT(ApiDumpInstance::current(), instance, callback, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2KHR(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2KHR* pFormatProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceFormatProperties2KHR(physicalDevice, format, pFormatProperties);
    
    dump_vkGetPhysicalDeviceFormatProperties2KHR(ApiDumpInstance::current(), physicalDevice, format, pFormatProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDebugReportMessageEXT(VkInstance instance, VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage)
{
    instance_dispatch_table(instance)->DebugReportMessageEXT(instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage);
    
    dump_vkDebugReportMessageEXT(ApiDumpInstance::current(), instance, flags, objectType, object, location, messageCode, pLayerPrefix, pMessage);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2KHR(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties2KHR* pQueueFamilyProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
    
    dump_vkGetPhysicalDeviceQueueFamilyProperties2KHR(ApiDumpInstance::current(), physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2KHR* pMemoryProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceMemoryProperties2KHR(physicalDevice, pMemoryProperties);
    
    dump_vkGetPhysicalDeviceMemoryProperties2KHR(ApiDumpInstance::current(), physicalDevice, pMemoryProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties2KHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2KHR* pFormatInfo, uint32_t* pPropertyCount, VkSparseImageFormatProperties2KHR* pProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSparseImageFormatProperties2KHR(physicalDevice, pFormatInfo, pPropertyCount, pProperties);
    
    dump_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(ApiDumpInstance::current(), physicalDevice, pFormatInfo, pPropertyCount, pProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures* pFeatures)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceFeatures(physicalDevice, pFeatures);
    
    dump_vkGetPhysicalDeviceFeatures(ApiDumpInstance::current(), physicalDevice, pFeatures);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties* pFormatProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceFormatProperties(physicalDevice, format, pFormatProperties);
    
    dump_vkGetPhysicalDeviceFormatProperties(ApiDumpInstance::current(), physicalDevice, format, pFormatProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageTiling tiling, uint32_t* pPropertyCount, VkSparseImageFormatProperties* pProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties);
    
    dump_vkGetPhysicalDeviceSparseImageFormatProperties(ApiDumpInstance::current(), physicalDevice, format, type, samples, usage, tiling, pPropertyCount, pProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceProperties(physicalDevice, pProperties);
    
    dump_vkGetPhysicalDeviceProperties(ApiDumpInstance::current(), physicalDevice, pProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
    
    dump_vkGetPhysicalDeviceQueueFamilyProperties(ApiDumpInstance::current(), physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalFencePropertiesKHR(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalFenceInfoKHR* pExternalFenceInfo, VkExternalFencePropertiesKHR* pExternalFenceProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceExternalFencePropertiesKHR(physicalDevice, pExternalFenceInfo, pExternalFenceProperties);
    
    dump_vkGetPhysicalDeviceExternalFencePropertiesKHR(ApiDumpInstance::current(), physicalDevice, pExternalFenceInfo, pExternalFenceProperties);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX(VkPhysicalDevice physicalDevice, VkDeviceGeneratedCommandsFeaturesNVX* pFeatures, VkDeviceGeneratedCommandsLimitsNVX* pLimits)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceGeneratedCommandsPropertiesNVX(physicalDevice, pFeatures, pLimits);
    
    dump_vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX(ApiDumpInstance::current(), physicalDevice, pFeatures, pLimits);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties)
{
    instance_dispatch_table(physicalDevice)->GetPhysicalDeviceMemoryProperties(physicalDevice, pMemoryProperties);
    
    dump_vkGetPhysicalDeviceMemoryProperties(ApiDumpInstance::current(), physicalDevice, pMemoryProperties);
}

// Autogen device functions

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers)
{
    VkResult result = device_dispatch_table(device)->AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if(result == VK_SUCCESS)
ApiDumpInstance::current().addCmdBuffers(
device,
pAllocateInfo->commandPool,
std::vector<VkCommandBuffer>(pCommandBuffers, pCommandBuffers + pAllocateInfo->commandBufferCount),
pAllocateInfo->level
);
    dump_vkAllocateCommandBuffers(ApiDumpInstance::current(), result, device, pAllocateInfo, pCommandBuffers);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreWin32HandleKHR(VkDevice device, const VkSemaphoreGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
{
    VkResult result = device_dispatch_table(device)->GetSemaphoreWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
    
    dump_vkGetSemaphoreWin32HandleKHR(ApiDumpInstance::current(), result, device, pGetWin32HandleInfo, pHandle);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
    VkResult result = device_dispatch_table(device)->GetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
    
    dump_vkGetQueryPoolResults(ApiDumpInstance::current(), result, device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(VkDevice device, const VkEventCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkEvent* pEvent)
{
    VkResult result = device_dispatch_table(device)->CreateEvent(device, pCreateInfo, pAllocator, pEvent);
    
    dump_vkCreateEvent(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pEvent);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer)
{
    VkResult result = device_dispatch_table(device)->CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
    
    dump_vkCreateBuffer(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pBuffer);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo)
{
    VkResult result = device_dispatch_table(commandBuffer)->BeginCommandBuffer(commandBuffer, pBeginInfo);
    
    dump_vkBeginCommandBuffer(ApiDumpInstance::current(), result, commandBuffer, pBeginInfo);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout)
{
    VkResult result = device_dispatch_table(device)->CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);
    
    dump_vkCreateDescriptorSetLayout(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pSetLayout);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
    VkResult result = device_dispatch_table(device)->CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    
    dump_vkCreateRenderPass(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pRenderPass);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryWin32HandleNV(VkDevice device, VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagsNV handleType, HANDLE* pHandle)
{
    VkResult result = device_dispatch_table(device)->GetMemoryWin32HandleNV(device, memory, handleType, pHandle);
    
    dump_vkGetMemoryWin32HandleNV(ApiDumpInstance::current(), result, device, memory, handleType, pHandle);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDisplayPowerControlEXT(VkDevice device, VkDisplayKHR display, const VkDisplayPowerInfoEXT* pDisplayPowerInfo)
{
    VkResult result = device_dispatch_table(device)->DisplayPowerControlEXT(device, display, pDisplayPowerInfo);
    
    dump_vkDisplayPowerControlEXT(ApiDumpInstance::current(), result, device, display, pDisplayPowerInfo);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkRegisterDeviceEventEXT(VkDevice device, const VkDeviceEventInfoEXT* pDeviceEventInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
    VkResult result = device_dispatch_table(device)->RegisterDeviceEventEXT(device, pDeviceEventInfo, pAllocator, pFence);
    
    dump_vkRegisterDeviceEventEXT(ApiDumpInstance::current(), result, device, pDeviceEventInfo, pAllocator, pFence);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice device, VkEvent event)
{
    VkResult result = device_dispatch_table(device)->SetEvent(device, event);
    
    dump_vkSetEvent(ApiDumpInstance::current(), result, device, event);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(VkDevice device, VkEvent event)
{
    VkResult result = device_dispatch_table(device)->GetEventStatus(device, event);
    
    dump_vkGetEventStatus(ApiDumpInstance::current(), result, device, event);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkRegisterDisplayEventEXT(VkDevice device, VkDisplayKHR display, const VkDisplayEventInfoEXT* pDisplayEventInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
    VkResult result = device_dispatch_table(device)->RegisterDisplayEventEXT(device, display, pDisplayEventInfo, pAllocator, pFence);
    
    dump_vkRegisterDisplayEventEXT(ApiDumpInstance::current(), result, device, display, pDisplayEventInfo, pAllocator, pFence);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer commandBuffer)
{
    VkResult result = device_dispatch_table(commandBuffer)->EndCommandBuffer(commandBuffer);
    
    dump_vkEndCommandBuffer(ApiDumpInstance::current(), result, commandBuffer);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence)
{
    VkResult result = device_dispatch_table(queue)->QueueSubmit(queue, submitCount, pSubmits, fence);
    
    dump_vkQueueSubmit(ApiDumpInstance::current(), result, queue, submitCount, pSubmits, fence);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainCounterEXT(VkDevice device, VkSwapchainKHR swapchain, VkSurfaceCounterFlagBitsEXT counter, uint64_t* pCounterValue)
{
    VkResult result = device_dispatch_table(device)->GetSwapchainCounterEXT(device, swapchain, counter, pCounterValue);
    
    dump_vkGetSwapchainCounterEXT(ApiDumpInstance::current(), result, device, swapchain, counter, pCounterValue);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule)
{
    VkResult result = device_dispatch_table(device)->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    
    dump_vkCreateShaderModule(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pShaderModule);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkImportSemaphoreFdKHR(VkDevice device, const VkImportSemaphoreFdInfoKHR* pImportSemaphoreFdInfo)
{
    VkResult result = device_dispatch_table(device)->ImportSemaphoreFdKHR(device, pImportSemaphoreFdInfo);
    
    dump_vkImportSemaphoreFdKHR(ApiDumpInstance::current(), result, device, pImportSemaphoreFdInfo);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice device, VkEvent event)
{
    VkResult result = device_dispatch_table(device)->ResetEvent(device, event);
    
    dump_vkResetEvent(ApiDumpInstance::current(), result, device, event);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    VkResult result = device_dispatch_table(device)->CreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    
    dump_vkCreateComputePipelines(ApiDumpInstance::current(), result, device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
    VkResult result = device_dispatch_table(device)->CreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);
    
    dump_vkCreateQueryPool(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pQueryPool);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreFdKHR(VkDevice device, const VkSemaphoreGetFdInfoKHR* pGetFdInfo, int* pFd)
{
    VkResult result = device_dispatch_table(device)->GetSemaphoreFdKHR(device, pGetFdInfo, pFd);
    
    dump_vkGetSemaphoreFdKHR(ApiDumpInstance::current(), result, device, pGetFdInfo, pFd);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory2KHX(VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfoKHX* pBindInfos)
{
    VkResult result = device_dispatch_table(device)->BindBufferMemory2KHX(device, bindInfoCount, pBindInfos);
    
    dump_vkBindBufferMemory2KHX(ApiDumpInstance::current(), result, device, bindInfoCount, pBindInfos);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags)
{
    VkResult result = device_dispatch_table(commandBuffer)->ResetCommandBuffer(commandBuffer, flags);
    
    dump_vkResetCommandBuffer(ApiDumpInstance::current(), result, commandBuffer, flags);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetRefreshCycleDurationGOOGLE(VkDevice device, VkSwapchainKHR swapchain, VkRefreshCycleDurationGOOGLE* pDisplayTimingProperties)
{
    VkResult result = device_dispatch_table(device)->GetRefreshCycleDurationGOOGLE(device, swapchain, pDisplayTimingProperties);
    
    dump_vkGetRefreshCycleDurationGOOGLE(ApiDumpInstance::current(), result, device, swapchain, pDisplayTimingProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory2KHX(VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfoKHX* pBindInfos)
{
    VkResult result = device_dispatch_table(device)->BindImageMemory2KHX(device, bindInfoCount, pBindInfos);
    
    dump_vkBindImageMemory2KHX(ApiDumpInstance::current(), result, device, bindInfoCount, pBindInfos);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool)
{
    VkResult result = device_dispatch_table(device)->CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);
    
    dump_vkCreateDescriptorPool(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pDescriptorPool);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache)
{
    VkResult result = device_dispatch_table(device)->CreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache);
    
    dump_vkCreatePipelineCache(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pPipelineCache);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetDeviceGroupPresentCapabilitiesKHX(VkDevice device, VkDeviceGroupPresentCapabilitiesKHX* pDeviceGroupPresentCapabilities)
{
    VkResult result = device_dispatch_table(device)->GetDeviceGroupPresentCapabilitiesKHX(device, pDeviceGroupPresentCapabilities);
    
    dump_vkGetDeviceGroupPresentCapabilitiesKHX(ApiDumpInstance::current(), result, device, pDeviceGroupPresentCapabilities);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetDeviceGroupSurfacePresentModesKHX(VkDevice device, VkSurfaceKHR surface, VkDeviceGroupPresentModeFlagsKHX* pModes)
{
    VkResult result = device_dispatch_table(device)->GetDeviceGroupSurfacePresentModesKHX(device, surface, pModes);
    
    dump_vkGetDeviceGroupSurfacePresentModesKHX(ApiDumpInstance::current(), result, device, surface, pModes);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPastPresentationTimingGOOGLE(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pPresentationTimingCount, VkPastPresentationTimingGOOGLE* pPresentationTimings)
{
    VkResult result = device_dispatch_table(device)->GetPastPresentationTimingGOOGLE(device, swapchain, pPresentationTimingCount, pPresentationTimings);
    
    dump_vkGetPastPresentationTimingGOOGLE(ApiDumpInstance::current(), result, device, swapchain, pPresentationTimingCount, pPresentationTimings);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(VkDevice device, const VkBufferViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBufferView* pView)
{
    VkResult result = device_dispatch_table(device)->CreateBufferView(device, pCreateInfo, pAllocator, pView);
    
    dump_vkCreateBufferView(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pView);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImage2KHX(VkDevice device, const VkAcquireNextImageInfoKHX* pAcquireInfo, uint32_t* pImageIndex)
{
    VkResult result = device_dispatch_table(device)->AcquireNextImage2KHX(device, pAcquireInfo, pImageIndex);
    
    dump_vkAcquireNextImage2KHX(ApiDumpInstance::current(), result, device, pAcquireInfo, pImageIndex);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkImportSemaphoreWin32HandleKHR(VkDevice device, const VkImportSemaphoreWin32HandleInfoKHR* pImportSemaphoreWin32HandleInfo)
{
    VkResult result = device_dispatch_table(device)->ImportSemaphoreWin32HandleKHR(device, pImportSemaphoreWin32HandleInfo);
    
    dump_vkImportSemaphoreWin32HandleKHR(ApiDumpInstance::current(), result, device, pImportSemaphoreWin32HandleInfo);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue queue)
{
    VkResult result = device_dispatch_table(queue)->QueueWaitIdle(queue);
    
    dump_vkQueueWaitIdle(ApiDumpInstance::current(), result, queue);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex)
{
    VkResult result = device_dispatch_table(device)->AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);
    
    dump_vkAcquireNextImageKHR(ApiDumpInstance::current(), result, device, swapchain, timeout, semaphore, fence, pImageIndex);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice device, VkPipelineCache pipelineCache, size_t* pDataSize, void* pData)
{
    VkResult result = device_dispatch_table(device)->GetPipelineCacheData(device, pipelineCache, pDataSize, pData);
    
    dump_vkGetPipelineCacheData(ApiDumpInstance::current(), result, device, pipelineCache, pDataSize, pData);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    VkResult result = device_dispatch_table(device)->AllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets);
    
    dump_vkAllocateDescriptorSets(ApiDumpInstance::current(), result, device, pAllocateInfo, pDescriptorSets);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags)
{
    VkResult result = device_dispatch_table(device)->ResetDescriptorPool(device, descriptorPool, flags);
    
    dump_vkResetDescriptorPool(ApiDumpInstance::current(), result, device, descriptorPool, flags);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice device)
{
    VkResult result = device_dispatch_table(device)->DeviceWaitIdle(device);
    
    dump_vkDeviceWaitIdle(ApiDumpInstance::current(), result, device);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
    VkResult result = device_dispatch_table(device)->CreateFence(device, pCreateInfo, pAllocator, pFence);
    
    dump_vkCreateFence(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pFence);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages)
{
    VkResult result = device_dispatch_table(device)->GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
    
    dump_vkGetSwapchainImagesKHR(ApiDumpInstance::current(), result, device, swapchain, pSwapchainImageCount, pSwapchainImages);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMergePipelineCaches(VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount, const VkPipelineCache* pSrcCaches)
{
    VkResult result = device_dispatch_table(device)->MergePipelineCaches(device, dstCache, srcCacheCount, pSrcCaches);
    
    dump_vkMergePipelineCaches(ApiDumpInstance::current(), result, device, dstCache, srcCacheCount, pSrcCaches);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory)
{
    VkResult result = device_dispatch_table(device)->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    
    dump_vkAllocateMemory(ApiDumpInstance::current(), result, device, pAllocateInfo, pAllocator, pMemory);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
    VkResult result = device_dispatch_table(device)->CreateImage(device, pCreateInfo, pAllocator, pImage);
    
    dump_vkCreateImage(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pImage);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    VkResult result = device_dispatch_table(device)->CreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    
    dump_vkCreateGraphicsPipelines(ApiDumpInstance::current(), result, device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout)
{
    VkResult result = device_dispatch_table(device)->CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);
    
    dump_vkCreatePipelineLayout(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pPipelineLayout);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryWin32HandleKHR(VkDevice device, const VkMemoryGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
{
    VkResult result = device_dispatch_table(device)->GetMemoryWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
    
    dump_vkGetMemoryWin32HandleKHR(ApiDumpInstance::current(), result, device, pGetWin32HandleInfo, pHandle);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    VkResult result = device_dispatch_table(device)->FreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets);
    
    dump_vkFreeDescriptorSets(ApiDumpInstance::current(), result, device, descriptorPool, descriptorSetCount, pDescriptorSets);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryWin32HandlePropertiesKHR(VkDevice device, VkExternalMemoryHandleTypeFlagBitsKHR handleType, HANDLE handle, VkMemoryWin32HandlePropertiesKHR* pMemoryWin32HandleProperties)
{
    VkResult result = device_dispatch_table(device)->GetMemoryWin32HandlePropertiesKHR(device, handleType, handle, pMemoryWin32HandleProperties);
    
    dump_vkGetMemoryWin32HandlePropertiesKHR(ApiDumpInstance::current(), result, device, handleType, handle, pMemoryWin32HandleProperties);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData)
{
    VkResult result = device_dispatch_table(device)->MapMemory(device, memory, offset, size, flags, ppData);
    
    dump_vkMapMemory(ApiDumpInstance::current(), result, device, memory, offset, size, flags, ppData);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorUpdateTemplateKHR(VkDevice device, const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorUpdateTemplateKHR* pDescriptorUpdateTemplate)
{
    VkResult result = device_dispatch_table(device)->CreateDescriptorUpdateTemplateKHR(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
    
    dump_vkCreateDescriptorUpdateTemplateKHR(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainStatusKHR(VkDevice device, VkSwapchainKHR swapchain)
{
    VkResult result = device_dispatch_table(device)->GetSwapchainStatusKHR(device, swapchain);
    
    dump_vkGetSwapchainStatusKHR(ApiDumpInstance::current(), result, device, swapchain);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges)
{
    VkResult result = device_dispatch_table(device)->FlushMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);
    
    dump_vkFlushMappedMemoryRanges(ApiDumpInstance::current(), result, device, memoryRangeCount, pMemoryRanges);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool)
{
    VkResult result = device_dispatch_table(device)->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
    
    dump_vkCreateCommandPool(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pCommandPool);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkImportFenceWin32HandleKHR(VkDevice device, const VkImportFenceWin32HandleInfoKHR* pImportFenceWin32HandleInfo)
{
    VkResult result = device_dispatch_table(device)->ImportFenceWin32HandleKHR(device, pImportFenceWin32HandleInfo);
    
    dump_vkImportFenceWin32HandleKHR(ApiDumpInstance::current(), result, device, pImportFenceWin32HandleInfo);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
    VkResult result = device_dispatch_table(device)->CreateSampler(device, pCreateInfo, pAllocator, pSampler);
    
    dump_vkCreateSampler(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pSampler);
    return result;
}
#if defined(VK_USE_PLATFORM_WIN32_KHR)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceWin32HandleKHR(VkDevice device, const VkFenceGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
{
    VkResult result = device_dispatch_table(device)->GetFenceWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
    
    dump_vkGetFenceWin32HandleKHR(ApiDumpInstance::current(), result, device, pGetWin32HandleInfo, pHandle);
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueBindSparse(VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo* pBindInfo, VkFence fence)
{
    VkResult result = device_dispatch_table(queue)->QueueBindSparse(queue, bindInfoCount, pBindInfo, fence);
    
    dump_vkQueueBindSparse(ApiDumpInstance::current(), result, queue, bindInfoCount, pBindInfo, fence);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateIndirectCommandsLayoutNVX(VkDevice device, const VkIndirectCommandsLayoutCreateInfoNVX* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkIndirectCommandsLayoutNVX* pIndirectCommandsLayout)
{
    VkResult result = device_dispatch_table(device)->CreateIndirectCommandsLayoutNVX(device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
    
    dump_vkCreateIndirectCommandsLayoutNVX(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges)
{
    VkResult result = device_dispatch_table(device)->InvalidateMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);
    
    dump_vkInvalidateMappedMemoryRanges(ApiDumpInstance::current(), result, device, memoryRangeCount, pMemoryRanges);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    VkResult result = device_dispatch_table(device)->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    
    dump_vkCreateSwapchainKHR(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pSwapchain);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd)
{
    VkResult result = device_dispatch_table(device)->GetMemoryFdKHR(device, pGetFdInfo, pFd);
    
    dump_vkGetMemoryFdKHR(ApiDumpInstance::current(), result, device, pGetFdInfo, pFd);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateObjectTableNVX(VkDevice device, const VkObjectTableCreateInfoNVX* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkObjectTableNVX* pObjectTable)
{
    VkResult result = device_dispatch_table(device)->CreateObjectTableNVX(device, pCreateInfo, pAllocator, pObjectTable);
    
    dump_vkCreateObjectTableNVX(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pObjectTable);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
    VkResult result = device_dispatch_table(device)->CreateImageView(device, pCreateInfo, pAllocator, pView);
    
    dump_vkCreateImageView(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pView);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryFdPropertiesKHR(VkDevice device, VkExternalMemoryHandleTypeFlagBitsKHR handleType, int fd, VkMemoryFdPropertiesKHR* pMemoryFdProperties)
{
    VkResult result = device_dispatch_table(device)->GetMemoryFdPropertiesKHR(device, handleType, fd, pMemoryFdProperties);
    
    dump_vkGetMemoryFdPropertiesKHR(ApiDumpInstance::current(), result, device, handleType, fd, pMemoryFdProperties);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer)
{
    VkResult result = device_dispatch_table(device)->CreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer);
    
    dump_vkCreateFramebuffer(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pFramebuffer);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences)
{
    VkResult result = device_dispatch_table(device)->ResetFences(device, fenceCount, pFences);
    
    dump_vkResetFences(ApiDumpInstance::current(), result, device, fenceCount, pFences);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkUnregisterObjectsNVX(VkDevice device, VkObjectTableNVX objectTable, uint32_t objectCount, const VkObjectEntryTypeNVX* pObjectEntryTypes, const uint32_t* pObjectIndices)
{
    VkResult result = device_dispatch_table(device)->UnregisterObjectsNVX(device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);
    
    dump_vkUnregisterObjectsNVX(ApiDumpInstance::current(), result, device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkRegisterObjectsNVX(VkDevice device, VkObjectTableNVX objectTable, uint32_t objectCount, const VkObjectTableEntryNVX* const*    ppObjectTableEntries, const uint32_t* pObjectIndices)
{
    VkResult result = device_dispatch_table(device)->RegisterObjectsNVX(device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);
    
    dump_vkRegisterObjectsNVX(ApiDumpInstance::current(), result, device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    VkResult result = device_dispatch_table(device)->BindBufferMemory(device, buffer, memory, memoryOffset);
    
    dump_vkBindBufferMemory(ApiDumpInstance::current(), result, device, buffer, memory, memoryOffset);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice device, VkFence fence)
{
    VkResult result = device_dispatch_table(device)->GetFenceStatus(device, fence);
    
    dump_vkGetFenceStatus(ApiDumpInstance::current(), result, device, fence);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDebugMarkerSetObjectTagEXT(VkDevice device, const VkDebugMarkerObjectTagInfoEXT* pTagInfo)
{
    VkResult result = device_dispatch_table(device)->DebugMarkerSetObjectTagEXT(device, pTagInfo);
    
    dump_vkDebugMarkerSetObjectTagEXT(ApiDumpInstance::current(), result, device, pTagInfo);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSharedSwapchainsKHR(VkDevice device, uint32_t swapchainCount, const VkSwapchainCreateInfoKHR* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchains)
{
    VkResult result = device_dispatch_table(device)->CreateSharedSwapchainsKHR(device, swapchainCount, pCreateInfos, pAllocator, pSwapchains);
    
    dump_vkCreateSharedSwapchainsKHR(ApiDumpInstance::current(), result, device, swapchainCount, pCreateInfos, pAllocator, pSwapchains);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkImportFenceFdKHR(VkDevice device, const VkImportFenceFdInfoKHR* pImportFenceFdInfo)
{
    VkResult result = device_dispatch_table(device)->ImportFenceFdKHR(device, pImportFenceFdInfo);
    
    dump_vkImportFenceFdKHR(ApiDumpInstance::current(), result, device, pImportFenceFdInfo);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout)
{
    VkResult result = device_dispatch_table(device)->WaitForFences(device, fenceCount, pFences, waitAll, timeout);
    
    dump_vkWaitForFences(ApiDumpInstance::current(), result, device, fenceCount, pFences, waitAll, timeout);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    VkResult result = device_dispatch_table(device)->BindImageMemory(device, image, memory, memoryOffset);
    
    dump_vkBindImageMemory(ApiDumpInstance::current(), result, device, image, memory, memoryOffset);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore)
{
    VkResult result = device_dispatch_table(device)->CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
    
    dump_vkCreateSemaphore(ApiDumpInstance::current(), result, device, pCreateInfo, pAllocator, pSemaphore);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags)
{
    VkResult result = device_dispatch_table(device)->ResetCommandPool(device, commandPool, flags);
    
    dump_vkResetCommandPool(ApiDumpInstance::current(), result, device, commandPool, flags);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceFdKHR(VkDevice device, const VkFenceGetFdInfoKHR* pGetFdInfo, int* pFd)
{
    VkResult result = device_dispatch_table(device)->GetFenceFdKHR(device, pGetFdInfo, pFd);
    
    dump_vkGetFenceFdKHR(ApiDumpInstance::current(), result, device, pGetFdInfo, pFd);
    return result;
}
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDebugMarkerSetObjectNameEXT(VkDevice device, const VkDebugMarkerObjectNameInfoEXT* pNameInfo)
{
    VkResult result = device_dispatch_table(device)->DebugMarkerSetObjectNameEXT(device, pNameInfo);
    
    dump_vkDebugMarkerSetObjectNameEXT(ApiDumpInstance::current(), result, device, pNameInfo);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBias(VkCommandBuffer commandBuffer, float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor)
{
    device_dispatch_table(commandBuffer)->CmdSetDepthBias(commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
    
    dump_vkCmdSetDepthBias(ApiDumpInstance::current(), commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions)
{
    device_dispatch_table(commandBuffer)->CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
    
    dump_vkCmdCopyImageToBuffer(ApiDumpInstance::current(), commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroySemaphore(device, semaphore, pAllocator);
    
    dump_vkDestroySemaphore(ApiDumpInstance::current(), device, semaphore, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
    device_dispatch_table(commandBuffer)->CmdDrawIndexedIndirect(commandBuffer, buffer, offset, drawCount, stride);
    
    dump_vkCmdDrawIndexedIndirect(ApiDumpInstance::current(), commandBuffer, buffer, offset, drawCount, stride);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyQueryPool(device, queryPool, pAllocator);
    
    dump_vkDestroyQueryPool(ApiDumpInstance::current(), device, queryPool, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetBlendConstants(VkCommandBuffer commandBuffer, const float blendConstants[4])
{
    device_dispatch_table(commandBuffer)->CmdSetBlendConstants(commandBuffer, blendConstants);
    
    dump_vkCmdSetBlendConstants(ApiDumpInstance::current(), commandBuffer, blendConstants);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
    device_dispatch_table(commandBuffer)->CmdSetEvent(commandBuffer, event, stageMask);
    
    dump_vkCmdSetEvent(ApiDumpInstance::current(), commandBuffer, event, stageMask);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers)
{
    device_dispatch_table(device)->FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
    ApiDumpInstance::current().eraseCmdBuffers(device, commandPool, std::vector<VkCommandBuffer>(pCommandBuffers, pCommandBuffers + commandBufferCount));
    dump_vkFreeCommandBuffers(ApiDumpInstance::current(), device, commandPool, commandBufferCount, pCommandBuffers);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBounds(VkCommandBuffer commandBuffer, float minDepthBounds, float maxDepthBounds)
{
    device_dispatch_table(commandBuffer)->CmdSetDepthBounds(commandBuffer, minDepthBounds, maxDepthBounds);
    
    dump_vkCmdSetDepthBounds(ApiDumpInstance::current(), commandBuffer, minDepthBounds, maxDepthBounds);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    device_dispatch_table(commandBuffer)->CmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
    
    dump_vkCmdDispatch(ApiDumpInstance::current(), commandBuffer, groupCountX, groupCountY, groupCountZ);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetViewportWScalingNV(VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewportWScalingNV* pViewportWScalings)
{
    device_dispatch_table(commandBuffer)->CmdSetViewportWScalingNV(commandBuffer, firstViewport, viewportCount, pViewportWScalings);
    
    dump_vkCmdSetViewportWScalingNV(ApiDumpInstance::current(), commandBuffer, firstViewport, viewportCount, pViewportWScalings);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyFramebuffer(device, framebuffer, pAllocator);
    
    dump_vkDestroyFramebuffer(ApiDumpInstance::current(), device, framebuffer, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset)
{
    device_dispatch_table(commandBuffer)->CmdDispatchIndirect(commandBuffer, buffer, offset);
    
    dump_vkCmdDispatchIndirect(ApiDumpInstance::current(), commandBuffer, buffer, offset);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroySampler(device, sampler, pAllocator);
    
    dump_vkDestroySampler(ApiDumpInstance::current(), device, sampler, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const void* pData)
{
    device_dispatch_table(commandBuffer)->CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData);
    
    dump_vkCmdUpdateBuffer(ApiDumpInstance::current(), commandBuffer, dstBuffer, dstOffset, dataSize, pData);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilCompareMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t compareMask)
{
    device_dispatch_table(commandBuffer)->CmdSetStencilCompareMask(commandBuffer, faceMask, compareMask);
    
    dump_vkCmdSetStencilCompareMask(ApiDumpInstance::current(), commandBuffer, faceMask, compareMask);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions)
{
    device_dispatch_table(commandBuffer)->CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
    
    dump_vkCmdCopyBuffer(ApiDumpInstance::current(), commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
    device_dispatch_table(commandBuffer)->CmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);
    
    dump_vkCmdFillBuffer(ApiDumpInstance::current(), commandBuffer, dstBuffer, dstOffset, size, data);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
    device_dispatch_table(commandBuffer)->CmdResetEvent(commandBuffer, event, stageMask);
    
    dump_vkCmdResetEvent(ApiDumpInstance::current(), commandBuffer, event, stageMask);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyEvent(VkDevice device, VkEvent event, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyEvent(device, event, pAllocator);
    
    dump_vkDestroyEvent(ApiDumpInstance::current(), device, event, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions)
{
    device_dispatch_table(commandBuffer)->CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
    
    dump_vkCmdCopyImage(ApiDumpInstance::current(), commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilWriteMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t writeMask)
{
    device_dispatch_table(commandBuffer)->CmdSetStencilWriteMask(commandBuffer, faceMask, writeMask);
    
    dump_vkCmdSetStencilWriteMask(ApiDumpInstance::current(), commandBuffer, faceMask, writeMask);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    device_dispatch_table(commandBuffer)->CmdWaitEvents(commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
    
    dump_vkCmdWaitEvents(ApiDumpInstance::current(), commandBuffer, eventCount, pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearColorValue* pColor, uint32_t rangeCount, const VkImageSubresourceRange* pRanges)
{
    device_dispatch_table(commandBuffer)->CmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
    
    dump_vkCmdClearColorImage(ApiDumpInstance::current(), commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);
    
    dump_vkDestroyDescriptorSetLayout(ApiDumpInstance::current(), device, descriptorSetLayout, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue)
{
    device_dispatch_table(device)->GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
    
    dump_vkGetDeviceQueue(ApiDumpInstance::current(), device, queueFamilyIndex, queueIndex, pQueue);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilReference(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask, uint32_t reference)
{
    device_dispatch_table(commandBuffer)->CmdSetStencilReference(commandBuffer, faceMask, reference);
    
    dump_vkCmdSetStencilReference(ApiDumpInstance::current(), commandBuffer, faceMask, reference);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyImageView(device, imageView, pAllocator);
    
    dump_vkDestroyImageView(ApiDumpInstance::current(), device, imageView, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets)
{
    device_dispatch_table(commandBuffer)->CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
    
    dump_vkCmdBindDescriptorSets(ApiDumpInstance::current(), commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, VkFilter filter)
{
    device_dispatch_table(commandBuffer)->CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
    
    dump_vkCmdBlitImage(ApiDumpInstance::current(), commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const VkImageSubresourceRange* pRanges)
{
    device_dispatch_table(commandBuffer)->CmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
    
    dump_vkCmdClearDepthStencilImage(ApiDumpInstance::current(), commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType)
{
    device_dispatch_table(commandBuffer)->CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);
    
    dump_vkCmdBindIndexBuffer(ApiDumpInstance::current(), commandBuffer, buffer, offset, indexType);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceGroupPeerMemoryFeaturesKHX(VkDevice device, uint32_t heapIndex, uint32_t localDeviceIndex, uint32_t remoteDeviceIndex, VkPeerMemoryFeatureFlagsKHX* pPeerMemoryFeatures)
{
    device_dispatch_table(device)->GetDeviceGroupPeerMemoryFeaturesKHX(device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);
    
    dump_vkGetDeviceGroupPeerMemoryFeaturesKHX(ApiDumpInstance::current(), device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets)
{
    device_dispatch_table(commandBuffer)->CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
    
    dump_vkCmdBindVertexBuffers(ApiDumpInstance::current(), commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites)
{
    device_dispatch_table(commandBuffer)->CmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
    
    dump_vkCmdPushDescriptorSetKHR(ApiDumpInstance::current(), commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyBuffer(device, buffer, pAllocator);
    
    dump_vkDestroyBuffer(ApiDumpInstance::current(), device, buffer, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t rectCount, const VkClearRect* pRects)
{
    device_dispatch_table(commandBuffer)->CmdClearAttachments(commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
    
    dump_vkCmdClearAttachments(ApiDumpInstance::current(), commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    device_dispatch_table(commandBuffer)->CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
    
    dump_vkCmdPipelineBarrier(ApiDumpInstance::current(), commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDeviceMaskKHX(VkCommandBuffer commandBuffer, uint32_t deviceMask)
{
    device_dispatch_table(commandBuffer)->CmdSetDeviceMaskKHX(commandBuffer, deviceMask);
    
    dump_vkCmdSetDeviceMaskKHX(ApiDumpInstance::current(), commandBuffer, deviceMask);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline)
{
    device_dispatch_table(commandBuffer)->CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
    
    dump_vkCmdBindPipeline(ApiDumpInstance::current(), commandBuffer, pipelineBindPoint, pipeline);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyShaderModule(device, shaderModule, pAllocator);
    
    dump_vkDestroyShaderModule(ApiDumpInstance::current(), device, shaderModule, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    device_dispatch_table(commandBuffer)->CmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    
    dump_vkCmdDraw(ApiDumpInstance::current(), commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewport* pViewports)
{
    device_dispatch_table(commandBuffer)->CmdSetViewport(commandBuffer, firstViewport, viewportCount, pViewports);
    
    dump_vkCmdSetViewport(ApiDumpInstance::current(), commandBuffer, firstViewport, viewportCount, pViewports);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions)
{
    device_dispatch_table(commandBuffer)->CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
    
    dump_vkCmdCopyBufferToImage(ApiDumpInstance::current(), commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    device_dispatch_table(commandBuffer)->CmdDrawIndexed(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    
    dump_vkCmdDrawIndexed(ApiDumpInstance::current(), commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetLineWidth(VkCommandBuffer commandBuffer, float lineWidth)
{
    device_dispatch_table(commandBuffer)->CmdSetLineWidth(commandBuffer, lineWidth);
    
    dump_vkCmdSetLineWidth(ApiDumpInstance::current(), commandBuffer, lineWidth);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors)
{
    device_dispatch_table(commandBuffer)->CmdSetScissor(commandBuffer, firstScissor, scissorCount, pScissors);
    
    dump_vkCmdSetScissor(ApiDumpInstance::current(), commandBuffer, firstScissor, scissorCount, pScissors);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageResolve* pRegions)
{
    device_dispatch_table(commandBuffer)->CmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
    
    dump_vkCmdResolveImage(ApiDumpInstance::current(), commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyPipeline(device, pipeline, pAllocator);
    
    dump_vkDestroyPipeline(ApiDumpInstance::current(), device, pipeline, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
    device_dispatch_table(commandBuffer)->CmdDrawIndirect(commandBuffer, buffer, offset, drawCount, stride);
    
    dump_vkCmdDrawIndirect(ApiDumpInstance::current(), commandBuffer, buffer, offset, drawCount, stride);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchBaseKHX(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    device_dispatch_table(commandBuffer)->CmdDispatchBaseKHX(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
    
    dump_vkCmdDispatchBaseKHX(ApiDumpInstance::current(), commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyPipelineCache(device, pipelineCache, pAllocator);
    
    dump_vkDestroyPipelineCache(ApiDumpInstance::current(), device, pipelineCache, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements)
{
    device_dispatch_table(device)->GetImageMemoryRequirements(device, image, pMemoryRequirements);
    
    dump_vkGetImageMemoryRequirements(ApiDumpInstance::current(), device, image, pMemoryRequirements);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyDescriptorPool(device, descriptorPool, pAllocator);
    
    dump_vkDestroyDescriptorPool(ApiDumpInstance::current(), device, descriptorPool, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags)
{
    device_dispatch_table(commandBuffer)->CmdBeginQuery(commandBuffer, queryPool, query, flags);
    
    dump_vkCmdBeginQuery(ApiDumpInstance::current(), commandBuffer, queryPool, query, flags);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDebugMarkerEndEXT(VkCommandBuffer commandBuffer)
{
    device_dispatch_table(commandBuffer)->CmdDebugMarkerEndEXT(commandBuffer);
    
    dump_vkCmdDebugMarkerEndEXT(ApiDumpInstance::current(), commandBuffer);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkSetHdrMetadataEXT(VkDevice device, uint32_t swapchainCount, const VkSwapchainKHR* pSwapchains, const VkHdrMetadataEXT* pMetadata)
{
    device_dispatch_table(device)->SetHdrMetadataEXT(device, swapchainCount, pSwapchains, pMetadata);
    
    dump_vkSetHdrMetadataEXT(ApiDumpInstance::current(), device, swapchainCount, pSwapchains, pMetadata);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDebugMarkerInsertEXT(VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT* pMarkerInfo)
{
    device_dispatch_table(commandBuffer)->CmdDebugMarkerInsertEXT(commandBuffer, pMarkerInfo);
    
    dump_vkCmdDebugMarkerInsertEXT(ApiDumpInstance::current(), commandBuffer, pMarkerInfo);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBufferView(VkDevice device, VkBufferView bufferView, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyBufferView(device, bufferView, pAllocator);
    
    dump_vkDestroyBufferView(ApiDumpInstance::current(), device, bufferView, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements(VkDevice device, VkImage image, uint32_t* pSparseMemoryRequirementCount, VkSparseImageMemoryRequirements* pSparseMemoryRequirements)
{
    device_dispatch_table(device)->GetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
    
    dump_vkGetImageSparseMemoryRequirements(ApiDumpInstance::current(), device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query)
{
    device_dispatch_table(commandBuffer)->CmdEndQuery(commandBuffer, queryPool, query);
    
    dump_vkCmdEndQuery(ApiDumpInstance::current(), commandBuffer, queryPool, query);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    device_dispatch_table(commandBuffer)->CmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount);
    
    dump_vkCmdResetQueryPool(ApiDumpInstance::current(), commandBuffer, queryPool, firstQuery, queryCount);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyImage(device, image, pAllocator);
    
    dump_vkDestroyImage(ApiDumpInstance::current(), device, image, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyRenderPass(device, renderPass, pAllocator);
    
    dump_vkDestroyRenderPass(ApiDumpInstance::current(), device, renderPass, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyPipelineLayout(device, pipelineLayout, pAllocator);
    
    dump_vkDestroyPipelineLayout(ApiDumpInstance::current(), device, pipelineLayout, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage, VkQueryPool queryPool, uint32_t query)
{
    device_dispatch_table(commandBuffer)->CmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, query);
    
    dump_vkCmdWriteTimestamp(ApiDumpInstance::current(), commandBuffer, pipelineStage, queryPool, query);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice device, VkDeviceMemory memory)
{
    device_dispatch_table(device)->UnmapMemory(device, memory);
    
    dump_vkUnmapMemory(ApiDumpInstance::current(), device, memory);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->FreeMemory(device, memory, pAllocator);
    
    dump_vkFreeMemory(ApiDumpInstance::current(), device, memory, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    device_dispatch_table(device)->UpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
    
    dump_vkUpdateDescriptorSets(ApiDumpInstance::current(), device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
    device_dispatch_table(commandBuffer)->CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);
    
    dump_vkCmdPushConstants(ApiDumpInstance::current(), commandBuffer, layout, stageFlags, offset, size, pValues);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags)
{
    device_dispatch_table(commandBuffer)->CmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
    
    dump_vkCmdCopyQueryPoolResults(ApiDumpInstance::current(), commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdProcessCommandsNVX(VkCommandBuffer commandBuffer, const VkCmdProcessCommandsInfoNVX* pProcessCommandsInfo)
{
    device_dispatch_table(commandBuffer)->CmdProcessCommandsNVX(commandBuffer, pProcessCommandsInfo);
    
    dump_vkCmdProcessCommandsNVX(ApiDumpInstance::current(), commandBuffer, pProcessCommandsInfo);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    device_dispatch_table(commandBuffer)->CmdDrawIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
    
    dump_vkCmdDrawIndirectCountAMD(ApiDumpInstance::current(), commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdReserveSpaceForCommandsNVX(VkCommandBuffer commandBuffer, const VkCmdReserveSpaceForCommandsInfoNVX* pReserveSpaceInfo)
{
    device_dispatch_table(commandBuffer)->CmdReserveSpaceForCommandsNVX(commandBuffer, pReserveSpaceInfo);
    
    dump_vkCmdReserveSpaceForCommandsNVX(ApiDumpInstance::current(), commandBuffer, pReserveSpaceInfo);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin, VkSubpassContents contents)
{
    device_dispatch_table(commandBuffer)->CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);
    
    dump_vkCmdBeginRenderPass(ApiDumpInstance::current(), commandBuffer, pRenderPassBegin, contents);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetRenderAreaGranularity(VkDevice device, VkRenderPass renderPass, VkExtent2D* pGranularity)
{
    device_dispatch_table(device)->GetRenderAreaGranularity(device, renderPass, pGranularity);
    
    dump_vkGetRenderAreaGranularity(ApiDumpInstance::current(), device, renderPass, pGranularity);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCountAMD(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    device_dispatch_table(commandBuffer)->CmdDrawIndexedIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
    
    dump_vkCmdDrawIndexedIndirectCountAMD(ApiDumpInstance::current(), commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorUpdateTemplateKHR(VkDevice device, VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyDescriptorUpdateTemplateKHR(device, descriptorUpdateTemplate, pAllocator);
    
    dump_vkDestroyDescriptorUpdateTemplateKHR(ApiDumpInstance::current(), device, descriptorUpdateTemplate, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyIndirectCommandsLayoutNVX(VkDevice device, VkIndirectCommandsLayoutNVX indirectCommandsLayout, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyIndirectCommandsLayoutNVX(device, indirectCommandsLayout, pAllocator);
    
    dump_vkDestroyIndirectCommandsLayoutNVX(ApiDumpInstance::current(), device, indirectCommandsLayout, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyFence(device, fence, pAllocator);
    
    dump_vkDestroyFence(ApiDumpInstance::current(), device, fence, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements2KHR(VkDevice device, const VkImageMemoryRequirementsInfo2KHR* pInfo, VkMemoryRequirements2KHR* pMemoryRequirements)
{
    device_dispatch_table(device)->GetImageMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);
    
    dump_vkGetImageMemoryRequirements2KHR(ApiDumpInstance::current(), device, pInfo, pMemoryRequirements);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyObjectTableNVX(VkDevice device, VkObjectTableNVX objectTable, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyObjectTableNVX(device, objectTable, pAllocator);
    
    dump_vkDestroyObjectTableNVX(ApiDumpInstance::current(), device, objectTable, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(VkDevice device, VkImage image, const VkImageSubresource* pSubresource, VkSubresourceLayout* pLayout)
{
    device_dispatch_table(device)->GetImageSubresourceLayout(device, image, pSubresource, pLayout);
    
    dump_vkGetImageSubresourceLayout(ApiDumpInstance::current(), device, image, pSubresource, pLayout);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSetWithTemplateKHR(VkDevice device, VkDescriptorSet descriptorSet, VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, const void* pData)
{
    device_dispatch_table(device)->UpdateDescriptorSetWithTemplateKHR(device, descriptorSet, descriptorUpdateTemplate, pData);
    
    dump_vkUpdateDescriptorSetWithTemplateKHR(ApiDumpInstance::current(), device, descriptorSet, descriptorUpdateTemplate, pData);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkTrimCommandPoolKHR(VkDevice device, VkCommandPool commandPool, VkCommandPoolTrimFlagsKHR flags)
{
    device_dispatch_table(device)->TrimCommandPoolKHR(device, commandPool, flags);
    
    dump_vkTrimCommandPoolKHR(ApiDumpInstance::current(), device, commandPool, flags);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements2KHR(VkDevice device, const VkBufferMemoryRequirementsInfo2KHR* pInfo, VkMemoryRequirements2KHR* pMemoryRequirements)
{
    device_dispatch_table(device)->GetBufferMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);
    
    dump_vkGetBufferMemoryRequirements2KHR(ApiDumpInstance::current(), device, pInfo, pMemoryRequirements);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroySwapchainKHR(device, swapchain, pAllocator);
    
    dump_vkDestroySwapchainKHR(ApiDumpInstance::current(), device, swapchain, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer commandBuffer, VkDescriptorUpdateTemplateKHR descriptorUpdateTemplate, VkPipelineLayout layout, uint32_t set, const void* pData)
{
    device_dispatch_table(commandBuffer)->CmdPushDescriptorSetWithTemplateKHR(commandBuffer, descriptorUpdateTemplate, layout, set, pData);
    
    dump_vkCmdPushDescriptorSetWithTemplateKHR(ApiDumpInstance::current(), commandBuffer, descriptorUpdateTemplate, layout, set, pData);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize* pCommittedMemoryInBytes)
{
    device_dispatch_table(device)->GetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes);
    
    dump_vkGetDeviceMemoryCommitment(ApiDumpInstance::current(), device, memory, pCommittedMemoryInBytes);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents)
{
    device_dispatch_table(commandBuffer)->CmdNextSubpass(commandBuffer, contents);
    
    dump_vkCmdNextSubpass(ApiDumpInstance::current(), commandBuffer, contents);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDiscardRectangleEXT(VkCommandBuffer commandBuffer, uint32_t firstDiscardRectangle, uint32_t discardRectangleCount, const VkRect2D* pDiscardRectangles)
{
    device_dispatch_table(commandBuffer)->CmdSetDiscardRectangleEXT(commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles);
    
    dump_vkCmdSetDiscardRectangleEXT(ApiDumpInstance::current(), commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements2KHR(VkDevice device, const VkImageSparseMemoryRequirementsInfo2KHR* pInfo, uint32_t* pSparseMemoryRequirementCount, VkSparseImageMemoryRequirements2KHR* pSparseMemoryRequirements)
{
    device_dispatch_table(device)->GetImageSparseMemoryRequirements2KHR(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
    
    dump_vkGetImageSparseMemoryRequirements2KHR(ApiDumpInstance::current(), device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator)
{
    device_dispatch_table(device)->DestroyCommandPool(device, commandPool, pAllocator);
    ApiDumpInstance::current().eraseCmdBufferPool(device, commandPool);
    dump_vkDestroyCommandPool(ApiDumpInstance::current(), device, commandPool, pAllocator);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer commandBuffer)
{
    device_dispatch_table(commandBuffer)->CmdEndRenderPass(commandBuffer);
    
    dump_vkCmdEndRenderPass(ApiDumpInstance::current(), commandBuffer);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers)
{
    device_dispatch_table(commandBuffer)->CmdExecuteCommands(commandBuffer, commandBufferCount, pCommandBuffers);
    
    dump_vkCmdExecuteCommands(ApiDumpInstance::current(), commandBuffer, commandBufferCount, pCommandBuffers);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements)
{
    device_dispatch_table(device)->GetBufferMemoryRequirements(device, buffer, pMemoryRequirements);
    
    dump_vkGetBufferMemoryRequirements(ApiDumpInstance::current(), device, buffer, pMemoryRequirements);
}
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDebugMarkerBeginEXT(VkCommandBuffer commandBuffer, const VkDebugMarkerMarkerInfoEXT* pMarkerInfo)
{
    device_dispatch_table(commandBuffer)->CmdDebugMarkerBeginEXT(commandBuffer, pMarkerInfo);
    
    dump_vkCmdDebugMarkerBeginEXT(ApiDumpInstance::current(), commandBuffer, pMarkerInfo);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName)
{

#if defined(VK_USE_PLATFORM_MIR_KHR)
    if(strcmp(pName, "vkCreateMirSurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateMirSurfaceKHR);
#endif // VK_USE_PLATFORM_MIR_KHR
    if(strcmp(pName, "vkEnumerateDeviceLayerProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateDeviceLayerProperties);
    if(strcmp(pName, "vkDestroySurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroySurfaceKHR);
    if(strcmp(pName, "vkGetPhysicalDeviceSurfaceCapabilities2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceCapabilities2KHR);
    if(strcmp(pName, "vkReleaseDisplayEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkReleaseDisplayEXT);
    if(strcmp(pName, "vkGetInstanceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr);
    if(strcmp(pName, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceExternalSemaphorePropertiesKHR);
#if defined(VK_USE_PLATFORM_MIR_KHR)
    if(strcmp(pName, "vkGetPhysicalDeviceMirPresentationSupportKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceMirPresentationSupportKHR);
#endif // VK_USE_PLATFORM_MIR_KHR
    if(strcmp(pName, "vkCreateInstance") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateInstance);
    if(strcmp(pName, "vkGetPhysicalDeviceSurfaceFormats2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceFormats2KHR);
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    if(strcmp(pName, "vkCreateAndroidSurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateAndroidSurfaceKHR);
#endif // VK_USE_PLATFORM_ANDROID_KHR
#if defined(VK_USE_PLATFORM_XLIB_XRANDR_EXT)
    if(strcmp(pName, "vkAcquireXlibDisplayEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkAcquireXlibDisplayEXT);
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
    if(strcmp(pName, "vkCreateDevice") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDevice);
#if defined(VK_USE_PLATFORM_IOS_MVK)
    if(strcmp(pName, "vkCreateIOSSurfaceMVK") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateIOSSurfaceMVK);
#endif // VK_USE_PLATFORM_IOS_MVK
#if defined(VK_USE_PLATFORM_XLIB_XRANDR_EXT)
    if(strcmp(pName, "vkGetRandROutputDisplayEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetRandROutputDisplayEXT);
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
    if(strcmp(pName, "vkGetPhysicalDeviceExternalBufferPropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceExternalBufferPropertiesKHR);
#if defined(VK_USE_PLATFORM_XCB_KHR)
    if(strcmp(pName, "vkCreateXcbSurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateXcbSurfaceKHR);
#endif // VK_USE_PLATFORM_XCB_KHR
    if(strcmp(pName, "vkGetPhysicalDeviceSurfaceSupportKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceSupportKHR);
    if(strcmp(pName, "vkCreateDebugReportCallbackEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDebugReportCallbackEXT);
    if(strcmp(pName, "vkGetPhysicalDeviceFeatures2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceFeatures2KHR);
    if(strcmp(pName, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    if(strcmp(pName, "vkGetPhysicalDeviceSurfaceCapabilities2EXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceCapabilities2EXT);
#if defined(VK_USE_PLATFORM_MACOS_MVK)
    if(strcmp(pName, "vkCreateMacOSSurfaceMVK") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateMacOSSurfaceMVK);
#endif // VK_USE_PLATFORM_MACOS_MVK
    if(strcmp(pName, "vkGetPhysicalDeviceProperties2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceProperties2KHR);
    if(strcmp(pName, "vkDestroyDebugReportCallbackEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyDebugReportCallbackEXT);
#if defined(VK_USE_PLATFORM_XCB_KHR)
    if(strcmp(pName, "vkGetPhysicalDeviceXcbPresentationSupportKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceXcbPresentationSupportKHR);
#endif // VK_USE_PLATFORM_XCB_KHR
    if(strcmp(pName, "vkGetPhysicalDeviceFormatProperties2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceFormatProperties2KHR);
    if(strcmp(pName, "vkGetPhysicalDeviceImageFormatProperties2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceImageFormatProperties2KHR);
    if(strcmp(pName, "vkDebugReportMessageEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDebugReportMessageEXT);
    if(strcmp(pName, "vkGetPhysicalDeviceQueueFamilyProperties2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceQueueFamilyProperties2KHR);
    if(strcmp(pName, "vkGetPhysicalDeviceMemoryProperties2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceMemoryProperties2KHR);
    if(strcmp(pName, "vkGetPhysicalDeviceSurfaceFormatsKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceFormatsKHR);
    if(strcmp(pName, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSparseImageFormatProperties2KHR);
    if(strcmp(pName, "vkGetPhysicalDevicePresentRectanglesKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDevicePresentRectanglesKHX);
    if(strcmp(pName, "vkGetPhysicalDeviceSurfacePresentModesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfacePresentModesKHR);
    if(strcmp(pName, "vkDestroyInstance") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyInstance);
    if(strcmp(pName, "vkEnumeratePhysicalDevices") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumeratePhysicalDevices);
#if defined(VK_USE_PLATFORM_XLIB_KHR)
    if(strcmp(pName, "vkCreateXlibSurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateXlibSurfaceKHR);
#endif // VK_USE_PLATFORM_XLIB_KHR
    if(strcmp(pName, "vkGetPhysicalDeviceFeatures") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceFeatures);
    if(strcmp(pName, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceDisplayPlanePropertiesKHR);
    if(strcmp(pName, "vkGetDisplayPlaneSupportedDisplaysKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDisplayPlaneSupportedDisplaysKHR);
#if defined(VK_USE_PLATFORM_XLIB_KHR)
    if(strcmp(pName, "vkGetPhysicalDeviceXlibPresentationSupportKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceXlibPresentationSupportKHR);
#endif // VK_USE_PLATFORM_XLIB_KHR
    if(strcmp(pName, "vkGetPhysicalDeviceFormatProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceFormatProperties);
    if(strcmp(pName, "vkGetDisplayModePropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDisplayModePropertiesKHR);
#if defined(VK_USE_PLATFORM_VI_NN)
    if(strcmp(pName, "vkCreateViSurfaceNN") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateViSurfaceNN);
#endif // VK_USE_PLATFORM_VI_NN
    if(strcmp(pName, "vkCreateDisplayModeKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDisplayModeKHR);
    if(strcmp(pName, "vkGetPhysicalDeviceSparseImageFormatProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSparseImageFormatProperties);
    if(strcmp(pName, "vkGetDisplayPlaneCapabilitiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDisplayPlaneCapabilitiesKHR);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkCreateWin32SurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateWin32SurfaceKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkCreateDisplayPlaneSurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDisplayPlaneSurfaceKHR);
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
    if(strcmp(pName, "vkCreateWaylandSurfaceKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateWaylandSurfaceKHR);
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkGetPhysicalDeviceWin32PresentationSupportKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceWin32PresentationSupportKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkGetPhysicalDeviceProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceProperties);
    if(strcmp(pName, "vkGetPhysicalDeviceQueueFamilyProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceQueueFamilyProperties);
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
    if(strcmp(pName, "vkGetPhysicalDeviceWaylandPresentationSupportKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceWaylandPresentationSupportKHR);
#endif // VK_USE_PLATFORM_WAYLAND_KHR
    if(strcmp(pName, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceExternalImageFormatPropertiesNV);
    if(strcmp(pName, "vkGetPhysicalDeviceExternalFencePropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceExternalFencePropertiesKHR);
    if(strcmp(pName, "vkGetPhysicalDeviceImageFormatProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceImageFormatProperties);
    if(strcmp(pName, "vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX);
    if(strcmp(pName, "vkGetPhysicalDeviceMemoryProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceMemoryProperties);
    if(strcmp(pName, "vkGetPhysicalDeviceDisplayPropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceDisplayPropertiesKHR);
    if(strcmp(pName, "vkEnumeratePhysicalDeviceGroupsKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumeratePhysicalDeviceGroupsKHX);

    if(instance_dispatch_table(instance)->GetInstanceProcAddr == NULL)
        return NULL;
    return instance_dispatch_table(instance)->GetInstanceProcAddr(instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName)
{

    if(strcmp(pName, "vkAllocateCommandBuffers") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkAllocateCommandBuffers);
    if(strcmp(pName, "vkCmdSetDepthBias") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetDepthBias);
    if(strcmp(pName, "vkCmdCopyImageToBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdCopyImageToBuffer);
    if(strcmp(pName, "vkDestroySemaphore") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroySemaphore);
    if(strcmp(pName, "vkCmdDrawIndexedIndirect") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDrawIndexedIndirect);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkGetSemaphoreWin32HandleKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetSemaphoreWin32HandleKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkDestroyQueryPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyQueryPool);
    if(strcmp(pName, "vkCmdSetBlendConstants") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetBlendConstants);
    if(strcmp(pName, "vkCmdSetEvent") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetEvent);
    if(strcmp(pName, "vkFreeCommandBuffers") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkFreeCommandBuffers);
    if(strcmp(pName, "vkCmdSetDepthBounds") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetDepthBounds);
    if(strcmp(pName, "vkGetQueryPoolResults") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetQueryPoolResults);
    if(strcmp(pName, "vkCmdDispatch") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDispatch);
    if(strcmp(pName, "vkCmdSetViewportWScalingNV") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetViewportWScalingNV);
    if(strcmp(pName, "vkEnumerateInstanceLayerProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateInstanceLayerProperties);
    if(strcmp(pName, "vkDestroyFramebuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyFramebuffer);
    if(strcmp(pName, "vkCmdDispatchIndirect") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDispatchIndirect);
    if(strcmp(pName, "vkCreateEvent") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateEvent);
    if(strcmp(pName, "vkDestroySampler") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroySampler);
    if(strcmp(pName, "vkCmdUpdateBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdUpdateBuffer);
    if(strcmp(pName, "vkCreateBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateBuffer);
    if(strcmp(pName, "vkBeginCommandBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkBeginCommandBuffer);
    if(strcmp(pName, "vkCmdSetStencilCompareMask") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetStencilCompareMask);
    if(strcmp(pName, "vkCreateDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDescriptorSetLayout);
    if(strcmp(pName, "vkCreateRenderPass") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateRenderPass);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkGetMemoryWin32HandleNV") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetMemoryWin32HandleNV);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkCmdCopyBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdCopyBuffer);
    if(strcmp(pName, "vkDisplayPowerControlEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDisplayPowerControlEXT);
    if(strcmp(pName, "vkCmdFillBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdFillBuffer);
    if(strcmp(pName, "vkCmdResetEvent") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdResetEvent);
    if(strcmp(pName, "vkDestroyEvent") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyEvent);
    if(strcmp(pName, "vkCmdCopyImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdCopyImage);
    if(strcmp(pName, "vkCmdSetStencilWriteMask") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetStencilWriteMask);
    if(strcmp(pName, "vkCmdWaitEvents") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdWaitEvents);
    if(strcmp(pName, "vkCmdClearColorImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdClearColorImage);
    if(strcmp(pName, "vkDestroyDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyDescriptorSetLayout);
    if(strcmp(pName, "vkRegisterDeviceEventEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkRegisterDeviceEventEXT);
    if(strcmp(pName, "vkSetEvent") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkSetEvent);
    if(strcmp(pName, "vkGetDeviceQueue") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceQueue);
    if(strcmp(pName, "vkGetEventStatus") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetEventStatus);
    if(strcmp(pName, "vkRegisterDisplayEventEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkRegisterDisplayEventEXT);
    if(strcmp(pName, "vkEndCommandBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEndCommandBuffer);
    if(strcmp(pName, "vkCmdSetStencilReference") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetStencilReference);
    if(strcmp(pName, "vkQueueSubmit") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkQueueSubmit);
    if(strcmp(pName, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    if(strcmp(pName, "vkGetSwapchainCounterEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetSwapchainCounterEXT);
    if(strcmp(pName, "vkDestroyImageView") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyImageView);
    if(strcmp(pName, "vkCmdBindDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBindDescriptorSets);
    if(strcmp(pName, "vkCmdBlitImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBlitImage);
    if(strcmp(pName, "vkCreateShaderModule") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateShaderModule);
    if(strcmp(pName, "vkCmdClearDepthStencilImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdClearDepthStencilImage);
    if(strcmp(pName, "vkImportSemaphoreFdKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkImportSemaphoreFdKHR);
    if(strcmp(pName, "vkResetEvent") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkResetEvent);
    if(strcmp(pName, "vkCreateComputePipelines") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateComputePipelines);
    if(strcmp(pName, "vkCmdBindIndexBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBindIndexBuffer);
    if(strcmp(pName, "vkCreateQueryPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateQueryPool);
    if(strcmp(pName, "vkGetDeviceGroupPeerMemoryFeaturesKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceGroupPeerMemoryFeaturesKHX);
    if(strcmp(pName, "vkGetSemaphoreFdKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetSemaphoreFdKHR);
    if(strcmp(pName, "vkDestroyDevice") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyDevice);
    if(strcmp(pName, "vkBindBufferMemory2KHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkBindBufferMemory2KHX);
    if(strcmp(pName, "vkCmdBindVertexBuffers") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBindVertexBuffers);
    if(strcmp(pName, "vkCmdPushDescriptorSetKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdPushDescriptorSetKHR);
    if(strcmp(pName, "vkDestroyBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyBuffer);
    if(strcmp(pName, "vkCmdClearAttachments") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdClearAttachments);
    if(strcmp(pName, "vkCmdPipelineBarrier") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdPipelineBarrier);
    if(strcmp(pName, "vkCmdSetDeviceMaskKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetDeviceMaskKHX);
    if(strcmp(pName, "vkResetCommandBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkResetCommandBuffer);
    if(strcmp(pName, "vkGetRefreshCycleDurationGOOGLE") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetRefreshCycleDurationGOOGLE);
    if(strcmp(pName, "vkCmdBindPipeline") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBindPipeline);
    if(strcmp(pName, "vkBindImageMemory2KHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkBindImageMemory2KHX);
    if(strcmp(pName, "vkDestroyShaderModule") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyShaderModule);
    if(strcmp(pName, "vkCreateDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDescriptorPool);
    if(strcmp(pName, "vkCmdDraw") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDraw);
    if(strcmp(pName, "vkCmdSetViewport") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetViewport);
    if(strcmp(pName, "vkCmdCopyBufferToImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdCopyBufferToImage);
    if(strcmp(pName, "vkCreatePipelineCache") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreatePipelineCache);
    if(strcmp(pName, "vkCmdDrawIndexed") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDrawIndexed);
    if(strcmp(pName, "vkCmdSetLineWidth") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetLineWidth);
    if(strcmp(pName, "vkGetDeviceGroupPresentCapabilitiesKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceGroupPresentCapabilitiesKHX);
    if(strcmp(pName, "vkGetDeviceGroupSurfacePresentModesKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceGroupSurfacePresentModesKHX);
    if(strcmp(pName, "vkCmdSetScissor") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetScissor);
    if(strcmp(pName, "vkGetPastPresentationTimingGOOGLE") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPastPresentationTimingGOOGLE);
    if(strcmp(pName, "vkCreateBufferView") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateBufferView);
    if(strcmp(pName, "vkCmdResolveImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdResolveImage);
    if(strcmp(pName, "vkAcquireNextImage2KHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkAcquireNextImage2KHX);
    if(strcmp(pName, "vkDestroyPipeline") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyPipeline);
    if(strcmp(pName, "vkCmdDrawIndirect") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDrawIndirect);
    if(strcmp(pName, "vkEnumerateInstanceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkEnumerateInstanceExtensionProperties);
    if(strcmp(pName, "vkCmdDispatchBaseKHX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDispatchBaseKHX);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkImportSemaphoreWin32HandleKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkImportSemaphoreWin32HandleKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkDestroyPipelineCache") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyPipelineCache);
    if(strcmp(pName, "vkQueueWaitIdle") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkQueueWaitIdle);
    if(strcmp(pName, "vkGetImageMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetImageMemoryRequirements);
    if(strcmp(pName, "vkDestroyDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyDescriptorPool);
    if(strcmp(pName, "vkCmdBeginQuery") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBeginQuery);
    if(strcmp(pName, "vkAcquireNextImageKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkAcquireNextImageKHR);
    if(strcmp(pName, "vkCmdDebugMarkerEndEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDebugMarkerEndEXT);
    if(strcmp(pName, "vkSetHdrMetadataEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkSetHdrMetadataEXT);
    if(strcmp(pName, "vkCmdDebugMarkerInsertEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDebugMarkerInsertEXT);
    if(strcmp(pName, "vkGetPipelineCacheData") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetPipelineCacheData);
    if(strcmp(pName, "vkAllocateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkAllocateDescriptorSets);
    if(strcmp(pName, "vkDestroyBufferView") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyBufferView);
    if(strcmp(pName, "vkResetDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkResetDescriptorPool);
    if(strcmp(pName, "vkDeviceWaitIdle") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDeviceWaitIdle);
    if(strcmp(pName, "vkCreateFence") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateFence);
    if(strcmp(pName, "vkGetSwapchainImagesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetSwapchainImagesKHR);
    if(strcmp(pName, "vkMergePipelineCaches") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkMergePipelineCaches);
    if(strcmp(pName, "vkAllocateMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkAllocateMemory);
    if(strcmp(pName, "vkGetImageSparseMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetImageSparseMemoryRequirements);
    if(strcmp(pName, "vkCreateImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateImage);
    if(strcmp(pName, "vkCreateGraphicsPipelines") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateGraphicsPipelines);
    if(strcmp(pName, "vkCreatePipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreatePipelineLayout);
    if(strcmp(pName, "vkCmdEndQuery") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdEndQuery);
    if(strcmp(pName, "vkCmdResetQueryPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdResetQueryPool);
    if(strcmp(pName, "vkQueuePresentKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkQueuePresentKHR);
    if(strcmp(pName, "vkDestroyImage") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyImage);
    if(strcmp(pName, "vkDestroyRenderPass") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyRenderPass);
    if(strcmp(pName, "vkDestroyPipelineLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyPipelineLayout);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkGetMemoryWin32HandleKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetMemoryWin32HandleKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkCmdWriteTimestamp") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdWriteTimestamp);
    if(strcmp(pName, "vkUnmapMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkUnmapMemory);
    if(strcmp(pName, "vkFreeDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkFreeDescriptorSets);
    if(strcmp(pName, "vkFreeMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkFreeMemory);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkGetMemoryWin32HandlePropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetMemoryWin32HandlePropertiesKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkMapMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkMapMemory);
    if(strcmp(pName, "vkUpdateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkUpdateDescriptorSets);
    if(strcmp(pName, "vkCmdPushConstants") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdPushConstants);
    if(strcmp(pName, "vkCmdCopyQueryPoolResults") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdCopyQueryPoolResults);
    if(strcmp(pName, "vkCreateDescriptorUpdateTemplateKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateDescriptorUpdateTemplateKHR);
    if(strcmp(pName, "vkGetSwapchainStatusKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetSwapchainStatusKHR);
    if(strcmp(pName, "vkCmdProcessCommandsNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdProcessCommandsNVX);
    if(strcmp(pName, "vkFlushMappedMemoryRanges") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkFlushMappedMemoryRanges);
    if(strcmp(pName, "vkCreateCommandPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateCommandPool);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkImportFenceWin32HandleKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkImportFenceWin32HandleKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkCreateSampler") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateSampler);
    if(strcmp(pName, "vkCmdDrawIndirectCountAMD") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDrawIndirectCountAMD);
    if(strcmp(pName, "vkCmdReserveSpaceForCommandsNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdReserveSpaceForCommandsNVX);
    if(strcmp(pName, "vkCmdBeginRenderPass") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdBeginRenderPass);
    if(strcmp(pName, "vkGetRenderAreaGranularity") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetRenderAreaGranularity);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if(strcmp(pName, "vkGetFenceWin32HandleKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetFenceWin32HandleKHR);
#endif // VK_USE_PLATFORM_WIN32_KHR
    if(strcmp(pName, "vkQueueBindSparse") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkQueueBindSparse);
    if(strcmp(pName, "vkCreateIndirectCommandsLayoutNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateIndirectCommandsLayoutNVX);
    if(strcmp(pName, "vkInvalidateMappedMemoryRanges") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkInvalidateMappedMemoryRanges);
    if(strcmp(pName, "vkCmdDrawIndexedIndirectCountAMD") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDrawIndexedIndirectCountAMD);
    if(strcmp(pName, "vkDestroyDescriptorUpdateTemplateKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyDescriptorUpdateTemplateKHR);
    if(strcmp(pName, "vkDestroyIndirectCommandsLayoutNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyIndirectCommandsLayoutNVX);
    if(strcmp(pName, "vkDestroyFence") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyFence);
    if(strcmp(pName, "vkGetImageMemoryRequirements2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetImageMemoryRequirements2KHR);
    if(strcmp(pName, "vkCreateSwapchainKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateSwapchainKHR);
    if(strcmp(pName, "vkGetMemoryFdKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetMemoryFdKHR);
    if(strcmp(pName, "vkDestroyObjectTableNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyObjectTableNVX);
    if(strcmp(pName, "vkGetImageSubresourceLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetImageSubresourceLayout);
    if(strcmp(pName, "vkUpdateDescriptorSetWithTemplateKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkUpdateDescriptorSetWithTemplateKHR);
    if(strcmp(pName, "vkTrimCommandPoolKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkTrimCommandPoolKHR);
    if(strcmp(pName, "vkCreateObjectTableNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateObjectTableNVX);
    if(strcmp(pName, "vkGetBufferMemoryRequirements2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetBufferMemoryRequirements2KHR);
    if(strcmp(pName, "vkDestroySwapchainKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroySwapchainKHR);
    if(strcmp(pName, "vkCmdPushDescriptorSetWithTemplateKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdPushDescriptorSetWithTemplateKHR);
    if(strcmp(pName, "vkCreateImageView") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateImageView);
    if(strcmp(pName, "vkGetDeviceMemoryCommitment") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceMemoryCommitment);
    if(strcmp(pName, "vkGetMemoryFdPropertiesKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetMemoryFdPropertiesKHR);
    if(strcmp(pName, "vkCreateFramebuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateFramebuffer);
    if(strcmp(pName, "vkResetFences") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkResetFences);
    if(strcmp(pName, "vkCmdNextSubpass") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdNextSubpass);
    if(strcmp(pName, "vkUnregisterObjectsNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkUnregisterObjectsNVX);
    if(strcmp(pName, "vkCmdSetDiscardRectangleEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdSetDiscardRectangleEXT);
    if(strcmp(pName, "vkGetImageSparseMemoryRequirements2KHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetImageSparseMemoryRequirements2KHR);
    if(strcmp(pName, "vkRegisterObjectsNVX") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkRegisterObjectsNVX);
    if(strcmp(pName, "vkBindBufferMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkBindBufferMemory);
    if(strcmp(pName, "vkGetFenceStatus") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetFenceStatus);
    if(strcmp(pName, "vkDebugMarkerSetObjectTagEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDebugMarkerSetObjectTagEXT);
    if(strcmp(pName, "vkDestroyCommandPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDestroyCommandPool);
    if(strcmp(pName, "vkCreateSharedSwapchainsKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateSharedSwapchainsKHR);
    if(strcmp(pName, "vkImportFenceFdKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkImportFenceFdKHR);
    if(strcmp(pName, "vkWaitForFences") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkWaitForFences);
    if(strcmp(pName, "vkBindImageMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkBindImageMemory);
    if(strcmp(pName, "vkCreateSemaphore") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCreateSemaphore);
    if(strcmp(pName, "vkResetCommandPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkResetCommandPool);
    if(strcmp(pName, "vkCmdEndRenderPass") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdEndRenderPass);
    if(strcmp(pName, "vkGetFenceFdKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetFenceFdKHR);
    if(strcmp(pName, "vkDebugMarkerSetObjectNameEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkDebugMarkerSetObjectNameEXT);
    if(strcmp(pName, "vkCmdExecuteCommands") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdExecuteCommands);
    if(strcmp(pName, "vkGetBufferMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkGetBufferMemoryRequirements);
    if(strcmp(pName, "vkCmdDebugMarkerBeginEXT") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vkCmdDebugMarkerBeginEXT);

    if(device_dispatch_table(device)->GetDeviceProcAddr == NULL)
        return NULL;
    return device_dispatch_table(device)->GetDeviceProcAddr(device, pName);
}

/*
** Copyright (c) 2015-2017 The Khronos Group Inc.
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

/*
** This header is generated from the Khronos Vulkan XML API Registry.
**
*/


namespace unique_objects {

// Unique Objects pNext extension handling function
void *CreateUnwrappedExtensionStructs(layer_data *dev_data, const void *pNext) {
    void *cur_pnext = const_cast<void *>(pNext);
    void *head_pnext = NULL;
    void *prev_ext_struct = NULL;
    void *cur_ext_struct = NULL;

    while (cur_pnext != NULL) {
        GenericHeader *header = reinterpret_cast<GenericHeader *>(cur_pnext);

        switch (header->sType) {
#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR: {
                    safe_VkD3D12FenceSubmitInfoKHR *safe_struct = new safe_VkD3D12FenceSubmitInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkD3D12FenceSubmitInfoKHR *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

            case VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO_KHX: {
                    safe_VkDeviceGroupSubmitInfoKHX *safe_struct = new safe_VkDeviceGroupSubmitInfoKHX;
                    safe_struct->initialize(reinterpret_cast<const VkDeviceGroupSubmitInfoKHX *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR: {
                    safe_VkWin32KeyedMutexAcquireReleaseInfoKHR *safe_struct = new safe_VkWin32KeyedMutexAcquireReleaseInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkWin32KeyedMutexAcquireReleaseInfoKHR *>(cur_pnext));
                    if (safe_struct->pAcquireSyncs) {
                        for (uint32_t index0 = 0; index0 < safe_struct->acquireCount; ++index0) {
                            safe_struct->pAcquireSyncs[index0] = Unwrap(dev_data, safe_struct->pAcquireSyncs[index0]);
                        }
                    }
                    if (safe_struct->pReleaseSyncs) {
                        for (uint32_t index0 = 0; index0 < safe_struct->releaseCount; ++index0) {
                            safe_struct->pReleaseSyncs[index0] = Unwrap(dev_data, safe_struct->pReleaseSyncs[index0]);
                        }
                    }
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_NV: {
                    safe_VkWin32KeyedMutexAcquireReleaseInfoNV *safe_struct = new safe_VkWin32KeyedMutexAcquireReleaseInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkWin32KeyedMutexAcquireReleaseInfoNV *>(cur_pnext));
                    if (safe_struct->pAcquireSyncs) {
                        for (uint32_t index0 = 0; index0 < safe_struct->acquireCount; ++index0) {
                            safe_struct->pAcquireSyncs[index0] = Unwrap(dev_data, safe_struct->pAcquireSyncs[index0]);
                        }
                    }
                    if (safe_struct->pReleaseSyncs) {
                        for (uint32_t index0 = 0; index0 < safe_struct->releaseCount; ++index0) {
                            safe_struct->pReleaseSyncs[index0] = Unwrap(dev_data, safe_struct->pReleaseSyncs[index0]);
                        }
                    }
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

            case VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV: {
                    safe_VkDedicatedAllocationMemoryAllocateInfoNV *safe_struct = new safe_VkDedicatedAllocationMemoryAllocateInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkDedicatedAllocationMemoryAllocateInfoNV *>(cur_pnext));
                    if (safe_struct->image) {
                        safe_struct->image = Unwrap(dev_data, safe_struct->image);
                    }
                    if (safe_struct->buffer) {
                        safe_struct->buffer = Unwrap(dev_data, safe_struct->buffer);
                    }
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR: {
                    safe_VkExportMemoryAllocateInfoKHR *safe_struct = new safe_VkExportMemoryAllocateInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkExportMemoryAllocateInfoKHR *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_NV: {
                    safe_VkExportMemoryAllocateInfoNV *safe_struct = new safe_VkExportMemoryAllocateInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkExportMemoryAllocateInfoNV *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR: {
                    safe_VkExportMemoryWin32HandleInfoKHR *safe_struct = new safe_VkExportMemoryWin32HandleInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkExportMemoryWin32HandleInfoKHR *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_NV: {
                    safe_VkExportMemoryWin32HandleInfoNV *safe_struct = new safe_VkExportMemoryWin32HandleInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkExportMemoryWin32HandleInfoNV *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

            case VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR: {
                    safe_VkImportMemoryFdInfoKHR *safe_struct = new safe_VkImportMemoryFdInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkImportMemoryFdInfoKHR *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR: {
                    safe_VkImportMemoryWin32HandleInfoKHR *safe_struct = new safe_VkImportMemoryWin32HandleInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkImportMemoryWin32HandleInfoKHR *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

#ifdef VK_USE_PLATFORM_WIN32_KHR 
            case VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_NV: {
                    safe_VkImportMemoryWin32HandleInfoNV *safe_struct = new safe_VkImportMemoryWin32HandleInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkImportMemoryWin32HandleInfoNV *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;
#endif // VK_USE_PLATFORM_WIN32_KHR 

            case VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHX: {
                    safe_VkMemoryAllocateFlagsInfoKHX *safe_struct = new safe_VkMemoryAllocateFlagsInfoKHX;
                    safe_struct->initialize(reinterpret_cast<const VkMemoryAllocateFlagsInfoKHX *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR: {
                    safe_VkMemoryDedicatedAllocateInfoKHR *safe_struct = new safe_VkMemoryDedicatedAllocateInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkMemoryDedicatedAllocateInfoKHR *>(cur_pnext));
                    if (safe_struct->image) {
                        safe_struct->image = Unwrap(dev_data, safe_struct->image);
                    }
                    if (safe_struct->buffer) {
                        safe_struct->buffer = Unwrap(dev_data, safe_struct->buffer);
                    }
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_IMAGE_CREATE_INFO_NV: {
                    safe_VkDedicatedAllocationImageCreateInfoNV *safe_struct = new safe_VkDedicatedAllocationImageCreateInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkDedicatedAllocationImageCreateInfoNV *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR: {
                    safe_VkExternalMemoryImageCreateInfoKHR *safe_struct = new safe_VkExternalMemoryImageCreateInfoKHR;
                    safe_struct->initialize(reinterpret_cast<const VkExternalMemoryImageCreateInfoKHR *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_NV: {
                    safe_VkExternalMemoryImageCreateInfoNV *safe_struct = new safe_VkExternalMemoryImageCreateInfoNV;
                    safe_struct->initialize(reinterpret_cast<const VkExternalMemoryImageCreateInfoNV *>(cur_pnext));
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHX: {
                    safe_VkImageSwapchainCreateInfoKHX *safe_struct = new safe_VkImageSwapchainCreateInfoKHX;
                    safe_struct->initialize(reinterpret_cast<const VkImageSwapchainCreateInfoKHX *>(cur_pnext));
                    if (safe_struct->swapchain) {
                        safe_struct->swapchain = Unwrap(dev_data, safe_struct->swapchain);
                    }
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            case VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHX: {
                    safe_VkBindImageMemorySwapchainInfoKHX *safe_struct = new safe_VkBindImageMemorySwapchainInfoKHX;
                    safe_struct->initialize(reinterpret_cast<const VkBindImageMemorySwapchainInfoKHX *>(cur_pnext));
                    if (safe_struct->swapchain) {
                        safe_struct->swapchain = Unwrap(dev_data, safe_struct->swapchain);
                    }
                    cur_ext_struct = reinterpret_cast<void *>(safe_struct);
                } break;

            default:
                break;
        }

        // Save pointer to the first structure in the pNext chain
        head_pnext = (head_pnext ? head_pnext : cur_ext_struct);

        // For any extension structure but the first, link the last struct's pNext to the current ext struct
        if (prev_ext_struct) {
            (reinterpret_cast<GenericHeader *>(prev_ext_struct))->pNext = cur_ext_struct;
        }
        prev_ext_struct = cur_ext_struct;

        // Process the next structure in the chain
        cur_pnext = const_cast<void *>(header->pNext);
    }
    return head_pnext;
}

// Free a pNext extension chain
void FreeUnwrappedExtensionStructs(void *head) {
    void * curr_ptr = head;
    while (curr_ptr) {
        GenericHeader *header = reinterpret_cast<GenericHeader *>(curr_ptr);
        void *temp = curr_ptr;
        curr_ptr = header->pNext;
        free(temp);
    }
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
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceLayerProperties(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties);

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(
    VkQueue                                     queue,
    uint32_t                                    submitCount,
    const VkSubmitInfo*                         pSubmits,
    VkFence                                     fence)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    safe_VkSubmitInfo *local_pSubmits = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pSubmits) {
            local_pSubmits = new safe_VkSubmitInfo[submitCount];
            for (uint32_t index0 = 0; index0 < submitCount; ++index0) {
                local_pSubmits[index0].initialize(&pSubmits[index0]);
                local_pSubmits[index0].pNext = CreateUnwrappedExtensionStructs(dev_data, local_pSubmits[index0].pNext);
                if (local_pSubmits[index0].pWaitSemaphores) {
                    for (uint32_t index1 = 0; index1 < local_pSubmits[index0].waitSemaphoreCount; ++index1) {
                        local_pSubmits[index0].pWaitSemaphores[index1] = Unwrap(dev_data, local_pSubmits[index0].pWaitSemaphores[index1]);
                    }
                }
                if (local_pSubmits[index0].pSignalSemaphores) {
                    for (uint32_t index1 = 0; index1 < local_pSubmits[index0].signalSemaphoreCount; ++index1) {
                        local_pSubmits[index0].pSignalSemaphores[index1] = Unwrap(dev_data, local_pSubmits[index0].pSignalSemaphores[index1]);
                    }
                }
            }
        }
        fence = Unwrap(dev_data, fence);
    }
    VkResult result = dev_data->dispatch_table.QueueSubmit(queue, submitCount, (const VkSubmitInfo*)local_pSubmits, fence);
    if (local_pSubmits) {
        for (uint32_t index0 = 0; index0 < submitCount; ++index0) {
            FreeUnwrappedExtensionStructs(const_cast<void *>(local_pSubmits[index0].pNext));
        }
        delete[] local_pSubmits;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(
    VkDevice                                    device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkMemoryAllocateInfo *local_pAllocateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pAllocateInfo) {
            local_pAllocateInfo = new safe_VkMemoryAllocateInfo(pAllocateInfo);
            local_pAllocateInfo->pNext = CreateUnwrappedExtensionStructs(dev_data, local_pAllocateInfo->pNext);
        }
    }
    VkResult result = dev_data->dispatch_table.AllocateMemory(device, (const VkMemoryAllocateInfo*)local_pAllocateInfo, pAllocator, pMemory);
    if (local_pAllocateInfo) {
        FreeUnwrappedExtensionStructs(const_cast<void *>(local_pAllocateInfo->pNext));
        delete local_pAllocateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pMemory = WrapNew(dev_data, *pMemory);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t memory_id = reinterpret_cast<uint64_t &>(memory);
    memory = (VkDeviceMemory)dev_data->unique_id_mapping[memory_id];
    dev_data->unique_id_mapping.erase(memory_id);
    lock.unlock();
    dev_data->dispatch_table.FreeMemory(device, memory, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize                                offset,
    VkDeviceSize                                size,
    VkMemoryMapFlags                            flags,
    void**                                      ppData)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        memory = Unwrap(dev_data, memory);
    }
    VkResult result = dev_data->dispatch_table.MapMemory(device, memory, offset, size, flags, ppData);

    return result;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        memory = Unwrap(dev_data, memory);
    }
    dev_data->dispatch_table.UnmapMemory(device, memory);

}

VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkMappedMemoryRange *local_pMemoryRanges = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pMemoryRanges) {
            local_pMemoryRanges = new safe_VkMappedMemoryRange[memoryRangeCount];
            for (uint32_t index0 = 0; index0 < memoryRangeCount; ++index0) {
                local_pMemoryRanges[index0].initialize(&pMemoryRanges[index0]);
                if (pMemoryRanges[index0].memory) {
                    local_pMemoryRanges[index0].memory = Unwrap(dev_data, pMemoryRanges[index0].memory);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.FlushMappedMemoryRanges(device, memoryRangeCount, (const VkMappedMemoryRange*)local_pMemoryRanges);
    if (local_pMemoryRanges) {
        delete[] local_pMemoryRanges;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkMappedMemoryRange *local_pMemoryRanges = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pMemoryRanges) {
            local_pMemoryRanges = new safe_VkMappedMemoryRange[memoryRangeCount];
            for (uint32_t index0 = 0; index0 < memoryRangeCount; ++index0) {
                local_pMemoryRanges[index0].initialize(&pMemoryRanges[index0]);
                if (pMemoryRanges[index0].memory) {
                    local_pMemoryRanges[index0].memory = Unwrap(dev_data, pMemoryRanges[index0].memory);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.InvalidateMappedMemoryRanges(device, memoryRangeCount, (const VkMappedMemoryRange*)local_pMemoryRanges);
    if (local_pMemoryRanges) {
        delete[] local_pMemoryRanges;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceMemoryCommitment(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize*                               pCommittedMemoryInBytes)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        memory = Unwrap(dev_data, memory);
    }
    dev_data->dispatch_table.GetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes);

}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkDeviceMemory                              memory,
    VkDeviceSize                                memoryOffset)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
        memory = Unwrap(dev_data, memory);
    }
    VkResult result = dev_data->dispatch_table.BindBufferMemory(device, buffer, memory, memoryOffset);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(
    VkDevice                                    device,
    VkImage                                     image,
    VkDeviceMemory                              memory,
    VkDeviceSize                                memoryOffset)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        image = Unwrap(dev_data, image);
        memory = Unwrap(dev_data, memory);
    }
    VkResult result = dev_data->dispatch_table.BindImageMemory(device, image, memory, memoryOffset);

    return result;
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkMemoryRequirements*                       pMemoryRequirements)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
    }
    dev_data->dispatch_table.GetBufferMemoryRequirements(device, buffer, pMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    VkMemoryRequirements*                       pMemoryRequirements)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        image = Unwrap(dev_data, image);
    }
    dev_data->dispatch_table.GetImageMemoryRequirements(device, image, pMemoryRequirements);

}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements*            pSparseMemoryRequirements)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        image = Unwrap(dev_data, image);
    }
    dev_data->dispatch_table.GetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);

}

VKAPI_ATTR VkResult VKAPI_CALL QueueBindSparse(
    VkQueue                                     queue,
    uint32_t                                    bindInfoCount,
    const VkBindSparseInfo*                     pBindInfo,
    VkFence                                     fence)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    safe_VkBindSparseInfo *local_pBindInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pBindInfo) {
            local_pBindInfo = new safe_VkBindSparseInfo[bindInfoCount];
            for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
                local_pBindInfo[index0].initialize(&pBindInfo[index0]);
                if (local_pBindInfo[index0].pWaitSemaphores) {
                    for (uint32_t index1 = 0; index1 < local_pBindInfo[index0].waitSemaphoreCount; ++index1) {
                        local_pBindInfo[index0].pWaitSemaphores[index1] = Unwrap(dev_data, local_pBindInfo[index0].pWaitSemaphores[index1]);
                    }
                }
                if (local_pBindInfo[index0].pBufferBinds) {
                    for (uint32_t index1 = 0; index1 < local_pBindInfo[index0].bufferBindCount; ++index1) {
                        if (pBindInfo[index0].pBufferBinds[index1].buffer) {
                            local_pBindInfo[index0].pBufferBinds[index1].buffer = Unwrap(dev_data, pBindInfo[index0].pBufferBinds[index1].buffer);
                        }
                        if (local_pBindInfo[index0].pBufferBinds[index1].pBinds) {
                            for (uint32_t index2 = 0; index2 < local_pBindInfo[index0].pBufferBinds[index1].bindCount; ++index2) {
                                if (pBindInfo[index0].pBufferBinds[index1].pBinds[index2].memory) {
                                    local_pBindInfo[index0].pBufferBinds[index1].pBinds[index2].memory = Unwrap(dev_data, pBindInfo[index0].pBufferBinds[index1].pBinds[index2].memory);
                                }
                            }
                        }
                    }
                }
                if (local_pBindInfo[index0].pImageOpaqueBinds) {
                    for (uint32_t index1 = 0; index1 < local_pBindInfo[index0].imageOpaqueBindCount; ++index1) {
                        if (pBindInfo[index0].pImageOpaqueBinds[index1].image) {
                            local_pBindInfo[index0].pImageOpaqueBinds[index1].image = Unwrap(dev_data, pBindInfo[index0].pImageOpaqueBinds[index1].image);
                        }
                        if (local_pBindInfo[index0].pImageOpaqueBinds[index1].pBinds) {
                            for (uint32_t index2 = 0; index2 < local_pBindInfo[index0].pImageOpaqueBinds[index1].bindCount; ++index2) {
                                if (pBindInfo[index0].pImageOpaqueBinds[index1].pBinds[index2].memory) {
                                    local_pBindInfo[index0].pImageOpaqueBinds[index1].pBinds[index2].memory = Unwrap(dev_data, pBindInfo[index0].pImageOpaqueBinds[index1].pBinds[index2].memory);
                                }
                            }
                        }
                    }
                }
                if (local_pBindInfo[index0].pImageBinds) {
                    for (uint32_t index1 = 0; index1 < local_pBindInfo[index0].imageBindCount; ++index1) {
                        if (pBindInfo[index0].pImageBinds[index1].image) {
                            local_pBindInfo[index0].pImageBinds[index1].image = Unwrap(dev_data, pBindInfo[index0].pImageBinds[index1].image);
                        }
                        if (local_pBindInfo[index0].pImageBinds[index1].pBinds) {
                            for (uint32_t index2 = 0; index2 < local_pBindInfo[index0].pImageBinds[index1].bindCount; ++index2) {
                                if (pBindInfo[index0].pImageBinds[index1].pBinds[index2].memory) {
                                    local_pBindInfo[index0].pImageBinds[index1].pBinds[index2].memory = Unwrap(dev_data, pBindInfo[index0].pImageBinds[index1].pBinds[index2].memory);
                                }
                            }
                        }
                    }
                }
                if (local_pBindInfo[index0].pSignalSemaphores) {
                    for (uint32_t index1 = 0; index1 < local_pBindInfo[index0].signalSemaphoreCount; ++index1) {
                        local_pBindInfo[index0].pSignalSemaphores[index1] = Unwrap(dev_data, local_pBindInfo[index0].pSignalSemaphores[index1]);
                    }
                }
            }
        }
        fence = Unwrap(dev_data, fence);
    }
    VkResult result = dev_data->dispatch_table.QueueBindSparse(queue, bindInfoCount, (const VkBindSparseInfo*)local_pBindInfo, fence);
    if (local_pBindInfo) {
        delete[] local_pBindInfo;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(
    VkDevice                                    device,
    const VkFenceCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateFence(device, pCreateInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pFence = WrapNew(dev_data, *pFence);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(
    VkDevice                                    device,
    VkFence                                     fence,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t fence_id = reinterpret_cast<uint64_t &>(fence);
    fence = (VkFence)dev_data->unique_id_mapping[fence_id];
    dev_data->unique_id_mapping.erase(fence_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyFence(device, fence, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkFence *local_pFences = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pFences) {
            local_pFences = new VkFence[fenceCount];
            for (uint32_t index0 = 0; index0 < fenceCount; ++index0) {
                local_pFences[index0] = Unwrap(dev_data, pFences[index0]);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.ResetFences(device, fenceCount, (const VkFence*)local_pFences);
    if (local_pFences)
        delete[] local_pFences;
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(
    VkDevice                                    device,
    VkFence                                     fence)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        fence = Unwrap(dev_data, fence);
    }
    VkResult result = dev_data->dispatch_table.GetFenceStatus(device, fence);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences,
    VkBool32                                    waitAll,
    uint64_t                                    timeout)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkFence *local_pFences = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pFences) {
            local_pFences = new VkFence[fenceCount];
            for (uint32_t index0 = 0; index0 < fenceCount; ++index0) {
                local_pFences[index0] = Unwrap(dev_data, pFences[index0]);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.WaitForFences(device, fenceCount, (const VkFence*)local_pFences, waitAll, timeout);
    if (local_pFences)
        delete[] local_pFences;
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSemaphore(
    VkDevice                                    device,
    const VkSemaphoreCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSemaphore*                                pSemaphore)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSemaphore = WrapNew(dev_data, *pSemaphore);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySemaphore(
    VkDevice                                    device,
    VkSemaphore                                 semaphore,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t semaphore_id = reinterpret_cast<uint64_t &>(semaphore);
    semaphore = (VkSemaphore)dev_data->unique_id_mapping[semaphore_id];
    dev_data->unique_id_mapping.erase(semaphore_id);
    lock.unlock();
    dev_data->dispatch_table.DestroySemaphore(device, semaphore, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateEvent(
    VkDevice                                    device,
    const VkEventCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkEvent*                                    pEvent)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateEvent(device, pCreateInfo, pAllocator, pEvent);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pEvent = WrapNew(dev_data, *pEvent);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyEvent(
    VkDevice                                    device,
    VkEvent                                     event,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t event_id = reinterpret_cast<uint64_t &>(event);
    event = (VkEvent)dev_data->unique_id_mapping[event_id];
    dev_data->unique_id_mapping.erase(event_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyEvent(device, event, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetEventStatus(
    VkDevice                                    device,
    VkEvent                                     event)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        event = Unwrap(dev_data, event);
    }
    VkResult result = dev_data->dispatch_table.GetEventStatus(device, event);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL SetEvent(
    VkDevice                                    device,
    VkEvent                                     event)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        event = Unwrap(dev_data, event);
    }
    VkResult result = dev_data->dispatch_table.SetEvent(device, event);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetEvent(
    VkDevice                                    device,
    VkEvent                                     event)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        event = Unwrap(dev_data, event);
    }
    VkResult result = dev_data->dispatch_table.ResetEvent(device, event);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(
    VkDevice                                    device,
    const VkQueryPoolCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkQueryPool*                                pQueryPool)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pQueryPool = WrapNew(dev_data, *pQueryPool);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t queryPool_id = reinterpret_cast<uint64_t &>(queryPool);
    queryPool = (VkQueryPool)dev_data->unique_id_mapping[queryPool_id];
    dev_data->unique_id_mapping.erase(queryPool_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyQueryPool(device, queryPool, pAllocator);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        queryPool = Unwrap(dev_data, queryPool);
    }
    VkResult result = dev_data->dispatch_table.GetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(
    VkDevice                                    device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pBuffer = WrapNew(dev_data, *pBuffer);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t buffer_id = reinterpret_cast<uint64_t &>(buffer);
    buffer = (VkBuffer)dev_data->unique_id_mapping[buffer_id];
    dev_data->unique_id_mapping.erase(buffer_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyBuffer(device, buffer, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateBufferView(
    VkDevice                                    device,
    const VkBufferViewCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBufferView*                               pView)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkBufferViewCreateInfo *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkBufferViewCreateInfo(pCreateInfo);
            if (pCreateInfo->buffer) {
                local_pCreateInfo->buffer = Unwrap(dev_data, pCreateInfo->buffer);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateBufferView(device, (const VkBufferViewCreateInfo*)local_pCreateInfo, pAllocator, pView);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pView = WrapNew(dev_data, *pView);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBufferView(
    VkDevice                                    device,
    VkBufferView                                bufferView,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t bufferView_id = reinterpret_cast<uint64_t &>(bufferView);
    bufferView = (VkBufferView)dev_data->unique_id_mapping[bufferView_id];
    dev_data->unique_id_mapping.erase(bufferView_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyBufferView(device, bufferView, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(
    VkDevice                                    device,
    const VkImageCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImage*                                    pImage)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImageCreateInfo *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkImageCreateInfo(pCreateInfo);
            local_pCreateInfo->pNext = CreateUnwrappedExtensionStructs(dev_data, local_pCreateInfo->pNext);
        }
    }
    VkResult result = dev_data->dispatch_table.CreateImage(device, (const VkImageCreateInfo*)local_pCreateInfo, pAllocator, pImage);
    if (local_pCreateInfo) {
        FreeUnwrappedExtensionStructs(const_cast<void *>(local_pCreateInfo->pNext));
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pImage = WrapNew(dev_data, *pImage);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(
    VkDevice                                    device,
    VkImage                                     image,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t image_id = reinterpret_cast<uint64_t &>(image);
    image = (VkImage)dev_data->unique_id_mapping[image_id];
    dev_data->unique_id_mapping.erase(image_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyImage(device, image, pAllocator);

}

VKAPI_ATTR void VKAPI_CALL GetImageSubresourceLayout(
    VkDevice                                    device,
    VkImage                                     image,
    const VkImageSubresource*                   pSubresource,
    VkSubresourceLayout*                        pLayout)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        image = Unwrap(dev_data, image);
    }
    dev_data->dispatch_table.GetImageSubresourceLayout(device, image, pSubresource, pLayout);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(
    VkDevice                                    device,
    const VkImageViewCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImageView*                                pView)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImageViewCreateInfo *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkImageViewCreateInfo(pCreateInfo);
            if (pCreateInfo->image) {
                local_pCreateInfo->image = Unwrap(dev_data, pCreateInfo->image);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateImageView(device, (const VkImageViewCreateInfo*)local_pCreateInfo, pAllocator, pView);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pView = WrapNew(dev_data, *pView);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(
    VkDevice                                    device,
    VkImageView                                 imageView,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t imageView_id = reinterpret_cast<uint64_t &>(imageView);
    imageView = (VkImageView)dev_data->unique_id_mapping[imageView_id];
    dev_data->unique_id_mapping.erase(imageView_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyImageView(device, imageView, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(
    VkDevice                                    device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pShaderModule = WrapNew(dev_data, *pShaderModule);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(
    VkDevice                                    device,
    VkShaderModule                              shaderModule,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t shaderModule_id = reinterpret_cast<uint64_t &>(shaderModule);
    shaderModule = (VkShaderModule)dev_data->unique_id_mapping[shaderModule_id];
    dev_data->unique_id_mapping.erase(shaderModule_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyShaderModule(device, shaderModule, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(
    VkDevice                                    device,
    const VkPipelineCacheCreateInfo*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineCache*                            pPipelineCache)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pPipelineCache = WrapNew(dev_data, *pPipelineCache);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t pipelineCache_id = reinterpret_cast<uint64_t &>(pipelineCache);
    pipelineCache = (VkPipelineCache)dev_data->unique_id_mapping[pipelineCache_id];
    dev_data->unique_id_mapping.erase(pipelineCache_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyPipelineCache(device, pipelineCache, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    size_t*                                     pDataSize,
    void*                                       pData)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        pipelineCache = Unwrap(dev_data, pipelineCache);
    }
    VkResult result = dev_data->dispatch_table.GetPipelineCacheData(device, pipelineCache, pDataSize, pData);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(
    VkDevice                                    device,
    VkPipelineCache                             dstCache,
    uint32_t                                    srcCacheCount,
    const VkPipelineCache*                      pSrcCaches)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkPipelineCache *local_pSrcCaches = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        dstCache = Unwrap(dev_data, dstCache);
        if (pSrcCaches) {
            local_pSrcCaches = new VkPipelineCache[srcCacheCount];
            for (uint32_t index0 = 0; index0 < srcCacheCount; ++index0) {
                local_pSrcCaches[index0] = Unwrap(dev_data, pSrcCaches[index0]);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.MergePipelineCaches(device, dstCache, srcCacheCount, (const VkPipelineCache*)local_pSrcCaches);
    if (local_pSrcCaches)
        delete[] local_pSrcCaches;
    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkGraphicsPipelineCreateInfo*         pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t pipeline_id = reinterpret_cast<uint64_t &>(pipeline);
    pipeline = (VkPipeline)dev_data->unique_id_mapping[pipeline_id];
    dev_data->unique_id_mapping.erase(pipeline_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyPipeline(device, pipeline, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(
    VkDevice                                    device,
    const VkPipelineLayoutCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineLayout*                           pPipelineLayout)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkPipelineLayoutCreateInfo *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkPipelineLayoutCreateInfo(pCreateInfo);
            if (local_pCreateInfo->pSetLayouts) {
                for (uint32_t index1 = 0; index1 < local_pCreateInfo->setLayoutCount; ++index1) {
                    local_pCreateInfo->pSetLayouts[index1] = Unwrap(dev_data, local_pCreateInfo->pSetLayouts[index1]);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreatePipelineLayout(device, (const VkPipelineLayoutCreateInfo*)local_pCreateInfo, pAllocator, pPipelineLayout);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pPipelineLayout = WrapNew(dev_data, *pPipelineLayout);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(
    VkDevice                                    device,
    VkPipelineLayout                            pipelineLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t pipelineLayout_id = reinterpret_cast<uint64_t &>(pipelineLayout);
    pipelineLayout = (VkPipelineLayout)dev_data->unique_id_mapping[pipelineLayout_id];
    dev_data->unique_id_mapping.erase(pipelineLayout_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyPipelineLayout(device, pipelineLayout, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(
    VkDevice                                    device,
    const VkSamplerCreateInfo*                  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSampler*                                  pSampler)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateSampler(device, pCreateInfo, pAllocator, pSampler);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSampler = WrapNew(dev_data, *pSampler);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(
    VkDevice                                    device,
    VkSampler                                   sampler,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t sampler_id = reinterpret_cast<uint64_t &>(sampler);
    sampler = (VkSampler)dev_data->unique_id_mapping[sampler_id];
    dev_data->unique_id_mapping.erase(sampler_id);
    lock.unlock();
    dev_data->dispatch_table.DestroySampler(device, sampler, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorSetLayout*                      pSetLayout)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkDescriptorSetLayoutCreateInfo *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkDescriptorSetLayoutCreateInfo(pCreateInfo);
            if (local_pCreateInfo->pBindings) {
                for (uint32_t index1 = 0; index1 < local_pCreateInfo->bindingCount; ++index1) {
                    if (local_pCreateInfo->pBindings[index1].pImmutableSamplers) {
                        for (uint32_t index2 = 0; index2 < local_pCreateInfo->pBindings[index1].descriptorCount; ++index2) {
                            local_pCreateInfo->pBindings[index1].pImmutableSamplers[index2] = Unwrap(dev_data, local_pCreateInfo->pBindings[index1].pImmutableSamplers[index2]);
                        }
                    }
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateDescriptorSetLayout(device, (const VkDescriptorSetLayoutCreateInfo*)local_pCreateInfo, pAllocator, pSetLayout);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSetLayout = WrapNew(dev_data, *pSetLayout);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(
    VkDevice                                    device,
    VkDescriptorSetLayout                       descriptorSetLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t descriptorSetLayout_id = reinterpret_cast<uint64_t &>(descriptorSetLayout);
    descriptorSetLayout = (VkDescriptorSetLayout)dev_data->unique_id_mapping[descriptorSetLayout_id];
    dev_data->unique_id_mapping.erase(descriptorSetLayout_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(
    VkDevice                                    device,
    const VkDescriptorPoolCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorPool*                           pDescriptorPool)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pDescriptorPool = WrapNew(dev_data, *pDescriptorPool);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t descriptorPool_id = reinterpret_cast<uint64_t &>(descriptorPool);
    descriptorPool = (VkDescriptorPool)dev_data->unique_id_mapping[descriptorPool_id];
    dev_data->unique_id_mapping.erase(descriptorPool_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyDescriptorPool(device, descriptorPool, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    VkDescriptorPoolResetFlags                  flags)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        descriptorPool = Unwrap(dev_data, descriptorPool);
    }
    VkResult result = dev_data->dispatch_table.ResetDescriptorPool(device, descriptorPool, flags);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(
    VkDevice                                    device,
    const VkDescriptorSetAllocateInfo*          pAllocateInfo,
    VkDescriptorSet*                            pDescriptorSets)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkDescriptorSetAllocateInfo *local_pAllocateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pAllocateInfo) {
            local_pAllocateInfo = new safe_VkDescriptorSetAllocateInfo(pAllocateInfo);
            if (pAllocateInfo->descriptorPool) {
                local_pAllocateInfo->descriptorPool = Unwrap(dev_data, pAllocateInfo->descriptorPool);
            }
            if (local_pAllocateInfo->pSetLayouts) {
                for (uint32_t index1 = 0; index1 < local_pAllocateInfo->descriptorSetCount; ++index1) {
                    local_pAllocateInfo->pSetLayouts[index1] = Unwrap(dev_data, local_pAllocateInfo->pSetLayouts[index1]);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.AllocateDescriptorSets(device, (const VkDescriptorSetAllocateInfo*)local_pAllocateInfo, pDescriptorSets);
    if (local_pAllocateInfo) {
        delete local_pAllocateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        for (uint32_t index0 = 0; index0 < pAllocateInfo->descriptorSetCount; index0++) {
            pDescriptorSets[index0] = WrapNew(dev_data, pDescriptorSets[index0]);
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkDescriptorSet *local_pDescriptorSets = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        descriptorPool = Unwrap(dev_data, descriptorPool);
        if (pDescriptorSets) {
            local_pDescriptorSets = new VkDescriptorSet[descriptorSetCount];
            for (uint32_t index0 = 0; index0 < descriptorSetCount; ++index0) {
                local_pDescriptorSets[index0] = Unwrap(dev_data, pDescriptorSets[index0]);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.FreeDescriptorSets(device, descriptorPool, descriptorSetCount, (const VkDescriptorSet*)local_pDescriptorSets);
    if (local_pDescriptorSets)
        delete[] local_pDescriptorSets;
    if ((VK_SUCCESS == result) && (pDescriptorSets)) {
        std::unique_lock<std::mutex> lock(global_lock);
        for (uint32_t index0 = 0; index0 < descriptorSetCount; index0++) {
            VkDescriptorSet handle = pDescriptorSets[index0];
            uint64_t unique_id = reinterpret_cast<uint64_t &>(handle);
            dev_data->unique_id_mapping.erase(unique_id);
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(
    VkDevice                                    device,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites,
    uint32_t                                    descriptorCopyCount,
    const VkCopyDescriptorSet*                  pDescriptorCopies)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkWriteDescriptorSet *local_pDescriptorWrites = NULL;
    safe_VkCopyDescriptorSet *local_pDescriptorCopies = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pDescriptorWrites) {
            local_pDescriptorWrites = new safe_VkWriteDescriptorSet[descriptorWriteCount];
            for (uint32_t index0 = 0; index0 < descriptorWriteCount; ++index0) {
                local_pDescriptorWrites[index0].initialize(&pDescriptorWrites[index0]);
                if (pDescriptorWrites[index0].dstSet) {
                    local_pDescriptorWrites[index0].dstSet = Unwrap(dev_data, pDescriptorWrites[index0].dstSet);
                }
                if (local_pDescriptorWrites[index0].pImageInfo) {
                    for (uint32_t index1 = 0; index1 < local_pDescriptorWrites[index0].descriptorCount; ++index1) {
                        if (pDescriptorWrites[index0].pImageInfo[index1].sampler) {
                            local_pDescriptorWrites[index0].pImageInfo[index1].sampler = Unwrap(dev_data, pDescriptorWrites[index0].pImageInfo[index1].sampler);
                        }
                        if (pDescriptorWrites[index0].pImageInfo[index1].imageView) {
                            local_pDescriptorWrites[index0].pImageInfo[index1].imageView = Unwrap(dev_data, pDescriptorWrites[index0].pImageInfo[index1].imageView);
                        }
                    }
                }
                if (local_pDescriptorWrites[index0].pBufferInfo) {
                    for (uint32_t index1 = 0; index1 < local_pDescriptorWrites[index0].descriptorCount; ++index1) {
                        if (pDescriptorWrites[index0].pBufferInfo[index1].buffer) {
                            local_pDescriptorWrites[index0].pBufferInfo[index1].buffer = Unwrap(dev_data, pDescriptorWrites[index0].pBufferInfo[index1].buffer);
                        }
                    }
                }
                if (local_pDescriptorWrites[index0].pTexelBufferView) {
                    for (uint32_t index1 = 0; index1 < local_pDescriptorWrites[index0].descriptorCount; ++index1) {
                        local_pDescriptorWrites[index0].pTexelBufferView[index1] = Unwrap(dev_data, local_pDescriptorWrites[index0].pTexelBufferView[index1]);
                    }
                }
            }
        }
        if (pDescriptorCopies) {
            local_pDescriptorCopies = new safe_VkCopyDescriptorSet[descriptorCopyCount];
            for (uint32_t index0 = 0; index0 < descriptorCopyCount; ++index0) {
                local_pDescriptorCopies[index0].initialize(&pDescriptorCopies[index0]);
                if (pDescriptorCopies[index0].srcSet) {
                    local_pDescriptorCopies[index0].srcSet = Unwrap(dev_data, pDescriptorCopies[index0].srcSet);
                }
                if (pDescriptorCopies[index0].dstSet) {
                    local_pDescriptorCopies[index0].dstSet = Unwrap(dev_data, pDescriptorCopies[index0].dstSet);
                }
            }
        }
    }
    dev_data->dispatch_table.UpdateDescriptorSets(device, descriptorWriteCount, (const VkWriteDescriptorSet*)local_pDescriptorWrites, descriptorCopyCount, (const VkCopyDescriptorSet*)local_pDescriptorCopies);
    if (local_pDescriptorWrites) {
        delete[] local_pDescriptorWrites;
    }
    if (local_pDescriptorCopies) {
        delete[] local_pDescriptorCopies;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFramebuffer(
    VkDevice                                    device,
    const VkFramebufferCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFramebuffer*                              pFramebuffer)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkFramebufferCreateInfo *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkFramebufferCreateInfo(pCreateInfo);
            if (pCreateInfo->renderPass) {
                local_pCreateInfo->renderPass = Unwrap(dev_data, pCreateInfo->renderPass);
            }
            if (local_pCreateInfo->pAttachments) {
                for (uint32_t index1 = 0; index1 < local_pCreateInfo->attachmentCount; ++index1) {
                    local_pCreateInfo->pAttachments[index1] = Unwrap(dev_data, local_pCreateInfo->pAttachments[index1]);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateFramebuffer(device, (const VkFramebufferCreateInfo*)local_pCreateInfo, pAllocator, pFramebuffer);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pFramebuffer = WrapNew(dev_data, *pFramebuffer);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFramebuffer(
    VkDevice                                    device,
    VkFramebuffer                               framebuffer,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t framebuffer_id = reinterpret_cast<uint64_t &>(framebuffer);
    framebuffer = (VkFramebuffer)dev_data->unique_id_mapping[framebuffer_id];
    dev_data->unique_id_mapping.erase(framebuffer_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyFramebuffer(device, framebuffer, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass(
    VkDevice                                    device,
    const VkRenderPassCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pRenderPass = WrapNew(dev_data, *pRenderPass);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyRenderPass(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t renderPass_id = reinterpret_cast<uint64_t &>(renderPass);
    renderPass = (VkRenderPass)dev_data->unique_id_mapping[renderPass_id];
    dev_data->unique_id_mapping.erase(renderPass_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyRenderPass(device, renderPass, pAllocator);

}

VKAPI_ATTR void VKAPI_CALL GetRenderAreaGranularity(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    VkExtent2D*                                 pGranularity)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        renderPass = Unwrap(dev_data, renderPass);
    }
    dev_data->dispatch_table.GetRenderAreaGranularity(device, renderPass, pGranularity);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(
    VkDevice                                    device,
    const VkCommandPoolCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkCommandPool*                              pCommandPool)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pCommandPool = WrapNew(dev_data, *pCommandPool);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t commandPool_id = reinterpret_cast<uint64_t &>(commandPool);
    commandPool = (VkCommandPool)dev_data->unique_id_mapping[commandPool_id];
    dev_data->unique_id_mapping.erase(commandPool_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyCommandPool(device, commandPool, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolResetFlags                     flags)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        commandPool = Unwrap(dev_data, commandPool);
    }
    VkResult result = dev_data->dispatch_table.ResetCommandPool(device, commandPool, flags);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(
    VkDevice                                    device,
    const VkCommandBufferAllocateInfo*          pAllocateInfo,
    VkCommandBuffer*                            pCommandBuffers)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkCommandBufferAllocateInfo *local_pAllocateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pAllocateInfo) {
            local_pAllocateInfo = new safe_VkCommandBufferAllocateInfo(pAllocateInfo);
            if (pAllocateInfo->commandPool) {
                local_pAllocateInfo->commandPool = Unwrap(dev_data, pAllocateInfo->commandPool);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.AllocateCommandBuffers(device, (const VkCommandBufferAllocateInfo*)local_pAllocateInfo, pCommandBuffers);
    if (local_pAllocateInfo) {
        delete local_pAllocateInfo;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        commandPool = Unwrap(dev_data, commandPool);
    }
    dev_data->dispatch_table.FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);

}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    const VkCommandBufferBeginInfo*             pBeginInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    safe_VkCommandBufferBeginInfo *local_pBeginInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pBeginInfo) {
            local_pBeginInfo = new safe_VkCommandBufferBeginInfo(pBeginInfo);
            if (local_pBeginInfo->pInheritanceInfo) {
                if (pBeginInfo->pInheritanceInfo->renderPass) {
                    local_pBeginInfo->pInheritanceInfo->renderPass = Unwrap(dev_data, pBeginInfo->pInheritanceInfo->renderPass);
                }
                if (pBeginInfo->pInheritanceInfo->framebuffer) {
                    local_pBeginInfo->pInheritanceInfo->framebuffer = Unwrap(dev_data, pBeginInfo->pInheritanceInfo->framebuffer);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.BeginCommandBuffer(commandBuffer, (const VkCommandBufferBeginInfo*)local_pBeginInfo);
    if (local_pBeginInfo) {
        delete local_pBeginInfo;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipeline                                  pipeline)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        pipeline = Unwrap(dev_data, pipeline);
    }
    dev_data->dispatch_table.CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    VkDescriptorSet *local_pDescriptorSets = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        layout = Unwrap(dev_data, layout);
        if (pDescriptorSets) {
            local_pDescriptorSets = new VkDescriptorSet[descriptorSetCount];
            for (uint32_t index0 = 0; index0 < descriptorSetCount; ++index0) {
                local_pDescriptorSets[index0] = Unwrap(dev_data, pDescriptorSets[index0]);
            }
        }
    }
    dev_data->dispatch_table.CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, (const VkDescriptorSet*)local_pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
    if (local_pDescriptorSets)
        delete[] local_pDescriptorSets;
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkIndexType                                 indexType)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
    }
    dev_data->dispatch_table.CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);

}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    VkBuffer *local_pBuffers = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pBuffers) {
            local_pBuffers = new VkBuffer[bindingCount];
            for (uint32_t index0 = 0; index0 < bindingCount; ++index0) {
                local_pBuffers[index0] = Unwrap(dev_data, pBuffers[index0]);
            }
        }
    }
    dev_data->dispatch_table.CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, (const VkBuffer*)local_pBuffers, pOffsets);
    if (local_pBuffers)
        delete[] local_pBuffers;
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
    }
    dev_data->dispatch_table.CmdDrawIndirect(commandBuffer, buffer, offset, drawCount, stride);

}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
    }
    dev_data->dispatch_table.CmdDrawIndexedIndirect(commandBuffer, buffer, offset, drawCount, stride);

}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
    }
    dev_data->dispatch_table.CmdDispatchIndirect(commandBuffer, buffer, offset);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    srcBuffer,
    VkBuffer                                    dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferCopy*                         pRegions)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        srcBuffer = Unwrap(dev_data, srcBuffer);
        dstBuffer = Unwrap(dev_data, dstBuffer);
    }
    dev_data->dispatch_table.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        srcImage = Unwrap(dev_data, srcImage);
        dstImage = Unwrap(dev_data, dstImage);
    }
    dev_data->dispatch_table.CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        srcImage = Unwrap(dev_data, srcImage);
        dstImage = Unwrap(dev_data, dstImage);
    }
    dev_data->dispatch_table.CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    srcBuffer,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        srcBuffer = Unwrap(dev_data, srcBuffer);
        dstImage = Unwrap(dev_data, dstImage);
    }
    dev_data->dispatch_table.CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkBuffer                                    dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        srcImage = Unwrap(dev_data, srcImage);
        dstBuffer = Unwrap(dev_data, dstBuffer);
    }
    dev_data->dispatch_table.CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                dataSize,
    const void*                                 pData)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        dstBuffer = Unwrap(dev_data, dstBuffer);
    }
    dev_data->dispatch_table.CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData);

}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                size,
    uint32_t                                    data)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        dstBuffer = Unwrap(dev_data, dstBuffer);
    }
    dev_data->dispatch_table.CmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);

}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearColorValue*                    pColor,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        image = Unwrap(dev_data, image);
    }
    dev_data->dispatch_table.CmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);

}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearDepthStencilValue*             pDepthStencil,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        image = Unwrap(dev_data, image);
    }
    dev_data->dispatch_table.CmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        srcImage = Unwrap(dev_data, srcImage);
        dstImage = Unwrap(dev_data, dstImage);
    }
    dev_data->dispatch_table.CmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);

}

VKAPI_ATTR void VKAPI_CALL CmdSetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        event = Unwrap(dev_data, event);
    }
    dev_data->dispatch_table.CmdSetEvent(commandBuffer, event, stageMask);

}

VKAPI_ATTR void VKAPI_CALL CmdResetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        event = Unwrap(dev_data, event);
    }
    dev_data->dispatch_table.CmdResetEvent(commandBuffer, event, stageMask);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    VkEvent *local_pEvents = NULL;
    safe_VkBufferMemoryBarrier *local_pBufferMemoryBarriers = NULL;
    safe_VkImageMemoryBarrier *local_pImageMemoryBarriers = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pEvents) {
            local_pEvents = new VkEvent[eventCount];
            for (uint32_t index0 = 0; index0 < eventCount; ++index0) {
                local_pEvents[index0] = Unwrap(dev_data, pEvents[index0]);
            }
        }
        if (pBufferMemoryBarriers) {
            local_pBufferMemoryBarriers = new safe_VkBufferMemoryBarrier[bufferMemoryBarrierCount];
            for (uint32_t index0 = 0; index0 < bufferMemoryBarrierCount; ++index0) {
                local_pBufferMemoryBarriers[index0].initialize(&pBufferMemoryBarriers[index0]);
                if (pBufferMemoryBarriers[index0].buffer) {
                    local_pBufferMemoryBarriers[index0].buffer = Unwrap(dev_data, pBufferMemoryBarriers[index0].buffer);
                }
            }
        }
        if (pImageMemoryBarriers) {
            local_pImageMemoryBarriers = new safe_VkImageMemoryBarrier[imageMemoryBarrierCount];
            for (uint32_t index0 = 0; index0 < imageMemoryBarrierCount; ++index0) {
                local_pImageMemoryBarriers[index0].initialize(&pImageMemoryBarriers[index0]);
                if (pImageMemoryBarriers[index0].image) {
                    local_pImageMemoryBarriers[index0].image = Unwrap(dev_data, pImageMemoryBarriers[index0].image);
                }
            }
        }
    }
    dev_data->dispatch_table.CmdWaitEvents(commandBuffer, eventCount, (const VkEvent*)local_pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, (const VkBufferMemoryBarrier*)local_pBufferMemoryBarriers, imageMemoryBarrierCount, (const VkImageMemoryBarrier*)local_pImageMemoryBarriers);
    if (local_pEvents)
        delete[] local_pEvents;
    if (local_pBufferMemoryBarriers) {
        delete[] local_pBufferMemoryBarriers;
    }
    if (local_pImageMemoryBarriers) {
        delete[] local_pImageMemoryBarriers;
    }
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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    safe_VkBufferMemoryBarrier *local_pBufferMemoryBarriers = NULL;
    safe_VkImageMemoryBarrier *local_pImageMemoryBarriers = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pBufferMemoryBarriers) {
            local_pBufferMemoryBarriers = new safe_VkBufferMemoryBarrier[bufferMemoryBarrierCount];
            for (uint32_t index0 = 0; index0 < bufferMemoryBarrierCount; ++index0) {
                local_pBufferMemoryBarriers[index0].initialize(&pBufferMemoryBarriers[index0]);
                if (pBufferMemoryBarriers[index0].buffer) {
                    local_pBufferMemoryBarriers[index0].buffer = Unwrap(dev_data, pBufferMemoryBarriers[index0].buffer);
                }
            }
        }
        if (pImageMemoryBarriers) {
            local_pImageMemoryBarriers = new safe_VkImageMemoryBarrier[imageMemoryBarrierCount];
            for (uint32_t index0 = 0; index0 < imageMemoryBarrierCount; ++index0) {
                local_pImageMemoryBarriers[index0].initialize(&pImageMemoryBarriers[index0]);
                if (pImageMemoryBarriers[index0].image) {
                    local_pImageMemoryBarriers[index0].image = Unwrap(dev_data, pImageMemoryBarriers[index0].image);
                }
            }
        }
    }
    dev_data->dispatch_table.CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, (const VkBufferMemoryBarrier*)local_pBufferMemoryBarriers, imageMemoryBarrierCount, (const VkImageMemoryBarrier*)local_pImageMemoryBarriers);
    if (local_pBufferMemoryBarriers) {
        delete[] local_pBufferMemoryBarriers;
    }
    if (local_pImageMemoryBarriers) {
        delete[] local_pImageMemoryBarriers;
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        queryPool = Unwrap(dev_data, queryPool);
    }
    dev_data->dispatch_table.CmdBeginQuery(commandBuffer, queryPool, query, flags);

}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        queryPool = Unwrap(dev_data, queryPool);
    }
    dev_data->dispatch_table.CmdEndQuery(commandBuffer, queryPool, query);

}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        queryPool = Unwrap(dev_data, queryPool);
    }
    dev_data->dispatch_table.CmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount);

}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        queryPool = Unwrap(dev_data, queryPool);
    }
    dev_data->dispatch_table.CmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, query);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        queryPool = Unwrap(dev_data, queryPool);
        dstBuffer = Unwrap(dev_data, dstBuffer);
    }
    dev_data->dispatch_table.CmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);

}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(
    VkCommandBuffer                             commandBuffer,
    VkPipelineLayout                            layout,
    VkShaderStageFlags                          stageFlags,
    uint32_t                                    offset,
    uint32_t                                    size,
    const void*                                 pValues)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        layout = Unwrap(dev_data, layout);
    }
    dev_data->dispatch_table.CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);

}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(
    VkCommandBuffer                             commandBuffer,
    const VkRenderPassBeginInfo*                pRenderPassBegin,
    VkSubpassContents                           contents)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    safe_VkRenderPassBeginInfo *local_pRenderPassBegin = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pRenderPassBegin) {
            local_pRenderPassBegin = new safe_VkRenderPassBeginInfo(pRenderPassBegin);
            if (pRenderPassBegin->renderPass) {
                local_pRenderPassBegin->renderPass = Unwrap(dev_data, pRenderPassBegin->renderPass);
            }
            if (pRenderPassBegin->framebuffer) {
                local_pRenderPassBegin->framebuffer = Unwrap(dev_data, pRenderPassBegin->framebuffer);
            }
        }
    }
    dev_data->dispatch_table.CmdBeginRenderPass(commandBuffer, (const VkRenderPassBeginInfo*)local_pRenderPassBegin, contents);
    if (local_pRenderPassBegin) {
        delete local_pRenderPassBegin;
    }
}

VKAPI_ATTR void VKAPI_CALL DestroySurfaceKHR(
    VkInstance                                  instance,
    VkSurfaceKHR                                surface,
    const VkAllocationCallbacks*                pAllocator)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t surface_id = reinterpret_cast<uint64_t &>(surface);
    surface = (VkSurfaceKHR)dev_data->unique_id_mapping[surface_id];
    dev_data->unique_id_mapping.erase(surface_id);
    lock.unlock();
    dev_data->dispatch_table.DestroySurfaceKHR(instance, surface, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    VkSurfaceKHR                                surface,
    VkBool32*                                   pSupported)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    VkSurfaceCapabilitiesKHR*                   pSurfaceCapabilities)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pSurfaceFormatCount,
    VkSurfaceFormatKHR*                         pSurfaceFormats)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pPresentModeCount,
    VkPresentModeKHR*                           pPresentModes)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes);

    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(
    VkDevice                                    device,
    const VkSwapchainCreateInfoKHR*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSwapchainKHR*                             pSwapchain);

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t swapchain_id = reinterpret_cast<uint64_t &>(swapchain);
    swapchain = (VkSwapchainKHR)dev_data->unique_id_mapping[swapchain_id];
    dev_data->unique_id_mapping.erase(swapchain_id);
    lock.unlock();
    dev_data->dispatch_table.DestroySwapchainKHR(device, swapchain, pAllocator);

}

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        swapchain = Unwrap(dev_data, swapchain);
        semaphore = Unwrap(dev_data, semaphore);
        fence = Unwrap(dev_data, fence);
    }
    VkResult result = dev_data->dispatch_table.AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);

    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(
    VkQueue                                     queue,
    const VkPresentInfoKHR*                     pPresentInfo);

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayModeKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display,
    const VkDisplayModeCreateInfoKHR*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDisplayModeKHR*                           pMode)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateDisplayModeKHR(physicalDevice, display, pCreateInfo, pAllocator, pMode);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pMode = WrapNew(dev_data, *pMode);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayPlaneSurfaceKHR(
    VkInstance                                  instance,
    const VkDisplaySurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    safe_VkDisplaySurfaceCreateInfoKHR *local_pCreateInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pCreateInfo) {
            local_pCreateInfo = new safe_VkDisplaySurfaceCreateInfoKHR(pCreateInfo);
            if (pCreateInfo->displayMode) {
                local_pCreateInfo->displayMode = Unwrap(dev_data, pCreateInfo->displayMode);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.CreateDisplayPlaneSurfaceKHR(instance, (const VkDisplaySurfaceCreateInfoKHR*)local_pCreateInfo, pAllocator, pSurface);
    if (local_pCreateInfo) {
        delete local_pCreateInfo;
    }
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
    return result;
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateSharedSwapchainsKHR(
    VkDevice                                    device,
    uint32_t                                    swapchainCount,
    const VkSwapchainCreateInfoKHR*             pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkSwapchainKHR*                             pSwapchains);

#ifdef VK_USE_PLATFORM_XLIB_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateXlibSurfaceKHR(
    VkInstance                                  instance,
    const VkXlibSurfaceCreateInfoKHR*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateXlibSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateXcbSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateWaylandSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateMirSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateAndroidSurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR void VKAPI_CALL TrimCommandPoolKHR(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolTrimFlagsKHR                   flags)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        commandPool = Unwrap(dev_data, commandPool);
    }
    dev_data->dispatch_table.TrimCommandPoolKHR(device, commandPool, flags);

}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleKHR(
    VkDevice                                    device,
    const VkMemoryGetWin32HandleInfoKHR*        pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkMemoryGetWin32HandleInfoKHR *local_pGetWin32HandleInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pGetWin32HandleInfo) {
            local_pGetWin32HandleInfo = new safe_VkMemoryGetWin32HandleInfoKHR(pGetWin32HandleInfo);
            if (pGetWin32HandleInfo->memory) {
                local_pGetWin32HandleInfo->memory = Unwrap(dev_data, pGetWin32HandleInfo->memory);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetMemoryWin32HandleKHR(device, (const VkMemoryGetWin32HandleInfoKHR*)local_pGetWin32HandleInfo, pHandle);
    if (local_pGetWin32HandleInfo) {
        delete local_pGetWin32HandleInfo;
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdKHR(
    VkDevice                                    device,
    const VkMemoryGetFdInfoKHR*                 pGetFdInfo,
    int*                                        pFd)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkMemoryGetFdInfoKHR *local_pGetFdInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pGetFdInfo) {
            local_pGetFdInfo = new safe_VkMemoryGetFdInfoKHR(pGetFdInfo);
            if (pGetFdInfo->memory) {
                local_pGetFdInfo->memory = Unwrap(dev_data, pGetFdInfo->memory);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetMemoryFdKHR(device, (const VkMemoryGetFdInfoKHR*)local_pGetFdInfo, pFd);
    if (local_pGetFdInfo) {
        delete local_pGetFdInfo;
    }
    return result;
}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkImportSemaphoreWin32HandleInfoKHR*  pImportSemaphoreWin32HandleInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImportSemaphoreWin32HandleInfoKHR *local_pImportSemaphoreWin32HandleInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pImportSemaphoreWin32HandleInfo) {
            local_pImportSemaphoreWin32HandleInfo = new safe_VkImportSemaphoreWin32HandleInfoKHR(pImportSemaphoreWin32HandleInfo);
            if (pImportSemaphoreWin32HandleInfo->semaphore) {
                local_pImportSemaphoreWin32HandleInfo->semaphore = Unwrap(dev_data, pImportSemaphoreWin32HandleInfo->semaphore);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.ImportSemaphoreWin32HandleKHR(device, (const VkImportSemaphoreWin32HandleInfoKHR*)local_pImportSemaphoreWin32HandleInfo);
    if (local_pImportSemaphoreWin32HandleInfo) {
        delete local_pImportSemaphoreWin32HandleInfo;
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkSemaphoreGetWin32HandleInfoKHR*     pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkSemaphoreGetWin32HandleInfoKHR *local_pGetWin32HandleInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pGetWin32HandleInfo) {
            local_pGetWin32HandleInfo = new safe_VkSemaphoreGetWin32HandleInfoKHR(pGetWin32HandleInfo);
            if (pGetWin32HandleInfo->semaphore) {
                local_pGetWin32HandleInfo->semaphore = Unwrap(dev_data, pGetWin32HandleInfo->semaphore);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetSemaphoreWin32HandleKHR(device, (const VkSemaphoreGetWin32HandleInfoKHR*)local_pGetWin32HandleInfo, pHandle);
    if (local_pGetWin32HandleInfo) {
        delete local_pGetWin32HandleInfo;
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreFdKHR(
    VkDevice                                    device,
    const VkImportSemaphoreFdInfoKHR*           pImportSemaphoreFdInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImportSemaphoreFdInfoKHR *local_pImportSemaphoreFdInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pImportSemaphoreFdInfo) {
            local_pImportSemaphoreFdInfo = new safe_VkImportSemaphoreFdInfoKHR(pImportSemaphoreFdInfo);
            if (pImportSemaphoreFdInfo->semaphore) {
                local_pImportSemaphoreFdInfo->semaphore = Unwrap(dev_data, pImportSemaphoreFdInfo->semaphore);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.ImportSemaphoreFdKHR(device, (const VkImportSemaphoreFdInfoKHR*)local_pImportSemaphoreFdInfo);
    if (local_pImportSemaphoreFdInfo) {
        delete local_pImportSemaphoreFdInfo;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreFdKHR(
    VkDevice                                    device,
    const VkSemaphoreGetFdInfoKHR*              pGetFdInfo,
    int*                                        pFd)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkSemaphoreGetFdInfoKHR *local_pGetFdInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pGetFdInfo) {
            local_pGetFdInfo = new safe_VkSemaphoreGetFdInfoKHR(pGetFdInfo);
            if (pGetFdInfo->semaphore) {
                local_pGetFdInfo->semaphore = Unwrap(dev_data, pGetFdInfo->semaphore);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetSemaphoreFdKHR(device, (const VkSemaphoreGetFdInfoKHR*)local_pGetFdInfo, pFd);
    if (local_pGetFdInfo) {
        delete local_pGetFdInfo;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    safe_VkWriteDescriptorSet *local_pDescriptorWrites = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        layout = Unwrap(dev_data, layout);
        if (pDescriptorWrites) {
            local_pDescriptorWrites = new safe_VkWriteDescriptorSet[descriptorWriteCount];
            for (uint32_t index0 = 0; index0 < descriptorWriteCount; ++index0) {
                local_pDescriptorWrites[index0].initialize(&pDescriptorWrites[index0]);
                if (pDescriptorWrites[index0].dstSet) {
                    local_pDescriptorWrites[index0].dstSet = Unwrap(dev_data, pDescriptorWrites[index0].dstSet);
                }
                if (local_pDescriptorWrites[index0].pImageInfo) {
                    for (uint32_t index1 = 0; index1 < local_pDescriptorWrites[index0].descriptorCount; ++index1) {
                        if (pDescriptorWrites[index0].pImageInfo[index1].sampler) {
                            local_pDescriptorWrites[index0].pImageInfo[index1].sampler = Unwrap(dev_data, pDescriptorWrites[index0].pImageInfo[index1].sampler);
                        }
                        if (pDescriptorWrites[index0].pImageInfo[index1].imageView) {
                            local_pDescriptorWrites[index0].pImageInfo[index1].imageView = Unwrap(dev_data, pDescriptorWrites[index0].pImageInfo[index1].imageView);
                        }
                    }
                }
                if (local_pDescriptorWrites[index0].pBufferInfo) {
                    for (uint32_t index1 = 0; index1 < local_pDescriptorWrites[index0].descriptorCount; ++index1) {
                        if (pDescriptorWrites[index0].pBufferInfo[index1].buffer) {
                            local_pDescriptorWrites[index0].pBufferInfo[index1].buffer = Unwrap(dev_data, pDescriptorWrites[index0].pBufferInfo[index1].buffer);
                        }
                    }
                }
                if (local_pDescriptorWrites[index0].pTexelBufferView) {
                    for (uint32_t index1 = 0; index1 < local_pDescriptorWrites[index0].descriptorCount; ++index1) {
                        local_pDescriptorWrites[index0].pTexelBufferView[index1] = Unwrap(dev_data, local_pDescriptorWrites[index0].pTexelBufferView[index1]);
                    }
                }
            }
        }
    }
    dev_data->dispatch_table.CmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, (const VkWriteDescriptorSet*)local_pDescriptorWrites);
    if (local_pDescriptorWrites) {
        delete[] local_pDescriptorWrites;
    }
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorUpdateTemplateKHR*              pDescriptorUpdateTemplate);

// Declare only
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    const VkAllocationCallbacks*                pAllocator);

// Declare only
VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSetWithTemplateKHR(
    VkDevice                                    device,
    VkDescriptorSet                             descriptorSet,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    const void*                                 pData);

// Declare only
VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer                             commandBuffer,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    const void*                                 pData);

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainStatusKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        swapchain = Unwrap(dev_data, swapchain);
    }
    VkResult result = dev_data->dispatch_table.GetSwapchainStatusKHR(device, swapchain);

    return result;
}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkImportFenceWin32HandleInfoKHR*      pImportFenceWin32HandleInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImportFenceWin32HandleInfoKHR *local_pImportFenceWin32HandleInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pImportFenceWin32HandleInfo) {
            local_pImportFenceWin32HandleInfo = new safe_VkImportFenceWin32HandleInfoKHR(pImportFenceWin32HandleInfo);
            if (pImportFenceWin32HandleInfo->fence) {
                local_pImportFenceWin32HandleInfo->fence = Unwrap(dev_data, pImportFenceWin32HandleInfo->fence);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.ImportFenceWin32HandleKHR(device, (const VkImportFenceWin32HandleInfoKHR*)local_pImportFenceWin32HandleInfo);
    if (local_pImportFenceWin32HandleInfo) {
        delete local_pImportFenceWin32HandleInfo;
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkFenceGetWin32HandleInfoKHR*         pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkFenceGetWin32HandleInfoKHR *local_pGetWin32HandleInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pGetWin32HandleInfo) {
            local_pGetWin32HandleInfo = new safe_VkFenceGetWin32HandleInfoKHR(pGetWin32HandleInfo);
            if (pGetWin32HandleInfo->fence) {
                local_pGetWin32HandleInfo->fence = Unwrap(dev_data, pGetWin32HandleInfo->fence);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetFenceWin32HandleKHR(device, (const VkFenceGetWin32HandleInfoKHR*)local_pGetWin32HandleInfo, pHandle);
    if (local_pGetWin32HandleInfo) {
        delete local_pGetWin32HandleInfo;
    }
    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceFdKHR(
    VkDevice                                    device,
    const VkImportFenceFdInfoKHR*               pImportFenceFdInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImportFenceFdInfoKHR *local_pImportFenceFdInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pImportFenceFdInfo) {
            local_pImportFenceFdInfo = new safe_VkImportFenceFdInfoKHR(pImportFenceFdInfo);
            if (pImportFenceFdInfo->fence) {
                local_pImportFenceFdInfo->fence = Unwrap(dev_data, pImportFenceFdInfo->fence);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.ImportFenceFdKHR(device, (const VkImportFenceFdInfoKHR*)local_pImportFenceFdInfo);
    if (local_pImportFenceFdInfo) {
        delete local_pImportFenceFdInfo;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceFdKHR(
    VkDevice                                    device,
    const VkFenceGetFdInfoKHR*                  pGetFdInfo,
    int*                                        pFd)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkFenceGetFdInfoKHR *local_pGetFdInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pGetFdInfo) {
            local_pGetFdInfo = new safe_VkFenceGetFdInfoKHR(pGetFdInfo);
            if (pGetFdInfo->fence) {
                local_pGetFdInfo->fence = Unwrap(dev_data, pGetFdInfo->fence);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetFenceFdKHR(device, (const VkFenceGetFdInfoKHR*)local_pGetFdInfo, pFd);
    if (local_pGetFdInfo) {
        delete local_pGetFdInfo;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    VkSurfaceCapabilities2KHR*                  pSurfaceCapabilities)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    safe_VkPhysicalDeviceSurfaceInfo2KHR *local_pSurfaceInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pSurfaceInfo) {
            local_pSurfaceInfo = new safe_VkPhysicalDeviceSurfaceInfo2KHR(pSurfaceInfo);
            if (pSurfaceInfo->surface) {
                local_pSurfaceInfo->surface = Unwrap(dev_data, pSurfaceInfo->surface);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfaceCapabilities2KHR(physicalDevice, (const VkPhysicalDeviceSurfaceInfo2KHR*)local_pSurfaceInfo, pSurfaceCapabilities);
    if (local_pSurfaceInfo) {
        delete local_pSurfaceInfo;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormats2KHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    uint32_t*                                   pSurfaceFormatCount,
    VkSurfaceFormat2KHR*                        pSurfaceFormats)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    safe_VkPhysicalDeviceSurfaceInfo2KHR *local_pSurfaceInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pSurfaceInfo) {
            local_pSurfaceInfo = new safe_VkPhysicalDeviceSurfaceInfo2KHR(pSurfaceInfo);
            if (pSurfaceInfo->surface) {
                local_pSurfaceInfo->surface = Unwrap(dev_data, pSurfaceInfo->surface);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfaceFormats2KHR(physicalDevice, (const VkPhysicalDeviceSurfaceInfo2KHR*)local_pSurfaceInfo, pSurfaceFormatCount, pSurfaceFormats);
    if (local_pSurfaceInfo) {
        delete local_pSurfaceInfo;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageMemoryRequirementsInfo2KHR*    pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImageMemoryRequirementsInfo2KHR *local_pInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pInfo) {
            local_pInfo = new safe_VkImageMemoryRequirementsInfo2KHR(pInfo);
            if (pInfo->image) {
                local_pInfo->image = Unwrap(dev_data, pInfo->image);
            }
        }
    }
    dev_data->dispatch_table.GetImageMemoryRequirements2KHR(device, (const VkImageMemoryRequirementsInfo2KHR*)local_pInfo, pMemoryRequirements);
    if (local_pInfo) {
        delete local_pInfo;
    }
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkBufferMemoryRequirementsInfo2KHR*   pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkBufferMemoryRequirementsInfo2KHR *local_pInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pInfo) {
            local_pInfo = new safe_VkBufferMemoryRequirementsInfo2KHR(pInfo);
            if (pInfo->buffer) {
                local_pInfo->buffer = Unwrap(dev_data, pInfo->buffer);
            }
        }
    }
    dev_data->dispatch_table.GetBufferMemoryRequirements2KHR(device, (const VkBufferMemoryRequirementsInfo2KHR*)local_pInfo, pMemoryRequirements);
    if (local_pInfo) {
        delete local_pInfo;
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageSparseMemoryRequirementsInfo2KHR* pInfo,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements2KHR*        pSparseMemoryRequirements)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkImageSparseMemoryRequirementsInfo2KHR *local_pInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pInfo) {
            local_pInfo = new safe_VkImageSparseMemoryRequirementsInfo2KHR(pInfo);
            if (pInfo->image) {
                local_pInfo->image = Unwrap(dev_data, pInfo->image);
            }
        }
    }
    dev_data->dispatch_table.GetImageSparseMemoryRequirements2KHR(device, (const VkImageSparseMemoryRequirementsInfo2KHR*)local_pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
    if (local_pInfo) {
        delete local_pInfo;
    }
}

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectTagEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectTagInfoEXT*        pTagInfo);

// Declare only
VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectNameEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectNameInfoEXT*       pNameInfo);

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountAMD(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
        countBuffer = Unwrap(dev_data, countBuffer);
    }
    dev_data->dispatch_table.CmdDrawIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);

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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        buffer = Unwrap(dev_data, buffer);
        countBuffer = Unwrap(dev_data, countBuffer);
    }
    dev_data->dispatch_table.CmdDrawIndexedIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);

}

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleNV(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkExternalMemoryHandleTypeFlagsNV           handleType,
    HANDLE*                                     pHandle)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        memory = Unwrap(dev_data, memory);
    }
    VkResult result = dev_data->dispatch_table.GetMemoryWin32HandleNV(device, memory, handleType, pHandle);

    return result;
}
#endif // VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory2KHX(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindBufferMemoryInfoKHX*            pBindInfos)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkBindBufferMemoryInfoKHX *local_pBindInfos = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pBindInfos) {
            local_pBindInfos = new safe_VkBindBufferMemoryInfoKHX[bindInfoCount];
            for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
                local_pBindInfos[index0].initialize(&pBindInfos[index0]);
                if (pBindInfos[index0].buffer) {
                    local_pBindInfos[index0].buffer = Unwrap(dev_data, pBindInfos[index0].buffer);
                }
                if (pBindInfos[index0].memory) {
                    local_pBindInfos[index0].memory = Unwrap(dev_data, pBindInfos[index0].memory);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.BindBufferMemory2KHX(device, bindInfoCount, (const VkBindBufferMemoryInfoKHX*)local_pBindInfos);
    if (local_pBindInfos) {
        delete[] local_pBindInfos;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2KHX(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindImageMemoryInfoKHX*             pBindInfos)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkBindImageMemoryInfoKHX *local_pBindInfos = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pBindInfos) {
            local_pBindInfos = new safe_VkBindImageMemoryInfoKHX[bindInfoCount];
            for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
                local_pBindInfos[index0].initialize(&pBindInfos[index0]);
                local_pBindInfos[index0].pNext = CreateUnwrappedExtensionStructs(dev_data, local_pBindInfos[index0].pNext);
                if (pBindInfos[index0].image) {
                    local_pBindInfos[index0].image = Unwrap(dev_data, pBindInfos[index0].image);
                }
                if (pBindInfos[index0].memory) {
                    local_pBindInfos[index0].memory = Unwrap(dev_data, pBindInfos[index0].memory);
                }
            }
        }
    }
    VkResult result = dev_data->dispatch_table.BindImageMemory2KHX(device, bindInfoCount, (const VkBindImageMemoryInfoKHX*)local_pBindInfos);
    if (local_pBindInfos) {
        for (uint32_t index0 = 0; index0 < bindInfoCount; ++index0) {
            FreeUnwrappedExtensionStructs(const_cast<void *>(local_pBindInfos[index0].pNext));
        }
        delete[] local_pBindInfos;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupSurfacePresentModesKHX(
    VkDevice                                    device,
    VkSurfaceKHR                                surface,
    VkDeviceGroupPresentModeFlagsKHX*           pModes)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetDeviceGroupSurfacePresentModesKHX(device, surface, pModes);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImage2KHX(
    VkDevice                                    device,
    const VkAcquireNextImageInfoKHX*            pAcquireInfo,
    uint32_t*                                   pImageIndex)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    safe_VkAcquireNextImageInfoKHX *local_pAcquireInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pAcquireInfo) {
            local_pAcquireInfo = new safe_VkAcquireNextImageInfoKHX(pAcquireInfo);
            if (pAcquireInfo->swapchain) {
                local_pAcquireInfo->swapchain = Unwrap(dev_data, pAcquireInfo->swapchain);
            }
            if (pAcquireInfo->semaphore) {
                local_pAcquireInfo->semaphore = Unwrap(dev_data, pAcquireInfo->semaphore);
            }
            if (pAcquireInfo->fence) {
                local_pAcquireInfo->fence = Unwrap(dev_data, pAcquireInfo->fence);
            }
        }
    }
    VkResult result = dev_data->dispatch_table.AcquireNextImage2KHX(device, (const VkAcquireNextImageInfoKHX*)local_pAcquireInfo, pImageIndex);
    if (local_pAcquireInfo) {
        delete local_pAcquireInfo;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDevicePresentRectanglesKHX(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pRectCount,
    VkRect2D*                                   pRects)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDevicePresentRectanglesKHX(physicalDevice, surface, pRectCount, pRects);

    return result;
}

#ifdef VK_USE_PLATFORM_VI_NN

VKAPI_ATTR VkResult VKAPI_CALL CreateViSurfaceNN(
    VkInstance                                  instance,
    const VkViSurfaceCreateInfoNN*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateViSurfaceNN(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
    return result;
}
#endif // VK_USE_PLATFORM_VI_NN

VKAPI_ATTR void VKAPI_CALL CmdProcessCommandsNVX(
    VkCommandBuffer                             commandBuffer,
    const VkCmdProcessCommandsInfoNVX*          pProcessCommandsInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    safe_VkCmdProcessCommandsInfoNVX *local_pProcessCommandsInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pProcessCommandsInfo) {
            local_pProcessCommandsInfo = new safe_VkCmdProcessCommandsInfoNVX(pProcessCommandsInfo);
            if (pProcessCommandsInfo->objectTable) {
                local_pProcessCommandsInfo->objectTable = Unwrap(dev_data, pProcessCommandsInfo->objectTable);
            }
            if (pProcessCommandsInfo->indirectCommandsLayout) {
                local_pProcessCommandsInfo->indirectCommandsLayout = Unwrap(dev_data, pProcessCommandsInfo->indirectCommandsLayout);
            }
            if (local_pProcessCommandsInfo->pIndirectCommandsTokens) {
                for (uint32_t index1 = 0; index1 < local_pProcessCommandsInfo->indirectCommandsTokenCount; ++index1) {
                    if (pProcessCommandsInfo->pIndirectCommandsTokens[index1].buffer) {
                        local_pProcessCommandsInfo->pIndirectCommandsTokens[index1].buffer = Unwrap(dev_data, pProcessCommandsInfo->pIndirectCommandsTokens[index1].buffer);
                    }
                }
            }
            if (pProcessCommandsInfo->sequencesCountBuffer) {
                local_pProcessCommandsInfo->sequencesCountBuffer = Unwrap(dev_data, pProcessCommandsInfo->sequencesCountBuffer);
            }
            if (pProcessCommandsInfo->sequencesIndexBuffer) {
                local_pProcessCommandsInfo->sequencesIndexBuffer = Unwrap(dev_data, pProcessCommandsInfo->sequencesIndexBuffer);
            }
        }
    }
    dev_data->dispatch_table.CmdProcessCommandsNVX(commandBuffer, (const VkCmdProcessCommandsInfoNVX*)local_pProcessCommandsInfo);
    if (local_pProcessCommandsInfo) {
        delete local_pProcessCommandsInfo;
    }
}

VKAPI_ATTR void VKAPI_CALL CmdReserveSpaceForCommandsNVX(
    VkCommandBuffer                             commandBuffer,
    const VkCmdReserveSpaceForCommandsInfoNVX*  pReserveSpaceInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(commandBuffer), layer_data_map);
    safe_VkCmdReserveSpaceForCommandsInfoNVX *local_pReserveSpaceInfo = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pReserveSpaceInfo) {
            local_pReserveSpaceInfo = new safe_VkCmdReserveSpaceForCommandsInfoNVX(pReserveSpaceInfo);
            if (pReserveSpaceInfo->objectTable) {
                local_pReserveSpaceInfo->objectTable = Unwrap(dev_data, pReserveSpaceInfo->objectTable);
            }
            if (pReserveSpaceInfo->indirectCommandsLayout) {
                local_pReserveSpaceInfo->indirectCommandsLayout = Unwrap(dev_data, pReserveSpaceInfo->indirectCommandsLayout);
            }
        }
    }
    dev_data->dispatch_table.CmdReserveSpaceForCommandsNVX(commandBuffer, (const VkCmdReserveSpaceForCommandsInfoNVX*)local_pReserveSpaceInfo);
    if (local_pReserveSpaceInfo) {
        delete local_pReserveSpaceInfo;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateIndirectCommandsLayoutNVX(
    VkDevice                                    device,
    const VkIndirectCommandsLayoutCreateInfoNVX* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkIndirectCommandsLayoutNVX*                pIndirectCommandsLayout)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateIndirectCommandsLayoutNVX(device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pIndirectCommandsLayout = WrapNew(dev_data, *pIndirectCommandsLayout);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyIndirectCommandsLayoutNVX(
    VkDevice                                    device,
    VkIndirectCommandsLayoutNVX                 indirectCommandsLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t indirectCommandsLayout_id = reinterpret_cast<uint64_t &>(indirectCommandsLayout);
    indirectCommandsLayout = (VkIndirectCommandsLayoutNVX)dev_data->unique_id_mapping[indirectCommandsLayout_id];
    dev_data->unique_id_mapping.erase(indirectCommandsLayout_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyIndirectCommandsLayoutNVX(device, indirectCommandsLayout, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL CreateObjectTableNVX(
    VkDevice                                    device,
    const VkObjectTableCreateInfoNVX*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkObjectTableNVX*                           pObjectTable)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateObjectTableNVX(device, pCreateInfo, pAllocator, pObjectTable);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pObjectTable = WrapNew(dev_data, *pObjectTable);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyObjectTableNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    const VkAllocationCallbacks*                pAllocator)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t objectTable_id = reinterpret_cast<uint64_t &>(objectTable);
    objectTable = (VkObjectTableNVX)dev_data->unique_id_mapping[objectTable_id];
    dev_data->unique_id_mapping.erase(objectTable_id);
    lock.unlock();
    dev_data->dispatch_table.DestroyObjectTableNVX(device, objectTable, pAllocator);

}

VKAPI_ATTR VkResult VKAPI_CALL RegisterObjectsNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    uint32_t                                    objectCount,
    const VkObjectTableEntryNVX* const*         ppObjectTableEntries,
    const uint32_t*                             pObjectIndices)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        objectTable = Unwrap(dev_data, objectTable);
    }
    VkResult result = dev_data->dispatch_table.RegisterObjectsNVX(device, objectTable, objectCount, ppObjectTableEntries, pObjectIndices);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL UnregisterObjectsNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    uint32_t                                    objectCount,
    const VkObjectEntryTypeNVX*                 pObjectEntryTypes,
    const uint32_t*                             pObjectIndices)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        objectTable = Unwrap(dev_data, objectTable);
    }
    VkResult result = dev_data->dispatch_table.UnregisterObjectsNVX(device, objectTable, objectCount, pObjectEntryTypes, pObjectIndices);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ReleaseDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        display = Unwrap(dev_data, display);
    }
    VkResult result = dev_data->dispatch_table.ReleaseDisplayEXT(physicalDevice, display);

    return result;
}

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT

VKAPI_ATTR VkResult VKAPI_CALL AcquireXlibDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    Display*                                    dpy,
    VkDisplayKHR                                display)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        display = Unwrap(dev_data, display);
    }
    VkResult result = dev_data->dispatch_table.AcquireXlibDisplayEXT(physicalDevice, dpy, display);

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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.GetRandROutputDisplayEXT(physicalDevice, dpy, rrOutput, pDisplay);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pDisplay = WrapNew(dev_data, *pDisplay);
    }
    return result;
}
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2EXT(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    VkSurfaceCapabilities2EXT*                  pSurfaceCapabilities)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), instance_layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        surface = Unwrap(dev_data, surface);
    }
    VkResult result = dev_data->dispatch_table.GetPhysicalDeviceSurfaceCapabilities2EXT(physicalDevice, surface, pSurfaceCapabilities);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DisplayPowerControlEXT(
    VkDevice                                    device,
    VkDisplayKHR                                display,
    const VkDisplayPowerInfoEXT*                pDisplayPowerInfo)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        display = Unwrap(dev_data, display);
    }
    VkResult result = dev_data->dispatch_table.DisplayPowerControlEXT(device, display, pDisplayPowerInfo);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDeviceEventEXT(
    VkDevice                                    device,
    const VkDeviceEventInfoEXT*                 pDeviceEventInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.RegisterDeviceEventEXT(device, pDeviceEventInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pFence = WrapNew(dev_data, *pFence);
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
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkResult result = dev_data->dispatch_table.RegisterDisplayEventEXT(device, display, pDisplayEventInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pFence = WrapNew(dev_data, *pFence);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainCounterEXT(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkSurfaceCounterFlagBitsEXT                 counter,
    uint64_t*                                   pCounterValue)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        swapchain = Unwrap(dev_data, swapchain);
    }
    VkResult result = dev_data->dispatch_table.GetSwapchainCounterEXT(device, swapchain, counter, pCounterValue);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetRefreshCycleDurationGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkRefreshCycleDurationGOOGLE*               pDisplayTimingProperties)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        swapchain = Unwrap(dev_data, swapchain);
    }
    VkResult result = dev_data->dispatch_table.GetRefreshCycleDurationGOOGLE(device, swapchain, pDisplayTimingProperties);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPastPresentationTimingGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint32_t*                                   pPresentationTimingCount,
    VkPastPresentationTimingGOOGLE*             pPresentationTimings)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    {
        std::lock_guard<std::mutex> lock(global_lock);
        swapchain = Unwrap(dev_data, swapchain);
    }
    VkResult result = dev_data->dispatch_table.GetPastPresentationTimingGOOGLE(device, swapchain, pPresentationTimingCount, pPresentationTimings);

    return result;
}

VKAPI_ATTR void VKAPI_CALL SetHdrMetadataEXT(
    VkDevice                                    device,
    uint32_t                                    swapchainCount,
    const VkSwapchainKHR*                       pSwapchains,
    const VkHdrMetadataEXT*                     pMetadata)
{
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    VkSwapchainKHR *local_pSwapchains = NULL;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        if (pSwapchains) {
            local_pSwapchains = new VkSwapchainKHR[swapchainCount];
            for (uint32_t index0 = 0; index0 < swapchainCount; ++index0) {
                local_pSwapchains[index0] = Unwrap(dev_data, pSwapchains[index0]);
            }
        }
    }
    dev_data->dispatch_table.SetHdrMetadataEXT(device, swapchainCount, (const VkSwapchainKHR*)local_pSwapchains, pMetadata);
    if (local_pSwapchains)
        delete[] local_pSwapchains;
}

#ifdef VK_USE_PLATFORM_IOS_MVK

VKAPI_ATTR VkResult VKAPI_CALL CreateIOSSurfaceMVK(
    VkInstance                                  instance,
    const VkIOSSurfaceCreateInfoMVK*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateIOSSurfaceMVK(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
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
    instance_layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(instance), instance_layer_data_map);
    VkResult result = dev_data->dispatch_table.CreateMacOSSurfaceMVK(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
        *pSurface = WrapNew(dev_data, *pSurface);
    }
    return result;
}
#endif // VK_USE_PLATFORM_MACOS_MVK

// Layer Device Extension Whitelist
static const char *kUniqueObjectsSupportedDeviceExtensions =
"VK_KHR_swapchain"
"VK_KHR_display_swapchain"
"VK_KHR_sampler_mirror_clamp_to_edge"
"VK_KHR_shader_draw_parameters"
"VK_KHR_maintenance1"
"VK_KHR_external_memory"
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_KHR_external_memory_win32"
#endif
"VK_KHR_external_memory_fd"
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_KHR_win32_keyed_mutex"
#endif
"VK_KHR_external_semaphore"
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_KHR_external_semaphore_win32"
#endif
"VK_KHR_external_semaphore_fd"
"VK_KHR_push_descriptor"
"VK_KHR_16bit_storage"
"VK_KHR_incremental_present"
"VK_KHR_descriptor_update_template"
"VK_KHR_shared_presentable_image"
"VK_KHR_external_fence"
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_KHR_external_fence_win32"
#endif
"VK_KHR_external_fence_fd"
"VK_KHR_variable_pointers"
"VK_KHR_dedicated_allocation"
"VK_KHR_storage_buffer_storage_class"
"VK_KHR_relaxed_block_layout"
"VK_KHR_get_memory_requirements2"
"VK_NV_glsl_shader"
"VK_EXT_depth_range_unrestricted"
"VK_IMG_filter_cubic"
"VK_AMD_rasterization_order"
"VK_AMD_shader_trinary_minmax"
"VK_AMD_shader_explicit_vertex_parameter"
"VK_EXT_debug_marker"
"VK_AMD_gcn_shader"
"VK_NV_dedicated_allocation"
"VK_AMD_draw_indirect_count"
"VK_AMD_negative_viewport_height"
"VK_AMD_gpu_shader_half_float"
"VK_AMD_shader_ballot"
"VK_AMD_texture_gather_bias_lod"
"VK_KHX_multiview"
"VK_IMG_format_pvrtc"
"VK_NV_external_memory"
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_NV_external_memory_win32"
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_NV_win32_keyed_mutex"
#endif
"VK_KHX_device_group"
"VK_EXT_shader_subgroup_ballot"
"VK_EXT_shader_subgroup_vote"
"VK_NVX_device_generated_commands"
"VK_NV_clip_space_w_scaling"
"VK_EXT_display_control"
"VK_GOOGLE_display_timing"
"VK_NV_sample_mask_override_coverage"
"VK_NV_geometry_shader_passthrough"
"VK_NV_viewport_array2"
"VK_NVX_multiview_per_view_attributes"
"VK_NV_viewport_swizzle"
"VK_EXT_discard_rectangles"
"VK_EXT_hdr_metadata"
"VK_EXT_sampler_filter_minmax"
"VK_AMD_gpu_shader_int16"
"VK_EXT_blend_operation_advanced"
"VK_NV_fragment_coverage_to_color"
"VK_NV_framebuffer_mixed_samples"
"VK_NV_fill_rectangle"
"VK_EXT_post_depth_coverage"
;


// Layer Instance Extension Whitelist
static const char *kUniqueObjectsSupportedInstanceExtensions =
"VK_KHR_surface"
"VK_KHR_display"
#ifdef VK_USE_PLATFORM_XLIB_KHR
"VK_KHR_xlib_surface"
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
"VK_KHR_xcb_surface"
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
"VK_KHR_wayland_surface"
#endif
#ifdef VK_USE_PLATFORM_MIR_KHR
"VK_KHR_mir_surface"
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
"VK_KHR_android_surface"
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
"VK_KHR_win32_surface"
#endif
"VK_KHR_get_physical_device_properties2"
"VK_KHR_external_memory_capabilities"
"VK_KHR_external_semaphore_capabilities"
"VK_KHR_external_fence_capabilities"
"VK_KHR_get_surface_capabilities2"
"VK_EXT_debug_report"
"VK_NV_external_memory_capabilities"
"VK_EXT_validation_flags"
#ifdef VK_USE_PLATFORM_VI_NN
"VK_NN_vi_surface"
#endif
"VK_KHX_device_group_creation"
"VK_EXT_direct_mode_display"
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
"VK_EXT_acquire_xlib_display"
#endif
"VK_EXT_display_surface_counter"
"VK_EXT_swapchain_colorspace"
#ifdef VK_USE_PLATFORM_IOS_MVK
"VK_MVK_ios_surface"
#endif
#ifdef VK_USE_PLATFORM_MACOS_MVK
"VK_MVK_macos_surface"
#endif
;


// Map of all APIs to be intercepted by this layer
static const std::unordered_map<std::string, void*> name_to_funcptr_map = {
    {"vkCreateInstance", (void *)CreateInstance},
    {"vkDestroyInstance", (void *)DestroyInstance},
    {"vkGetInstanceProcAddr", (void *)GetInstanceProcAddr},
    {"vkGetDeviceProcAddr", (void *)GetDeviceProcAddr},
    {"vkCreateDevice", (void *)CreateDevice},
    {"vkDestroyDevice", (void *)DestroyDevice},
    {"vkEnumerateInstanceExtensionProperties", (void *)EnumerateInstanceExtensionProperties},
    {"vkEnumerateInstanceLayerProperties", (void *)EnumerateInstanceLayerProperties},
    {"vkEnumerateDeviceLayerProperties", (void *)EnumerateDeviceLayerProperties},
    {"vkQueueSubmit", (void*)QueueSubmit},
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
    {"vkCreateGraphicsPipelines", (void *)CreateGraphicsPipelines},
    {"vkCreateComputePipelines", (void *)CreateComputePipelines},
    {"vkDestroyPipeline", (void*)DestroyPipeline},
    {"vkCreatePipelineLayout", (void*)CreatePipelineLayout},
    {"vkDestroyPipelineLayout", (void*)DestroyPipelineLayout},
    {"vkCreateSampler", (void*)CreateSampler},
    {"vkDestroySampler", (void*)DestroySampler},
    {"vkCreateDescriptorSetLayout", (void*)CreateDescriptorSetLayout},
    {"vkDestroyDescriptorSetLayout", (void*)DestroyDescriptorSetLayout},
    {"vkCreateDescriptorPool", (void*)CreateDescriptorPool},
    {"vkDestroyDescriptorPool", (void*)DestroyDescriptorPool},
    {"vkResetDescriptorPool", (void*)ResetDescriptorPool},
    {"vkAllocateDescriptorSets", (void*)AllocateDescriptorSets},
    {"vkFreeDescriptorSets", (void*)FreeDescriptorSets},
    {"vkUpdateDescriptorSets", (void*)UpdateDescriptorSets},
    {"vkCreateFramebuffer", (void*)CreateFramebuffer},
    {"vkDestroyFramebuffer", (void*)DestroyFramebuffer},
    {"vkCreateRenderPass", (void*)CreateRenderPass},
    {"vkDestroyRenderPass", (void*)DestroyRenderPass},
    {"vkGetRenderAreaGranularity", (void*)GetRenderAreaGranularity},
    {"vkCreateCommandPool", (void*)CreateCommandPool},
    {"vkDestroyCommandPool", (void*)DestroyCommandPool},
    {"vkResetCommandPool", (void*)ResetCommandPool},
    {"vkAllocateCommandBuffers", (void*)AllocateCommandBuffers},
    {"vkFreeCommandBuffers", (void*)FreeCommandBuffers},
    {"vkBeginCommandBuffer", (void*)BeginCommandBuffer},
    {"vkCmdBindPipeline", (void*)CmdBindPipeline},
    {"vkCmdBindDescriptorSets", (void*)CmdBindDescriptorSets},
    {"vkCmdBindIndexBuffer", (void*)CmdBindIndexBuffer},
    {"vkCmdBindVertexBuffers", (void*)CmdBindVertexBuffers},
    {"vkCmdDrawIndirect", (void*)CmdDrawIndirect},
    {"vkCmdDrawIndexedIndirect", (void*)CmdDrawIndexedIndirect},
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
    {"vkDestroySurfaceKHR", (void*)DestroySurfaceKHR},
    {"vkGetPhysicalDeviceSurfaceSupportKHR", (void*)GetPhysicalDeviceSurfaceSupportKHR},
    {"vkGetPhysicalDeviceSurfaceCapabilitiesKHR", (void*)GetPhysicalDeviceSurfaceCapabilitiesKHR},
    {"vkGetPhysicalDeviceSurfaceFormatsKHR", (void*)GetPhysicalDeviceSurfaceFormatsKHR},
    {"vkGetPhysicalDeviceSurfacePresentModesKHR", (void*)GetPhysicalDeviceSurfacePresentModesKHR},
    {"vkCreateSwapchainKHR", (void *)CreateSwapchainKHR},
    {"vkDestroySwapchainKHR", (void*)DestroySwapchainKHR},
    {"vkGetSwapchainImagesKHR", (void *)GetSwapchainImagesKHR},
    {"vkAcquireNextImageKHR", (void*)AcquireNextImageKHR},
    {"vkQueuePresentKHR", (void *)QueuePresentKHR},
    {"vkCreateDisplayModeKHR", (void*)CreateDisplayModeKHR},
    {"vkCreateDisplayPlaneSurfaceKHR", (void*)CreateDisplayPlaneSurfaceKHR},
    {"vkCreateSharedSwapchainsKHR", (void *)CreateSharedSwapchainsKHR},
#ifdef VK_USE_PLATFORM_XLIB_KHR
    {"vkCreateXlibSurfaceKHR", (void*)CreateXlibSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
    {"vkCreateXcbSurfaceKHR", (void*)CreateXcbSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    {"vkCreateWaylandSurfaceKHR", (void*)CreateWaylandSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_MIR_KHR
    {"vkCreateMirSurfaceKHR", (void*)CreateMirSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    {"vkCreateAndroidSurfaceKHR", (void*)CreateAndroidSurfaceKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkCreateWin32SurfaceKHR", (void*)CreateWin32SurfaceKHR},
#endif
    {"vkTrimCommandPoolKHR", (void*)TrimCommandPoolKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetMemoryWin32HandleKHR", (void*)GetMemoryWin32HandleKHR},
#endif
    {"vkGetMemoryFdKHR", (void*)GetMemoryFdKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkImportSemaphoreWin32HandleKHR", (void*)ImportSemaphoreWin32HandleKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetSemaphoreWin32HandleKHR", (void*)GetSemaphoreWin32HandleKHR},
#endif
    {"vkImportSemaphoreFdKHR", (void*)ImportSemaphoreFdKHR},
    {"vkGetSemaphoreFdKHR", (void*)GetSemaphoreFdKHR},
    {"vkCmdPushDescriptorSetKHR", (void*)CmdPushDescriptorSetKHR},
    {"vkCreateDescriptorUpdateTemplateKHR", (void *)CreateDescriptorUpdateTemplateKHR},
    {"vkDestroyDescriptorUpdateTemplateKHR", (void *)DestroyDescriptorUpdateTemplateKHR},
    {"vkUpdateDescriptorSetWithTemplateKHR", (void *)UpdateDescriptorSetWithTemplateKHR},
    {"vkCmdPushDescriptorSetWithTemplateKHR", (void *)CmdPushDescriptorSetWithTemplateKHR},
    {"vkGetSwapchainStatusKHR", (void*)GetSwapchainStatusKHR},
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
    {"vkDebugMarkerSetObjectTagEXT", (void *)DebugMarkerSetObjectTagEXT},
    {"vkDebugMarkerSetObjectNameEXT", (void *)DebugMarkerSetObjectNameEXT},
    {"vkCmdDrawIndirectCountAMD", (void*)CmdDrawIndirectCountAMD},
    {"vkCmdDrawIndexedIndirectCountAMD", (void*)CmdDrawIndexedIndirectCountAMD},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetMemoryWin32HandleNV", (void*)GetMemoryWin32HandleNV},
#endif
    {"vkBindBufferMemory2KHX", (void*)BindBufferMemory2KHX},
    {"vkBindImageMemory2KHX", (void*)BindImageMemory2KHX},
    {"vkGetDeviceGroupSurfacePresentModesKHX", (void*)GetDeviceGroupSurfacePresentModesKHX},
    {"vkAcquireNextImage2KHX", (void*)AcquireNextImage2KHX},
    {"vkGetPhysicalDevicePresentRectanglesKHX", (void*)GetPhysicalDevicePresentRectanglesKHX},
#ifdef VK_USE_PLATFORM_VI_NN
    {"vkCreateViSurfaceNN", (void*)CreateViSurfaceNN},
#endif
    {"vkCmdProcessCommandsNVX", (void*)CmdProcessCommandsNVX},
    {"vkCmdReserveSpaceForCommandsNVX", (void*)CmdReserveSpaceForCommandsNVX},
    {"vkCreateIndirectCommandsLayoutNVX", (void*)CreateIndirectCommandsLayoutNVX},
    {"vkDestroyIndirectCommandsLayoutNVX", (void*)DestroyIndirectCommandsLayoutNVX},
    {"vkCreateObjectTableNVX", (void*)CreateObjectTableNVX},
    {"vkDestroyObjectTableNVX", (void*)DestroyObjectTableNVX},
    {"vkRegisterObjectsNVX", (void*)RegisterObjectsNVX},
    {"vkUnregisterObjectsNVX", (void*)UnregisterObjectsNVX},
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
    {"vkSetHdrMetadataEXT", (void*)SetHdrMetadataEXT},
#ifdef VK_USE_PLATFORM_IOS_MVK
    {"vkCreateIOSSurfaceMVK", (void*)CreateIOSSurfaceMVK},
#endif
#ifdef VK_USE_PLATFORM_MACOS_MVK
    {"vkCreateMacOSSurfaceMVK", (void*)CreateMacOSSurfaceMVK},
#endif
};


} // namespace unique_objects

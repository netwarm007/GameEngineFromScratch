#ifndef __thread_check_h_
#define __thread_check_h_ 1

namespace threading {

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



// declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(
    const VkInstanceCreateInfo*                 pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkInstance*                                 pInstance);

// declare only
VKAPI_ATTR void VKAPI_CALL DestroyInstance(
    VkInstance                                  instance,
    const VkAllocationCallbacks*                pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(
    VkInstance                                  instance,
    uint32_t*                                   pPhysicalDeviceCount,
    VkPhysicalDevice*                           pPhysicalDevices)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->EnumeratePhysicalDevices(instance,pPhysicalDeviceCount,pPhysicalDevices);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}

// declare only
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(
    VkInstance                                  instance,
    const char*                                 pName);

// declare only
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(
    VkDevice                                    device,
    const char*                                 pName);

// declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(
    VkPhysicalDevice                            physicalDevice,
    const VkDeviceCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDevice*                                   pDevice);

// declare only
VKAPI_ATTR void VKAPI_CALL DestroyDevice(
    VkDevice                                    device,
    const VkAllocationCallbacks*                pAllocator);

// declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties);

// declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(
    VkPhysicalDevice                            physicalDevice,
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties);

// declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceLayerProperties(
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties);

// declare only
VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceLayerProperties(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties);

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(
    VkDevice                                    device,
    uint32_t                                    queueFamilyIndex,
    uint32_t                                    queueIndex,
    VkQueue*                                    pQueue)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    pTable->GetDeviceQueue(device,queueFamilyIndex,queueIndex,pQueue);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(
    VkQueue                                     queue,
    uint32_t                                    submitCount,
    const VkSubmitInfo*                         pSubmits,
    VkFence                                     fence)
{
    dispatch_key key = get_dispatch_key(queue);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, queue);
        for (uint32_t index=0;index<submitCount;index++) {
            for(uint32_t index2=0;index2<pSubmits[index].waitSemaphoreCount;index2++)
                startWriteObject(my_data, pSubmits[index].pWaitSemaphores[index2]);
            for(uint32_t index2=0;index2<pSubmits[index].signalSemaphoreCount;index2++)
                startWriteObject(my_data, pSubmits[index].pSignalSemaphores[index2]);
        }
        startWriteObject(my_data, fence);
        // Host access to queue must be externally synchronized
        // Host access to pSubmits[].pWaitSemaphores[],pSubmits[].pSignalSemaphores[] must be externally synchronized
        // Host access to fence must be externally synchronized
    }
    result = pTable->QueueSubmit(queue,submitCount,pSubmits,fence);
    if (threadChecks) {
        finishWriteObject(my_data, queue);
        for (uint32_t index=0;index<submitCount;index++) {
            for(uint32_t index2=0;index2<pSubmits[index].waitSemaphoreCount;index2++)
                finishWriteObject(my_data, pSubmits[index].pWaitSemaphores[index2]);
            for(uint32_t index2=0;index2<pSubmits[index].signalSemaphoreCount;index2++)
                finishWriteObject(my_data, pSubmits[index].pSignalSemaphores[index2]);
        }
        finishWriteObject(my_data, fence);
        // Host access to queue must be externally synchronized
        // Host access to pSubmits[].pWaitSemaphores[],pSubmits[].pSignalSemaphores[] must be externally synchronized
        // Host access to fence must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(
    VkQueue                                     queue)
{
    dispatch_key key = get_dispatch_key(queue);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, queue);
    }
    result = pTable->QueueWaitIdle(queue);
    if (threadChecks) {
        finishReadObject(my_data, queue);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(
    VkDevice                                    device)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        // all sname:VkQueue objects created from pname:device must be externally synchronized between host accesses
    }
    result = pTable->DeviceWaitIdle(device);
    if (threadChecks) {
        finishReadObject(my_data, device);
        // all sname:VkQueue objects created from pname:device must be externally synchronized between host accesses
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(
    VkDevice                                    device,
    const VkMemoryAllocateInfo*                 pAllocateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDeviceMemory*                             pMemory)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->AllocateMemory(device,pAllocateInfo,pAllocator,pMemory);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, memory);
        // Host access to memory must be externally synchronized
    }
    pTable->FreeMemory(device,memory,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, memory);
        // Host access to memory must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize                                offset,
    VkDeviceSize                                size,
    VkMemoryMapFlags                            flags,
    void**                                      ppData)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, memory);
        // Host access to memory must be externally synchronized
    }
    result = pTable->MapMemory(device,memory,offset,size,flags,ppData);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, memory);
        // Host access to memory must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(
    VkDevice                                    device,
    VkDeviceMemory                              memory)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, memory);
        // Host access to memory must be externally synchronized
    }
    pTable->UnmapMemory(device,memory);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, memory);
        // Host access to memory must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->FlushMappedMemoryRanges(device,memoryRangeCount,pMemoryRanges);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(
    VkDevice                                    device,
    uint32_t                                    memoryRangeCount,
    const VkMappedMemoryRange*                  pMemoryRanges)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->InvalidateMappedMemoryRanges(device,memoryRangeCount,pMemoryRanges);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceMemoryCommitment(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkDeviceSize*                               pCommittedMemoryInBytes)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, memory);
    }
    pTable->GetDeviceMemoryCommitment(device,memory,pCommittedMemoryInBytes);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, memory);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkDeviceMemory                              memory,
    VkDeviceSize                                memoryOffset)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, buffer);
        startReadObject(my_data, memory);
        // Host access to buffer must be externally synchronized
    }
    result = pTable->BindBufferMemory(device,buffer,memory,memoryOffset);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, buffer);
        finishReadObject(my_data, memory);
        // Host access to buffer must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(
    VkDevice                                    device,
    VkImage                                     image,
    VkDeviceMemory                              memory,
    VkDeviceSize                                memoryOffset)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, image);
        startReadObject(my_data, memory);
        // Host access to image must be externally synchronized
    }
    result = pTable->BindImageMemory(device,image,memory,memoryOffset);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, image);
        finishReadObject(my_data, memory);
        // Host access to image must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    VkMemoryRequirements*                       pMemoryRequirements)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, buffer);
    }
    pTable->GetBufferMemoryRequirements(device,buffer,pMemoryRequirements);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, buffer);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    VkMemoryRequirements*                       pMemoryRequirements)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, image);
    }
    pTable->GetImageMemoryRequirements(device,image,pMemoryRequirements);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, image);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements(
    VkDevice                                    device,
    VkImage                                     image,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements*            pSparseMemoryRequirements)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, image);
    }
    pTable->GetImageSparseMemoryRequirements(device,image,pSparseMemoryRequirementCount,pSparseMemoryRequirements);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, image);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL QueueBindSparse(
    VkQueue                                     queue,
    uint32_t                                    bindInfoCount,
    const VkBindSparseInfo*                     pBindInfo,
    VkFence                                     fence)
{
    dispatch_key key = get_dispatch_key(queue);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, queue);
        for (uint32_t index=0;index<bindInfoCount;index++) {
            for(uint32_t index2=0;index2<pBindInfo[index].waitSemaphoreCount;index2++)
                startWriteObject(my_data, pBindInfo[index].pWaitSemaphores[index2]);
            for(uint32_t index2=0;index2<pBindInfo[index].signalSemaphoreCount;index2++)
                startWriteObject(my_data, pBindInfo[index].pSignalSemaphores[index2]);
            for(uint32_t index2=0;index2<pBindInfo[index].bufferBindCount;index2++)
                startWriteObject(my_data, pBindInfo[index].pBufferBinds[index2].buffer);
            for(uint32_t index2=0;index2<pBindInfo[index].imageOpaqueBindCount;index2++)
                startWriteObject(my_data, pBindInfo[index].pImageOpaqueBinds[index2].image);
            for(uint32_t index2=0;index2<pBindInfo[index].imageBindCount;index2++)
                startWriteObject(my_data, pBindInfo[index].pImageBinds[index2].image);
        }
        startWriteObject(my_data, fence);
        // Host access to queue must be externally synchronized
        // Host access to pBindInfo[].pWaitSemaphores[],pBindInfo[].pSignalSemaphores[],pBindInfo[].pBufferBinds[].buffer,pBindInfo[].pImageOpaqueBinds[].image,pBindInfo[].pImageBinds[].image must be externally synchronized
        // Host access to fence must be externally synchronized
    }
    result = pTable->QueueBindSparse(queue,bindInfoCount,pBindInfo,fence);
    if (threadChecks) {
        finishWriteObject(my_data, queue);
        for (uint32_t index=0;index<bindInfoCount;index++) {
            for(uint32_t index2=0;index2<pBindInfo[index].waitSemaphoreCount;index2++)
                finishWriteObject(my_data, pBindInfo[index].pWaitSemaphores[index2]);
            for(uint32_t index2=0;index2<pBindInfo[index].signalSemaphoreCount;index2++)
                finishWriteObject(my_data, pBindInfo[index].pSignalSemaphores[index2]);
            for(uint32_t index2=0;index2<pBindInfo[index].bufferBindCount;index2++)
                finishWriteObject(my_data, pBindInfo[index].pBufferBinds[index2].buffer);
            for(uint32_t index2=0;index2<pBindInfo[index].imageOpaqueBindCount;index2++)
                finishWriteObject(my_data, pBindInfo[index].pImageOpaqueBinds[index2].image);
            for(uint32_t index2=0;index2<pBindInfo[index].imageBindCount;index2++)
                finishWriteObject(my_data, pBindInfo[index].pImageBinds[index2].image);
        }
        finishWriteObject(my_data, fence);
        // Host access to queue must be externally synchronized
        // Host access to pBindInfo[].pWaitSemaphores[],pBindInfo[].pSignalSemaphores[],pBindInfo[].pBufferBinds[].buffer,pBindInfo[].pImageOpaqueBinds[].image,pBindInfo[].pImageBinds[].image must be externally synchronized
        // Host access to fence must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(
    VkDevice                                    device,
    const VkFenceCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateFence(device,pCreateInfo,pAllocator,pFence);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(
    VkDevice                                    device,
    VkFence                                     fence,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, fence);
        // Host access to fence must be externally synchronized
    }
    pTable->DestroyFence(device,fence,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, fence);
        // Host access to fence must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        for (uint32_t index=0;index<fenceCount;index++) {
            startWriteObject(my_data, pFences[index]);
        }
        // Host access to each member of pFences must be externally synchronized
    }
    result = pTable->ResetFences(device,fenceCount,pFences);
    if (threadChecks) {
        finishReadObject(my_data, device);
        for (uint32_t index=0;index<fenceCount;index++) {
            finishWriteObject(my_data, pFences[index]);
        }
        // Host access to each member of pFences must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(
    VkDevice                                    device,
    VkFence                                     fence)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, fence);
    }
    result = pTable->GetFenceStatus(device,fence);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, fence);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(
    VkDevice                                    device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences,
    VkBool32                                    waitAll,
    uint64_t                                    timeout)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        for (uint32_t index = 0; index < fenceCount; index++) {
            startReadObject(my_data, pFences[index]);
        }
    }
    result = pTable->WaitForFences(device,fenceCount,pFences,waitAll,timeout);
    if (threadChecks) {
        finishReadObject(my_data, device);
        for (uint32_t index = 0; index < fenceCount; index++) {
            finishReadObject(my_data, pFences[index]);
        }
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSemaphore(
    VkDevice                                    device,
    const VkSemaphoreCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSemaphore*                                pSemaphore)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateSemaphore(device,pCreateInfo,pAllocator,pSemaphore);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySemaphore(
    VkDevice                                    device,
    VkSemaphore                                 semaphore,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, semaphore);
        // Host access to semaphore must be externally synchronized
    }
    pTable->DestroySemaphore(device,semaphore,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, semaphore);
        // Host access to semaphore must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateEvent(
    VkDevice                                    device,
    const VkEventCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkEvent*                                    pEvent)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateEvent(device,pCreateInfo,pAllocator,pEvent);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyEvent(
    VkDevice                                    device,
    VkEvent                                     event,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, event);
        // Host access to event must be externally synchronized
    }
    pTable->DestroyEvent(device,event,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, event);
        // Host access to event must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetEventStatus(
    VkDevice                                    device,
    VkEvent                                     event)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, event);
    }
    result = pTable->GetEventStatus(device,event);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, event);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL SetEvent(
    VkDevice                                    device,
    VkEvent                                     event)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, event);
        // Host access to event must be externally synchronized
    }
    result = pTable->SetEvent(device,event);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, event);
        // Host access to event must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetEvent(
    VkDevice                                    device,
    VkEvent                                     event)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, event);
        // Host access to event must be externally synchronized
    }
    result = pTable->ResetEvent(device,event);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, event);
        // Host access to event must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(
    VkDevice                                    device,
    const VkQueryPoolCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkQueryPool*                                pQueryPool)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateQueryPool(device,pCreateInfo,pAllocator,pQueryPool);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, queryPool);
        // Host access to queryPool must be externally synchronized
    }
    pTable->DestroyQueryPool(device,queryPool,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, queryPool);
        // Host access to queryPool must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, queryPool);
    }
    result = pTable->GetQueryPoolResults(device,queryPool,firstQuery,queryCount,dataSize,pData,stride,flags);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, queryPool);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(
    VkDevice                                    device,
    const VkBufferCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBuffer*                                   pBuffer)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateBuffer(device,pCreateInfo,pAllocator,pBuffer);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(
    VkDevice                                    device,
    VkBuffer                                    buffer,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, buffer);
        // Host access to buffer must be externally synchronized
    }
    pTable->DestroyBuffer(device,buffer,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, buffer);
        // Host access to buffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateBufferView(
    VkDevice                                    device,
    const VkBufferViewCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkBufferView*                               pView)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateBufferView(device,pCreateInfo,pAllocator,pView);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyBufferView(
    VkDevice                                    device,
    VkBufferView                                bufferView,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, bufferView);
        // Host access to bufferView must be externally synchronized
    }
    pTable->DestroyBufferView(device,bufferView,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, bufferView);
        // Host access to bufferView must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(
    VkDevice                                    device,
    const VkImageCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImage*                                    pImage)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateImage(device,pCreateInfo,pAllocator,pImage);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(
    VkDevice                                    device,
    VkImage                                     image,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, image);
        // Host access to image must be externally synchronized
    }
    pTable->DestroyImage(device,image,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, image);
        // Host access to image must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageSubresourceLayout(
    VkDevice                                    device,
    VkImage                                     image,
    const VkImageSubresource*                   pSubresource,
    VkSubresourceLayout*                        pLayout)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, image);
    }
    pTable->GetImageSubresourceLayout(device,image,pSubresource,pLayout);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, image);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(
    VkDevice                                    device,
    const VkImageViewCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkImageView*                                pView)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateImageView(device,pCreateInfo,pAllocator,pView);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(
    VkDevice                                    device,
    VkImageView                                 imageView,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, imageView);
        // Host access to imageView must be externally synchronized
    }
    pTable->DestroyImageView(device,imageView,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, imageView);
        // Host access to imageView must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(
    VkDevice                                    device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateShaderModule(device,pCreateInfo,pAllocator,pShaderModule);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(
    VkDevice                                    device,
    VkShaderModule                              shaderModule,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, shaderModule);
        // Host access to shaderModule must be externally synchronized
    }
    pTable->DestroyShaderModule(device,shaderModule,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, shaderModule);
        // Host access to shaderModule must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(
    VkDevice                                    device,
    const VkPipelineCacheCreateInfo*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineCache*                            pPipelineCache)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreatePipelineCache(device,pCreateInfo,pAllocator,pPipelineCache);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, pipelineCache);
        // Host access to pipelineCache must be externally synchronized
    }
    pTable->DestroyPipelineCache(device,pipelineCache,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, pipelineCache);
        // Host access to pipelineCache must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    size_t*                                     pDataSize,
    void*                                       pData)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, pipelineCache);
    }
    result = pTable->GetPipelineCacheData(device,pipelineCache,pDataSize,pData);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, pipelineCache);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(
    VkDevice                                    device,
    VkPipelineCache                             dstCache,
    uint32_t                                    srcCacheCount,
    const VkPipelineCache*                      pSrcCaches)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, dstCache);
        for (uint32_t index = 0; index < srcCacheCount; index++) {
            startReadObject(my_data, pSrcCaches[index]);
        }
        // Host access to dstCache must be externally synchronized
    }
    result = pTable->MergePipelineCaches(device,dstCache,srcCacheCount,pSrcCaches);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, dstCache);
        for (uint32_t index = 0; index < srcCacheCount; index++) {
            finishReadObject(my_data, pSrcCaches[index]);
        }
        // Host access to dstCache must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, pipelineCache);
    }
    result = pTable->CreateGraphicsPipelines(device,pipelineCache,createInfoCount,pCreateInfos,pAllocator,pPipelines);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, pipelineCache);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkComputePipelineCreateInfo*          pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, pipelineCache);
    }
    result = pTable->CreateComputePipelines(device,pipelineCache,createInfoCount,pCreateInfos,pAllocator,pPipelines);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, pipelineCache);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, pipeline);
        // Host access to pipeline must be externally synchronized
    }
    pTable->DestroyPipeline(device,pipeline,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, pipeline);
        // Host access to pipeline must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(
    VkDevice                                    device,
    const VkPipelineLayoutCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineLayout*                           pPipelineLayout)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreatePipelineLayout(device,pCreateInfo,pAllocator,pPipelineLayout);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(
    VkDevice                                    device,
    VkPipelineLayout                            pipelineLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, pipelineLayout);
        // Host access to pipelineLayout must be externally synchronized
    }
    pTable->DestroyPipelineLayout(device,pipelineLayout,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, pipelineLayout);
        // Host access to pipelineLayout must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(
    VkDevice                                    device,
    const VkSamplerCreateInfo*                  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSampler*                                  pSampler)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateSampler(device,pCreateInfo,pAllocator,pSampler);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(
    VkDevice                                    device,
    VkSampler                                   sampler,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, sampler);
        // Host access to sampler must be externally synchronized
    }
    pTable->DestroySampler(device,sampler,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, sampler);
        // Host access to sampler must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorSetLayout*                      pSetLayout)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateDescriptorSetLayout(device,pCreateInfo,pAllocator,pSetLayout);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(
    VkDevice                                    device,
    VkDescriptorSetLayout                       descriptorSetLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, descriptorSetLayout);
        // Host access to descriptorSetLayout must be externally synchronized
    }
    pTable->DestroyDescriptorSetLayout(device,descriptorSetLayout,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, descriptorSetLayout);
        // Host access to descriptorSetLayout must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(
    VkDevice                                    device,
    const VkDescriptorPoolCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorPool*                           pDescriptorPool)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateDescriptorPool(device,pCreateInfo,pAllocator,pDescriptorPool);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, descriptorPool);
        // Host access to descriptorPool must be externally synchronized
    }
    pTable->DestroyDescriptorPool(device,descriptorPool,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, descriptorPool);
        // Host access to descriptorPool must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    VkDescriptorPoolResetFlags                  flags)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, descriptorPool);
        // Host access to descriptorPool must be externally synchronized
        // any sname:VkDescriptorSet objects allocated from pname:descriptorPool must be externally synchronized between host accesses
    }
    result = pTable->ResetDescriptorPool(device,descriptorPool,flags);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, descriptorPool);
        // Host access to descriptorPool must be externally synchronized
        // any sname:VkDescriptorSet objects allocated from pname:descriptorPool must be externally synchronized between host accesses
    } else {
        finishMultiThread();
    }
    return result;
}

// declare only
VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(
    VkDevice                                    device,
    const VkDescriptorSetAllocateInfo*          pAllocateInfo,
    VkDescriptorSet*                            pDescriptorSets);

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(
    VkDevice                                    device,
    VkDescriptorPool                            descriptorPool,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, descriptorPool);
        for (uint32_t index=0;index<descriptorSetCount;index++) {
            startWriteObject(my_data, pDescriptorSets[index]);
        }
        // Host access to descriptorPool must be externally synchronized
        // Host access to each member of pDescriptorSets must be externally synchronized
    }
    result = pTable->FreeDescriptorSets(device,descriptorPool,descriptorSetCount,pDescriptorSets);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, descriptorPool);
        for (uint32_t index=0;index<descriptorSetCount;index++) {
            finishWriteObject(my_data, pDescriptorSets[index]);
        }
        // Host access to descriptorPool must be externally synchronized
        // Host access to each member of pDescriptorSets must be externally synchronized
    } else {
        finishMultiThread();
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
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        for (uint32_t index=0;index<descriptorWriteCount;index++) {
                startWriteObject(my_data, pDescriptorWrites[index].dstSet);
        }
        for (uint32_t index=0;index<descriptorCopyCount;index++) {
                startWriteObject(my_data, pDescriptorCopies[index].dstSet);
        }
        // Host access to pDescriptorWrites[].dstSet must be externally synchronized
        // Host access to pDescriptorCopies[].dstSet must be externally synchronized
    }
    pTable->UpdateDescriptorSets(device,descriptorWriteCount,pDescriptorWrites,descriptorCopyCount,pDescriptorCopies);
    if (threadChecks) {
        finishReadObject(my_data, device);
        for (uint32_t index=0;index<descriptorWriteCount;index++) {
                finishWriteObject(my_data, pDescriptorWrites[index].dstSet);
        }
        for (uint32_t index=0;index<descriptorCopyCount;index++) {
                finishWriteObject(my_data, pDescriptorCopies[index].dstSet);
        }
        // Host access to pDescriptorWrites[].dstSet must be externally synchronized
        // Host access to pDescriptorCopies[].dstSet must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFramebuffer(
    VkDevice                                    device,
    const VkFramebufferCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFramebuffer*                              pFramebuffer)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateFramebuffer(device,pCreateInfo,pAllocator,pFramebuffer);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyFramebuffer(
    VkDevice                                    device,
    VkFramebuffer                               framebuffer,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, framebuffer);
        // Host access to framebuffer must be externally synchronized
    }
    pTable->DestroyFramebuffer(device,framebuffer,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, framebuffer);
        // Host access to framebuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass(
    VkDevice                                    device,
    const VkRenderPassCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateRenderPass(device,pCreateInfo,pAllocator,pRenderPass);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyRenderPass(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, renderPass);
        // Host access to renderPass must be externally synchronized
    }
    pTable->DestroyRenderPass(device,renderPass,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, renderPass);
        // Host access to renderPass must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL GetRenderAreaGranularity(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    VkExtent2D*                                 pGranularity)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, renderPass);
    }
    pTable->GetRenderAreaGranularity(device,renderPass,pGranularity);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, renderPass);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(
    VkDevice                                    device,
    const VkCommandPoolCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkCommandPool*                              pCommandPool)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateCommandPool(device,pCreateInfo,pAllocator,pCommandPool);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, commandPool);
        // Host access to commandPool must be externally synchronized
    }
    pTable->DestroyCommandPool(device,commandPool,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, commandPool);
        // Host access to commandPool must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolResetFlags                     flags)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, commandPool);
        // Host access to commandPool must be externally synchronized
    }
    result = pTable->ResetCommandPool(device,commandPool,flags);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, commandPool);
        // Host access to commandPool must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

// declare only
VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(
    VkDevice                                    device,
    const VkCommandBufferAllocateInfo*          pAllocateInfo,
    VkCommandBuffer*                            pCommandBuffers);

// declare only
VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers);

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    const VkCommandBufferBeginInfo*             pBeginInfo)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
        // the sname:VkCommandPool that pname:commandBuffer was allocated from must be externally synchronized between host accesses
    }
    result = pTable->BeginCommandBuffer(commandBuffer,pBeginInfo);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
        // the sname:VkCommandPool that pname:commandBuffer was allocated from must be externally synchronized between host accesses
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(
    VkCommandBuffer                             commandBuffer)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
        // the sname:VkCommandPool that pname:commandBuffer was allocated from must be externally synchronized between host accesses
    }
    result = pTable->EndCommandBuffer(commandBuffer);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
        // the sname:VkCommandPool that pname:commandBuffer was allocated from must be externally synchronized between host accesses
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    VkCommandBufferResetFlags                   flags)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    result = pTable->ResetCommandBuffer(commandBuffer,flags);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipeline                                  pipeline)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, pipeline);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBindPipeline(commandBuffer,pipelineBindPoint,pipeline);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, pipeline);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewport*                           pViewports)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetViewport(commandBuffer,firstViewport,viewportCount,pViewports);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstScissor,
    uint32_t                                    scissorCount,
    const VkRect2D*                             pScissors)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetScissor(commandBuffer,firstScissor,scissorCount,pScissors);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetLineWidth(
    VkCommandBuffer                             commandBuffer,
    float                                       lineWidth)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetLineWidth(commandBuffer,lineWidth);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBias(
    VkCommandBuffer                             commandBuffer,
    float                                       depthBiasConstantFactor,
    float                                       depthBiasClamp,
    float                                       depthBiasSlopeFactor)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetDepthBias(commandBuffer,depthBiasConstantFactor,depthBiasClamp,depthBiasSlopeFactor);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetBlendConstants(
    VkCommandBuffer                             commandBuffer,
    const float                                 blendConstants[4])
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetBlendConstants(commandBuffer,blendConstants);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBounds(
    VkCommandBuffer                             commandBuffer,
    float                                       minDepthBounds,
    float                                       maxDepthBounds)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetDepthBounds(commandBuffer,minDepthBounds,maxDepthBounds);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilCompareMask(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    compareMask)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetStencilCompareMask(commandBuffer,faceMask,compareMask);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilWriteMask(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    writeMask)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetStencilWriteMask(commandBuffer,faceMask,writeMask);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilReference(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    reference)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetStencilReference(commandBuffer,faceMask,reference);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, layout);
        for (uint32_t index = 0; index < descriptorSetCount; index++) {
            startReadObject(my_data, pDescriptorSets[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBindDescriptorSets(commandBuffer,pipelineBindPoint,layout,firstSet,descriptorSetCount,pDescriptorSets,dynamicOffsetCount,pDynamicOffsets);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, layout);
        for (uint32_t index = 0; index < descriptorSetCount; index++) {
            finishReadObject(my_data, pDescriptorSets[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkIndexType                                 indexType)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBindIndexBuffer(commandBuffer,buffer,offset,indexType);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        for (uint32_t index = 0; index < bindingCount; index++) {
            startReadObject(my_data, pBuffers[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBindVertexBuffers(commandBuffer,firstBinding,bindingCount,pBuffers,pOffsets);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        for (uint32_t index = 0; index < bindingCount; index++) {
            finishReadObject(my_data, pBuffers[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    vertexCount,
    uint32_t                                    instanceCount,
    uint32_t                                    firstVertex,
    uint32_t                                    firstInstance)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDraw(commandBuffer,vertexCount,instanceCount,firstVertex,firstInstance);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    indexCount,
    uint32_t                                    instanceCount,
    uint32_t                                    firstIndex,
    int32_t                                     vertexOffset,
    uint32_t                                    firstInstance)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDrawIndexed(commandBuffer,indexCount,instanceCount,firstIndex,vertexOffset,firstInstance);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDrawIndirect(commandBuffer,buffer,offset,drawCount,stride);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDrawIndexedIndirect(commandBuffer,buffer,offset,drawCount,stride);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    groupCountX,
    uint32_t                                    groupCountY,
    uint32_t                                    groupCountZ)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDispatch(commandBuffer,groupCountX,groupCountY,groupCountZ);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDispatchIndirect(commandBuffer,buffer,offset);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, buffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    srcBuffer,
    VkBuffer                                    dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferCopy*                         pRegions)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, srcBuffer);
        startReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdCopyBuffer(commandBuffer,srcBuffer,dstBuffer,regionCount,pRegions);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, srcBuffer);
        finishReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, srcImage);
        startReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdCopyImage(commandBuffer,srcImage,srcImageLayout,dstImage,dstImageLayout,regionCount,pRegions);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, srcImage);
        finishReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, srcImage);
        startReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBlitImage(commandBuffer,srcImage,srcImageLayout,dstImage,dstImageLayout,regionCount,pRegions,filter);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, srcImage);
        finishReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    srcBuffer,
    VkImage                                     dstImage,
    VkImageLayout                               dstImageLayout,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, srcBuffer);
        startReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdCopyBufferToImage(commandBuffer,srcBuffer,dstImage,dstImageLayout,regionCount,pRegions);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, srcBuffer);
        finishReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     srcImage,
    VkImageLayout                               srcImageLayout,
    VkBuffer                                    dstBuffer,
    uint32_t                                    regionCount,
    const VkBufferImageCopy*                    pRegions)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, srcImage);
        startReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdCopyImageToBuffer(commandBuffer,srcImage,srcImageLayout,dstBuffer,regionCount,pRegions);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, srcImage);
        finishReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                dataSize,
    const void*                                 pData)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdUpdateBuffer(commandBuffer,dstBuffer,dstOffset,dataSize,pData);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    VkDeviceSize                                size,
    uint32_t                                    data)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdFillBuffer(commandBuffer,dstBuffer,dstOffset,size,data);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearColorValue*                    pColor,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, image);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdClearColorImage(commandBuffer,image,imageLayout,pColor,rangeCount,pRanges);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, image);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(
    VkCommandBuffer                             commandBuffer,
    VkImage                                     image,
    VkImageLayout                               imageLayout,
    const VkClearDepthStencilValue*             pDepthStencil,
    uint32_t                                    rangeCount,
    const VkImageSubresourceRange*              pRanges)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, image);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdClearDepthStencilImage(commandBuffer,image,imageLayout,pDepthStencil,rangeCount,pRanges);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, image);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    attachmentCount,
    const VkClearAttachment*                    pAttachments,
    uint32_t                                    rectCount,
    const VkClearRect*                          pRects)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdClearAttachments(commandBuffer,attachmentCount,pAttachments,rectCount,pRects);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, srcImage);
        startReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdResolveImage(commandBuffer,srcImage,srcImageLayout,dstImage,dstImageLayout,regionCount,pRegions);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, srcImage);
        finishReadObject(my_data, dstImage);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, event);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetEvent(commandBuffer,event,stageMask);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, event);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdResetEvent(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     event,
    VkPipelineStageFlags                        stageMask)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, event);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdResetEvent(commandBuffer,event,stageMask);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, event);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        for (uint32_t index = 0; index < eventCount; index++) {
            startReadObject(my_data, pEvents[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdWaitEvents(commandBuffer,eventCount,pEvents,srcStageMask,dstStageMask,memoryBarrierCount,pMemoryBarriers,bufferMemoryBarrierCount,pBufferMemoryBarriers,imageMemoryBarrierCount,pImageMemoryBarriers);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        for (uint32_t index = 0; index < eventCount; index++) {
            finishReadObject(my_data, pEvents[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdPipelineBarrier(commandBuffer,srcStageMask,dstStageMask,dependencyFlags,memoryBarrierCount,pMemoryBarriers,bufferMemoryBarrierCount,pBufferMemoryBarriers,imageMemoryBarrierCount,pImageMemoryBarriers);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBeginQuery(commandBuffer,queryPool,query,flags);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdEndQuery(commandBuffer,queryPool,query);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdResetQueryPool(commandBuffer,queryPool,firstQuery,queryCount);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdWriteTimestamp(commandBuffer,pipelineStage,queryPool,query);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, queryPool);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, queryPool);
        startReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdCopyQueryPoolResults(commandBuffer,queryPool,firstQuery,queryCount,dstBuffer,dstOffset,stride,flags);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, queryPool);
        finishReadObject(my_data, dstBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(
    VkCommandBuffer                             commandBuffer,
    VkPipelineLayout                            layout,
    VkShaderStageFlags                          stageFlags,
    uint32_t                                    offset,
    uint32_t                                    size,
    const void*                                 pValues)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, layout);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdPushConstants(commandBuffer,layout,stageFlags,offset,size,pValues);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, layout);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(
    VkCommandBuffer                             commandBuffer,
    const VkRenderPassBeginInfo*                pRenderPassBegin,
    VkSubpassContents                           contents)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdBeginRenderPass(commandBuffer,pRenderPassBegin,contents);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass(
    VkCommandBuffer                             commandBuffer,
    VkSubpassContents                           contents)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdNextSubpass(commandBuffer,contents);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(
    VkCommandBuffer                             commandBuffer)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdEndRenderPass(commandBuffer);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        for (uint32_t index = 0; index < commandBufferCount; index++) {
            startReadObject(my_data, pCommandBuffers[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdExecuteCommands(commandBuffer,commandBufferCount,pCommandBuffers);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        for (uint32_t index = 0; index < commandBufferCount; index++) {
            finishReadObject(my_data, pCommandBuffers[index]);
        }
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}


VKAPI_ATTR void VKAPI_CALL DestroySurfaceKHR(
    VkInstance                                  instance,
    VkSurfaceKHR                                surface,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
        startWriteObject(my_data, surface);
        // Host access to surface must be externally synchronized
    }
    pTable->DestroySurfaceKHR(instance,surface,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, instance);
        finishWriteObject(my_data, surface);
        // Host access to surface must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    VkSurfaceKHR                                surface,
    VkBool32*                                   pSupported)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, surface);
    }
    result = pTable->GetPhysicalDeviceSurfaceSupportKHR(physicalDevice,queueFamilyIndex,surface,pSupported);
    if (threadChecks) {
        finishReadObject(my_data, surface);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    VkSurfaceCapabilitiesKHR*                   pSurfaceCapabilities)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, surface);
    }
    result = pTable->GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice,surface,pSurfaceCapabilities);
    if (threadChecks) {
        finishReadObject(my_data, surface);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pSurfaceFormatCount,
    VkSurfaceFormatKHR*                         pSurfaceFormats)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, surface);
    }
    result = pTable->GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice,surface,pSurfaceFormatCount,pSurfaceFormats);
    if (threadChecks) {
        finishReadObject(my_data, surface);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pPresentModeCount,
    VkPresentModeKHR*                           pPresentModes)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, surface);
    }
    result = pTable->GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice,surface,pPresentModeCount,pPresentModes);
    if (threadChecks) {
        finishReadObject(my_data, surface);
    } else {
        finishMultiThread();
    }
    return result;
}


VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(
    VkDevice                                    device,
    const VkSwapchainCreateInfoKHR*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSwapchainKHR*                             pSwapchain)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, pCreateInfo->surface);
        startWriteObject(my_data, pCreateInfo->oldSwapchain);
        // Host access to pCreateInfo.surface,pCreateInfo.oldSwapchain must be externally synchronized
    }
    result = pTable->CreateSwapchainKHR(device,pCreateInfo,pAllocator,pSwapchain);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, pCreateInfo->surface);
        finishWriteObject(my_data, pCreateInfo->oldSwapchain);
        // Host access to pCreateInfo.surface,pCreateInfo.oldSwapchain must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    }
    pTable->DestroySwapchainKHR(device,swapchain,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    } else {
        finishMultiThread();
    }
}

// declare only
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
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, swapchain);
        startWriteObject(my_data, semaphore);
        startWriteObject(my_data, fence);
        // Host access to swapchain must be externally synchronized
        // Host access to semaphore must be externally synchronized
        // Host access to fence must be externally synchronized
    }
    result = pTable->AcquireNextImageKHR(device,swapchain,timeout,semaphore,fence,pImageIndex);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, swapchain);
        finishWriteObject(my_data, semaphore);
        finishWriteObject(my_data, fence);
        // Host access to swapchain must be externally synchronized
        // Host access to semaphore must be externally synchronized
        // Host access to fence must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}
// TODO - not wrapping EXT function vkQueuePresentKHR


VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneSupportedDisplaysKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    planeIndex,
    uint32_t*                                   pDisplayCount,
    VkDisplayKHR*                               pDisplays)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        for (uint32_t index = 0; index < *pDisplayCount; index++) {
            startReadObject(my_data, pDisplays[index]);
        }
    }
    result = pTable->GetDisplayPlaneSupportedDisplaysKHR(physicalDevice,planeIndex,pDisplayCount,pDisplays);
    if (threadChecks) {
        for (uint32_t index = 0; index < *pDisplayCount; index++) {
            finishReadObject(my_data, pDisplays[index]);
        }
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayModePropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display,
    uint32_t*                                   pPropertyCount,
    VkDisplayModePropertiesKHR*                 pProperties)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, display);
    }
    result = pTable->GetDisplayModePropertiesKHR(physicalDevice,display,pPropertyCount,pProperties);
    if (threadChecks) {
        finishReadObject(my_data, display);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayModeKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display,
    const VkDisplayModeCreateInfoKHR*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDisplayModeKHR*                           pMode)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, display);
        // Host access to display must be externally synchronized
    }
    result = pTable->CreateDisplayModeKHR(physicalDevice,display,pCreateInfo,pAllocator,pMode);
    if (threadChecks) {
        finishWriteObject(my_data, display);
        // Host access to display must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDisplayPlaneCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayModeKHR                            mode,
    uint32_t                                    planeIndex,
    VkDisplayPlaneCapabilitiesKHR*              pCapabilities)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, mode);
        // Host access to mode must be externally synchronized
    }
    result = pTable->GetDisplayPlaneCapabilitiesKHR(physicalDevice,mode,planeIndex,pCapabilities);
    if (threadChecks) {
        finishWriteObject(my_data, mode);
        // Host access to mode must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDisplayPlaneSurfaceKHR(
    VkInstance                                  instance,
    const VkDisplaySurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateDisplayPlaneSurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
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
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        for (uint32_t index=0;index<swapchainCount;index++) {
                startWriteObject(my_data, pCreateInfos[index].surface);
                startWriteObject(my_data, pCreateInfos[index].oldSwapchain);
        }
        for (uint32_t index = 0; index < swapchainCount; index++) {
            startReadObject(my_data, pSwapchains[index]);
        }
        // Host access to pCreateInfos[].surface,pCreateInfos[].oldSwapchain must be externally synchronized
    }
    result = pTable->CreateSharedSwapchainsKHR(device,swapchainCount,pCreateInfos,pAllocator,pSwapchains);
    if (threadChecks) {
        finishReadObject(my_data, device);
        for (uint32_t index=0;index<swapchainCount;index++) {
                finishWriteObject(my_data, pCreateInfos[index].surface);
                finishWriteObject(my_data, pCreateInfos[index].oldSwapchain);
        }
        for (uint32_t index = 0; index < swapchainCount; index++) {
            finishReadObject(my_data, pSwapchains[index]);
        }
        // Host access to pCreateInfos[].surface,pCreateInfos[].oldSwapchain must be externally synchronized
    } else {
        finishMultiThread();
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
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateXlibSurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_XLIB_KHR */

#ifdef VK_USE_PLATFORM_XCB_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateXcbSurfaceKHR(
    VkInstance                                  instance,
    const VkXcbSurfaceCreateInfoKHR*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateXcbSurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_XCB_KHR */

#ifdef VK_USE_PLATFORM_WAYLAND_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateWaylandSurfaceKHR(
    VkInstance                                  instance,
    const VkWaylandSurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateWaylandSurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_WAYLAND_KHR */

#ifdef VK_USE_PLATFORM_MIR_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateMirSurfaceKHR(
    VkInstance                                  instance,
    const VkMirSurfaceCreateInfoKHR*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateMirSurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_MIR_KHR */

#ifdef VK_USE_PLATFORM_ANDROID_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateAndroidSurfaceKHR(
    VkInstance                                  instance,
    const VkAndroidSurfaceCreateInfoKHR*        pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateAndroidSurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_ANDROID_KHR */

#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL CreateWin32SurfaceKHR(
    VkInstance                                  instance,
    const VkWin32SurfaceCreateInfoKHR*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateWin32SurfaceKHR(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_WIN32_KHR */





VKAPI_ATTR void VKAPI_CALL TrimCommandPoolKHR(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolTrimFlagsKHR                   flags)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, commandPool);
        // Host access to commandPool must be externally synchronized
    }
    pTable->TrimCommandPoolKHR(device,commandPool,flags);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, commandPool);
        // Host access to commandPool must be externally synchronized
    } else {
        finishMultiThread();
    }
}



#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleKHR(
    VkDevice                                    device,
    const VkMemoryGetWin32HandleInfoKHR*        pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetMemoryWin32HandleKHR(device,pGetWin32HandleInfo,pHandle);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandlePropertiesKHR(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBitsKHR       handleType,
    HANDLE                                      handle,
    VkMemoryWin32HandlePropertiesKHR*           pMemoryWin32HandleProperties)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetMemoryWin32HandlePropertiesKHR(device,handleType,handle,pMemoryWin32HandleProperties);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_WIN32_KHR */


VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdKHR(
    VkDevice                                    device,
    const VkMemoryGetFdInfoKHR*                 pGetFdInfo,
    int*                                        pFd)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetMemoryFdKHR(device,pGetFdInfo,pFd);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdPropertiesKHR(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBitsKHR       handleType,
    int                                         fd,
    VkMemoryFdPropertiesKHR*                    pMemoryFdProperties)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetMemoryFdPropertiesKHR(device,handleType,fd,pMemoryFdProperties);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

#ifdef VK_USE_PLATFORM_WIN32_KHR
#endif /* VK_USE_PLATFORM_WIN32_KHR */



#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkImportSemaphoreWin32HandleInfoKHR*  pImportSemaphoreWin32HandleInfo)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->ImportSemaphoreWin32HandleKHR(device,pImportSemaphoreWin32HandleInfo);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkSemaphoreGetWin32HandleInfoKHR*     pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetSemaphoreWin32HandleKHR(device,pGetWin32HandleInfo,pHandle);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_WIN32_KHR */


VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreFdKHR(
    VkDevice                                    device,
    const VkImportSemaphoreFdInfoKHR*           pImportSemaphoreFdInfo)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->ImportSemaphoreFdKHR(device,pImportSemaphoreFdInfo);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreFdKHR(
    VkDevice                                    device,
    const VkSemaphoreGetFdInfoKHR*              pGetFdInfo,
    int*                                        pFd)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetSemaphoreFdKHR(device,pGetFdInfo,pFd);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, layout);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdPushDescriptorSetKHR(commandBuffer,pipelineBindPoint,layout,set,descriptorWriteCount,pDescriptorWrites);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, layout);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}




VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    const VkDescriptorUpdateTemplateCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorUpdateTemplateKHR*              pDescriptorUpdateTemplate)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateDescriptorUpdateTemplateKHR(device,pCreateInfo,pAllocator,pDescriptorUpdateTemplate);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, descriptorUpdateTemplate);
        // Host access to descriptorUpdateTemplate must be externally synchronized
    }
    pTable->DestroyDescriptorUpdateTemplateKHR(device,descriptorUpdateTemplate,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, descriptorUpdateTemplate);
        // Host access to descriptorUpdateTemplate must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSetWithTemplateKHR(
    VkDevice                                    device,
    VkDescriptorSet                             descriptorSet,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    const void*                                 pData)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, descriptorSet);
        startReadObject(my_data, descriptorUpdateTemplate);
        // Host access to descriptorSet must be externally synchronized
    }
    pTable->UpdateDescriptorSetWithTemplateKHR(device,descriptorSet,descriptorUpdateTemplate,pData);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, descriptorSet);
        finishReadObject(my_data, descriptorUpdateTemplate);
        // Host access to descriptorSet must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer                             commandBuffer,
    VkDescriptorUpdateTemplateKHR               descriptorUpdateTemplate,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    const void*                                 pData)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, descriptorUpdateTemplate);
        startReadObject(my_data, layout);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdPushDescriptorSetWithTemplateKHR(commandBuffer,descriptorUpdateTemplate,layout,set,pData);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, descriptorUpdateTemplate);
        finishReadObject(my_data, layout);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}


VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainStatusKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    }
    result = pTable->GetSwapchainStatusKHR(device,swapchain);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}



#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkImportFenceWin32HandleInfoKHR*      pImportFenceWin32HandleInfo)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->ImportFenceWin32HandleKHR(device,pImportFenceWin32HandleInfo);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkFenceGetWin32HandleInfoKHR*         pGetWin32HandleInfo,
    HANDLE*                                     pHandle)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetFenceWin32HandleKHR(device,pGetWin32HandleInfo,pHandle);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_WIN32_KHR */


VKAPI_ATTR VkResult VKAPI_CALL ImportFenceFdKHR(
    VkDevice                                    device,
    const VkImportFenceFdInfoKHR*               pImportFenceFdInfo)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->ImportFenceFdKHR(device,pImportFenceFdInfo);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceFdKHR(
    VkDevice                                    device,
    const VkFenceGetFdInfoKHR*                  pGetFdInfo,
    int*                                        pFd)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetFenceFdKHR(device,pGetFdInfo,pFd);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}







VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageMemoryRequirementsInfo2KHR*    pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    pTable->GetImageMemoryRequirements2KHR(device,pInfo,pMemoryRequirements);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkBufferMemoryRequirementsInfo2KHR*   pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    pTable->GetBufferMemoryRequirements2KHR(device,pInfo,pMemoryRequirements);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageSparseMemoryRequirementsInfo2KHR* pInfo,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements2KHR*        pSparseMemoryRequirements)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    pTable->GetImageSparseMemoryRequirements2KHR(device,pInfo,pSparseMemoryRequirementCount,pSparseMemoryRequirements);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
}


// declare only
VKAPI_ATTR VkResult VKAPI_CALL CreateDebugReportCallbackEXT(
    VkInstance                                  instance,
    const VkDebugReportCallbackCreateInfoEXT*   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDebugReportCallbackEXT*                   pCallback);

// declare only
VKAPI_ATTR void VKAPI_CALL DestroyDebugReportCallbackEXT(
    VkInstance                                  instance,
    VkDebugReportCallbackEXT                    callback,
    const VkAllocationCallbacks*                pAllocator);

VKAPI_ATTR void VKAPI_CALL DebugReportMessageEXT(
    VkInstance                                  instance,
    VkDebugReportFlagsEXT                       flags,
    VkDebugReportObjectTypeEXT                  objectType,
    uint64_t                                    object,
    size_t                                      location,
    int32_t                                     messageCode,
    const char*                                 pLayerPrefix,
    const char*                                 pMessage)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    pTable->DebugReportMessageEXT(instance,flags,objectType,object,location,messageCode,pLayerPrefix,pMessage);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
}







// TODO - not wrapping EXT function vkDebugMarkerSetObjectTagEXT
// TODO - not wrapping EXT function vkDebugMarkerSetObjectNameEXT
// TODO - not wrapping EXT function vkCmdDebugMarkerBeginEXT
// TODO - not wrapping EXT function vkCmdDebugMarkerEndEXT
// TODO - not wrapping EXT function vkCmdDebugMarkerInsertEXT




VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountAMD(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, buffer);
        startReadObject(my_data, countBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDrawIndirectCountAMD(commandBuffer,buffer,offset,countBuffer,countBufferOffset,maxDrawCount,stride);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, buffer);
        finishReadObject(my_data, countBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        startReadObject(my_data, buffer);
        startReadObject(my_data, countBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDrawIndexedIndirectCountAMD(commandBuffer,buffer,offset,countBuffer,countBufferOffset,maxDrawCount,stride);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        finishReadObject(my_data, buffer);
        finishReadObject(my_data, countBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}









#ifdef VK_USE_PLATFORM_WIN32_KHR

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleNV(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkExternalMemoryHandleTypeFlagsNV           handleType,
    HANDLE*                                     pHandle)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, memory);
    }
    result = pTable->GetMemoryWin32HandleNV(device,memory,handleType,pHandle);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, memory);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_WIN32_KHR */

#ifdef VK_USE_PLATFORM_WIN32_KHR
#endif /* VK_USE_PLATFORM_WIN32_KHR */


VKAPI_ATTR void VKAPI_CALL GetDeviceGroupPeerMemoryFeaturesKHX(
    VkDevice                                    device,
    uint32_t                                    heapIndex,
    uint32_t                                    localDeviceIndex,
    uint32_t                                    remoteDeviceIndex,
    VkPeerMemoryFeatureFlagsKHX*                pPeerMemoryFeatures)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    pTable->GetDeviceGroupPeerMemoryFeaturesKHX(device,heapIndex,localDeviceIndex,remoteDeviceIndex,pPeerMemoryFeatures);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory2KHX(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindBufferMemoryInfoKHX*            pBindInfos)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->BindBufferMemory2KHX(device,bindInfoCount,pBindInfos);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2KHX(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindImageMemoryInfoKHX*             pBindInfos)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->BindImageMemory2KHX(device,bindInfoCount,pBindInfos);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdSetDeviceMaskKHX(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    deviceMask)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetDeviceMaskKHX(commandBuffer,deviceMask);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupPresentCapabilitiesKHX(
    VkDevice                                    device,
    VkDeviceGroupPresentCapabilitiesKHX*        pDeviceGroupPresentCapabilities)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->GetDeviceGroupPresentCapabilitiesKHX(device,pDeviceGroupPresentCapabilities);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupSurfacePresentModesKHX(
    VkDevice                                    device,
    VkSurfaceKHR                                surface,
    VkDeviceGroupPresentModeFlagsKHX*           pModes)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, surface);
        // Host access to surface must be externally synchronized
    }
    result = pTable->GetDeviceGroupSurfacePresentModesKHX(device,surface,pModes);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, surface);
        // Host access to surface must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImage2KHX(
    VkDevice                                    device,
    const VkAcquireNextImageInfoKHX*            pAcquireInfo,
    uint32_t*                                   pImageIndex)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->AcquireNextImage2KHX(device,pAcquireInfo,pImageIndex);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
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
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdDispatchBaseKHX(commandBuffer,baseGroupX,baseGroupY,baseGroupZ,groupCountX,groupCountY,groupCountZ);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDevicePresentRectanglesKHX(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    uint32_t*                                   pRectCount,
    VkRect2D*                                   pRects)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, surface);
        // Host access to surface must be externally synchronized
    }
    result = pTable->GetPhysicalDevicePresentRectanglesKHX(physicalDevice,surface,pRectCount,pRects);
    if (threadChecks) {
        finishWriteObject(my_data, surface);
        // Host access to surface must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}


#ifdef VK_USE_PLATFORM_VI_NN

VKAPI_ATTR VkResult VKAPI_CALL CreateViSurfaceNN(
    VkInstance                                  instance,
    const VkViSurfaceCreateInfoNN*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateViSurfaceNN(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_VI_NN */




VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDeviceGroupsKHX(
    VkInstance                                  instance,
    uint32_t*                                   pPhysicalDeviceGroupCount,
    VkPhysicalDeviceGroupPropertiesKHX*         pPhysicalDeviceGroupProperties)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->EnumeratePhysicalDeviceGroupsKHX(instance,pPhysicalDeviceGroupCount,pPhysicalDeviceGroupProperties);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}


VKAPI_ATTR void VKAPI_CALL CmdProcessCommandsNVX(
    VkCommandBuffer                             commandBuffer,
    const VkCmdProcessCommandsInfoNVX*          pProcessCommandsInfo)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdProcessCommandsNVX(commandBuffer,pProcessCommandsInfo);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR void VKAPI_CALL CmdReserveSpaceForCommandsNVX(
    VkCommandBuffer                             commandBuffer,
    const VkCmdReserveSpaceForCommandsInfoNVX*  pReserveSpaceInfo)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdReserveSpaceForCommandsNVX(commandBuffer,pReserveSpaceInfo);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateIndirectCommandsLayoutNVX(
    VkDevice                                    device,
    const VkIndirectCommandsLayoutCreateInfoNVX* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkIndirectCommandsLayoutNVX*                pIndirectCommandsLayout)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateIndirectCommandsLayoutNVX(device,pCreateInfo,pAllocator,pIndirectCommandsLayout);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyIndirectCommandsLayoutNVX(
    VkDevice                                    device,
    VkIndirectCommandsLayoutNVX                 indirectCommandsLayout,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, indirectCommandsLayout);
    }
    pTable->DestroyIndirectCommandsLayoutNVX(device,indirectCommandsLayout,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, indirectCommandsLayout);
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateObjectTableNVX(
    VkDevice                                    device,
    const VkObjectTableCreateInfoNVX*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkObjectTableNVX*                           pObjectTable)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->CreateObjectTableNVX(device,pCreateInfo,pAllocator,pObjectTable);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyObjectTableNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    const VkAllocationCallbacks*                pAllocator)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, objectTable);
        // Host access to objectTable must be externally synchronized
    }
    pTable->DestroyObjectTableNVX(device,objectTable,pAllocator);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, objectTable);
        // Host access to objectTable must be externally synchronized
    } else {
        finishMultiThread();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterObjectsNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    uint32_t                                    objectCount,
    const VkObjectTableEntryNVX* const*         ppObjectTableEntries,
    const uint32_t*                             pObjectIndices)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, objectTable);
        // Host access to objectTable must be externally synchronized
    }
    result = pTable->RegisterObjectsNVX(device,objectTable,objectCount,ppObjectTableEntries,pObjectIndices);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, objectTable);
        // Host access to objectTable must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL UnregisterObjectsNVX(
    VkDevice                                    device,
    VkObjectTableNVX                            objectTable,
    uint32_t                                    objectCount,
    const VkObjectEntryTypeNVX*                 pObjectEntryTypes,
    const uint32_t*                             pObjectIndices)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, objectTable);
        // Host access to objectTable must be externally synchronized
    }
    result = pTable->UnregisterObjectsNVX(device,objectTable,objectCount,pObjectEntryTypes,pObjectIndices);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, objectTable);
        // Host access to objectTable must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}


VKAPI_ATTR void VKAPI_CALL CmdSetViewportWScalingNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewportWScalingNV*                 pViewportWScalings)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetViewportWScalingNV(commandBuffer,firstViewport,viewportCount,pViewportWScalings);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}


VKAPI_ATTR VkResult VKAPI_CALL ReleaseDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    VkDisplayKHR                                display)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, display);
    }
    result = pTable->ReleaseDisplayEXT(physicalDevice,display);
    if (threadChecks) {
        finishReadObject(my_data, display);
    } else {
        finishMultiThread();
    }
    return result;
}

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT

VKAPI_ATTR VkResult VKAPI_CALL AcquireXlibDisplayEXT(
    VkPhysicalDevice                            physicalDevice,
    Display*                                    dpy,
    VkDisplayKHR                                display)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, display);
    }
    result = pTable->AcquireXlibDisplayEXT(physicalDevice,dpy,display);
    if (threadChecks) {
        finishReadObject(my_data, display);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_XLIB_XRANDR_EXT */


VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2EXT(
    VkPhysicalDevice                            physicalDevice,
    VkSurfaceKHR                                surface,
    VkSurfaceCapabilities2EXT*                  pSurfaceCapabilities)
{
    dispatch_key key = get_dispatch_key(physicalDevice);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, surface);
    }
    result = pTable->GetPhysicalDeviceSurfaceCapabilities2EXT(physicalDevice,surface,pSurfaceCapabilities);
    if (threadChecks) {
        finishReadObject(my_data, surface);
    } else {
        finishMultiThread();
    }
    return result;
}


VKAPI_ATTR VkResult VKAPI_CALL DisplayPowerControlEXT(
    VkDevice                                    device,
    VkDisplayKHR                                display,
    const VkDisplayPowerInfoEXT*                pDisplayPowerInfo)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, display);
    }
    result = pTable->DisplayPowerControlEXT(device,display,pDisplayPowerInfo);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, display);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDeviceEventEXT(
    VkDevice                                    device,
    const VkDeviceEventInfoEXT*                 pDeviceEventInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
    }
    result = pTable->RegisterDeviceEventEXT(device,pDeviceEventInfo,pAllocator,pFence);
    if (threadChecks) {
        finishReadObject(my_data, device);
    } else {
        finishMultiThread();
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
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, display);
    }
    result = pTable->RegisterDisplayEventEXT(device,display,pDisplayEventInfo,pAllocator,pFence);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, display);
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainCounterEXT(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkSurfaceCounterFlagBitsEXT                 counter,
    uint64_t*                                   pCounterValue)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startReadObject(my_data, swapchain);
    }
    result = pTable->GetSwapchainCounterEXT(device,swapchain,counter,pCounterValue);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishReadObject(my_data, swapchain);
    } else {
        finishMultiThread();
    }
    return result;
}


VKAPI_ATTR VkResult VKAPI_CALL GetRefreshCycleDurationGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkRefreshCycleDurationGOOGLE*               pDisplayTimingProperties)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    }
    result = pTable->GetRefreshCycleDurationGOOGLE(device,swapchain,pDisplayTimingProperties);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPastPresentationTimingGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint32_t*                                   pPresentationTimingCount,
    VkPastPresentationTimingGOOGLE*             pPresentationTimings)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        startWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    }
    result = pTable->GetPastPresentationTimingGOOGLE(device,swapchain,pPresentationTimingCount,pPresentationTimings);
    if (threadChecks) {
        finishReadObject(my_data, device);
        finishWriteObject(my_data, swapchain);
        // Host access to swapchain must be externally synchronized
    } else {
        finishMultiThread();
    }
    return result;
}







VKAPI_ATTR void VKAPI_CALL CmdSetDiscardRectangleEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstDiscardRectangle,
    uint32_t                                    discardRectangleCount,
    const VkRect2D*                             pDiscardRectangles)
{
    dispatch_key key = get_dispatch_key(commandBuffer);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    }
    pTable->CmdSetDiscardRectangleEXT(commandBuffer,firstDiscardRectangle,discardRectangleCount,pDiscardRectangles);
    if (threadChecks) {
        finishWriteObject(my_data, commandBuffer);
        // Host access to commandBuffer must be externally synchronized
    } else {
        finishMultiThread();
    }
}



VKAPI_ATTR void VKAPI_CALL SetHdrMetadataEXT(
    VkDevice                                    device,
    uint32_t                                    swapchainCount,
    const VkSwapchainKHR*                       pSwapchains,
    const VkHdrMetadataEXT*                     pMetadata)
{
    dispatch_key key = get_dispatch_key(device);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerDispatchTable *pTable = my_data->device_dispatch_table;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, device);
        for (uint32_t index = 0; index < swapchainCount; index++) {
            startReadObject(my_data, pSwapchains[index]);
        }
    }
    pTable->SetHdrMetadataEXT(device,swapchainCount,pSwapchains,pMetadata);
    if (threadChecks) {
        finishReadObject(my_data, device);
        for (uint32_t index = 0; index < swapchainCount; index++) {
            finishReadObject(my_data, pSwapchains[index]);
        }
    } else {
        finishMultiThread();
    }
}

#ifdef VK_USE_PLATFORM_IOS_MVK

VKAPI_ATTR VkResult VKAPI_CALL CreateIOSSurfaceMVK(
    VkInstance                                  instance,
    const VkIOSSurfaceCreateInfoMVK*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateIOSSurfaceMVK(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_IOS_MVK */

#ifdef VK_USE_PLATFORM_MACOS_MVK

VKAPI_ATTR VkResult VKAPI_CALL CreateMacOSSurfaceMVK(
    VkInstance                                  instance,
    const VkMacOSSurfaceCreateInfoMVK*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface)
{
    dispatch_key key = get_dispatch_key(instance);
    layer_data *my_data = GetLayerDataPtr(key, layer_data_map);
    VkLayerInstanceDispatchTable *pTable = my_data->instance_dispatch_table;
    VkResult result;
    bool threadChecks = startMultiThread();
    if (threadChecks) {
        startReadObject(my_data, instance);
    }
    result = pTable->CreateMacOSSurfaceMVK(instance,pCreateInfo,pAllocator,pSurface);
    if (threadChecks) {
        finishReadObject(my_data, instance);
    } else {
        finishMultiThread();
    }
    return result;
}
#endif /* VK_USE_PLATFORM_MACOS_MVK */









// Map of all APIs to be intercepted by this layer
static const std::unordered_map<std::string, void*> name_to_funcptr_map = {
    {"vkCreateInstance", (void*)CreateInstance},
    {"vkDestroyInstance", (void*)DestroyInstance},
    {"vkEnumeratePhysicalDevices", (void*)EnumeratePhysicalDevices},
    {"vkGetInstanceProcAddr", (void*)GetInstanceProcAddr},
    {"vkGetDeviceProcAddr", (void*)GetDeviceProcAddr},
    {"vkCreateDevice", (void*)CreateDevice},
    {"vkDestroyDevice", (void*)DestroyDevice},
    {"vkEnumerateInstanceExtensionProperties", (void*)EnumerateInstanceExtensionProperties},
    {"vkEnumerateDeviceExtensionProperties", (void*)EnumerateDeviceExtensionProperties},
    {"vkEnumerateInstanceLayerProperties", (void*)EnumerateInstanceLayerProperties},
    {"vkEnumerateDeviceLayerProperties", (void*)EnumerateDeviceLayerProperties},
    {"vkGetDeviceQueue", (void*)GetDeviceQueue},
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
    {"vkCreateComputePipelines", (void*)CreateComputePipelines},
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
    {"vkDestroySwapchainKHR", (void*)DestroySwapchainKHR},
    {"vkGetSwapchainImagesKHR", (void*)GetSwapchainImagesKHR},
    {"vkAcquireNextImageKHR", (void*)AcquireNextImageKHR},
    {"vkGetDisplayPlaneSupportedDisplaysKHR", (void*)GetDisplayPlaneSupportedDisplaysKHR},
    {"vkGetDisplayModePropertiesKHR", (void*)GetDisplayModePropertiesKHR},
    {"vkCreateDisplayModeKHR", (void*)CreateDisplayModeKHR},
    {"vkGetDisplayPlaneCapabilitiesKHR", (void*)GetDisplayPlaneCapabilitiesKHR},
    {"vkCreateDisplayPlaneSurfaceKHR", (void*)CreateDisplayPlaneSurfaceKHR},
    {"vkCreateSharedSwapchainsKHR", (void*)CreateSharedSwapchainsKHR},
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
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetMemoryWin32HandlePropertiesKHR", (void*)GetMemoryWin32HandlePropertiesKHR},
#endif
    {"vkGetMemoryFdKHR", (void*)GetMemoryFdKHR},
    {"vkGetMemoryFdPropertiesKHR", (void*)GetMemoryFdPropertiesKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkImportSemaphoreWin32HandleKHR", (void*)ImportSemaphoreWin32HandleKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetSemaphoreWin32HandleKHR", (void*)GetSemaphoreWin32HandleKHR},
#endif
    {"vkImportSemaphoreFdKHR", (void*)ImportSemaphoreFdKHR},
    {"vkGetSemaphoreFdKHR", (void*)GetSemaphoreFdKHR},
    {"vkCmdPushDescriptorSetKHR", (void*)CmdPushDescriptorSetKHR},
    {"vkCreateDescriptorUpdateTemplateKHR", (void*)CreateDescriptorUpdateTemplateKHR},
    {"vkDestroyDescriptorUpdateTemplateKHR", (void*)DestroyDescriptorUpdateTemplateKHR},
    {"vkUpdateDescriptorSetWithTemplateKHR", (void*)UpdateDescriptorSetWithTemplateKHR},
    {"vkCmdPushDescriptorSetWithTemplateKHR", (void*)CmdPushDescriptorSetWithTemplateKHR},
    {"vkGetSwapchainStatusKHR", (void*)GetSwapchainStatusKHR},
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkImportFenceWin32HandleKHR", (void*)ImportFenceWin32HandleKHR},
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    {"vkGetFenceWin32HandleKHR", (void*)GetFenceWin32HandleKHR},
#endif
    {"vkImportFenceFdKHR", (void*)ImportFenceFdKHR},
    {"vkGetFenceFdKHR", (void*)GetFenceFdKHR},
    {"vkGetImageMemoryRequirements2KHR", (void*)GetImageMemoryRequirements2KHR},
    {"vkGetBufferMemoryRequirements2KHR", (void*)GetBufferMemoryRequirements2KHR},
    {"vkGetImageSparseMemoryRequirements2KHR", (void*)GetImageSparseMemoryRequirements2KHR},
    {"vkCreateDebugReportCallbackEXT", (void*)CreateDebugReportCallbackEXT},
    {"vkDestroyDebugReportCallbackEXT", (void*)DestroyDebugReportCallbackEXT},
    {"vkDebugReportMessageEXT", (void*)DebugReportMessageEXT},
    {"vkCmdDrawIndirectCountAMD", (void*)CmdDrawIndirectCountAMD},
    {"vkCmdDrawIndexedIndirectCountAMD", (void*)CmdDrawIndexedIndirectCountAMD},
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
    {"vkCmdSetViewportWScalingNV", (void*)CmdSetViewportWScalingNV},
    {"vkReleaseDisplayEXT", (void*)ReleaseDisplayEXT},
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    {"vkAcquireXlibDisplayEXT", (void*)AcquireXlibDisplayEXT},
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


} // namespace threading

#endif

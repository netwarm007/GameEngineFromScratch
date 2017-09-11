/* Copyright (c) 2015-2017 The Khronos Group Inc.
 * Copyright (c) 2015-2017 Valve Corporation
 * Copyright (c) 2015-2017 LunarG, Inc.
 * Copyright (C) 2015-2017 Google Inc.
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
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Tobin Ehlis <tobin@lunarg.com>
 */

#include "object_tracker.h"

namespace object_tracker {

std::unordered_map<void *, layer_data *> layer_data_map;
device_table_map ot_device_table_map;
instance_table_map ot_instance_table_map;
std::mutex global_lock;
uint64_t object_track_index = 0;
uint32_t loader_layer_if_version = CURRENT_LOADER_LAYER_INTERFACE_VERSION;

void InitObjectTracker(layer_data *my_data, const VkAllocationCallbacks *pAllocator) {
    layer_debug_actions(my_data->report_data, my_data->logging_callback, pAllocator, "lunarg_object_tracker");
}

// Add new queue to head of global queue list
void AddQueueInfo(VkDevice device, uint32_t queue_node_index, VkQueue queue) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    auto queueItem = device_data->queue_info_map.find(queue);
    if (queueItem == device_data->queue_info_map.end()) {
        ObjTrackQueueInfo *p_queue_info = new ObjTrackQueueInfo;
        if (p_queue_info != NULL) {
            memset(p_queue_info, 0, sizeof(ObjTrackQueueInfo));
            p_queue_info->queue = queue;
            p_queue_info->queue_node_index = queue_node_index;
            device_data->queue_info_map[queue] = p_queue_info;
        } else {
            log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,
                    HandleToUint64(queue), __LINE__, OBJTRACK_INTERNAL_ERROR, LayerName,
                    "ERROR:  VK_ERROR_OUT_OF_HOST_MEMORY -- could not allocate memory for Queue Information");
        }
    }
}

// Destroy memRef lists and free all memory
void DestroyQueueDataStructures(VkDevice device) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    for (auto queue_item : device_data->queue_info_map) {
        delete queue_item.second;
    }
    device_data->queue_info_map.clear();

    // Destroy the items in the queue map
    auto queue = device_data->object_map[kVulkanObjectTypeQueue].begin();
    while (queue != device_data->object_map[kVulkanObjectTypeQueue].end()) {
        uint32_t obj_index = queue->second->object_type;
        assert(device_data->num_total_objects > 0);
        device_data->num_total_objects--;
        assert(device_data->num_objects[obj_index] > 0);
        device_data->num_objects[obj_index]--;
        log_msg(device_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,
                queue->second->handle, __LINE__, OBJTRACK_NONE, LayerName,
                "OBJ_STAT Destroy Queue obj 0x%" PRIxLEAST64 " (%" PRIu64 " total objs remain & %" PRIu64 " Queue objs).",
                queue->second->handle, device_data->num_total_objects, device_data->num_objects[obj_index]);
        delete queue->second;
        queue = device_data->object_map[kVulkanObjectTypeQueue].erase(queue);
    }
}

// Check Queue type flags for selected queue operations
void ValidateQueueFlags(VkQueue queue, const char *function) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(queue), layer_data_map);
    auto queue_item = device_data->queue_info_map.find(queue);
    if (queue_item != device_data->queue_info_map.end()) {
        ObjTrackQueueInfo *pQueueInfo = queue_item->second;
        if (pQueueInfo != NULL) {
            layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(device_data->physical_device), layer_data_map);
            if ((instance_data->queue_family_properties[pQueueInfo->queue_node_index].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) ==
                0) {
                log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,
                        HandleToUint64(queue), __LINE__, VALIDATION_ERROR_31600011, LayerName,
                        "Attempting %s on a non-memory-management capable queue -- VK_QUEUE_SPARSE_BINDING_BIT not set. %s",
                        function, validation_error_map[VALIDATION_ERROR_31600011]);
            }
        }
    }
}

void AllocateCommandBuffer(VkDevice device, const VkCommandPool command_pool, const VkCommandBuffer command_buffer,
                           VkCommandBufferLevel level) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    log_msg(device_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
            HandleToUint64(command_buffer), __LINE__, OBJTRACK_NONE, LayerName,
            "OBJ[0x%" PRIxLEAST64 "] : CREATE %s object 0x%" PRIxLEAST64, object_track_index++,
            "VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT", HandleToUint64(command_buffer));

    ObjTrackState *pNewObjNode = new ObjTrackState;
    pNewObjNode->object_type = kVulkanObjectTypeCommandBuffer;
    pNewObjNode->handle = HandleToUint64(command_buffer);
    pNewObjNode->parent_object = HandleToUint64(command_pool);
    if (level == VK_COMMAND_BUFFER_LEVEL_SECONDARY) {
        pNewObjNode->status = OBJSTATUS_COMMAND_BUFFER_SECONDARY;
    } else {
        pNewObjNode->status = OBJSTATUS_NONE;
    }
    device_data->object_map[kVulkanObjectTypeCommandBuffer][HandleToUint64(command_buffer)] = pNewObjNode;
    device_data->num_objects[kVulkanObjectTypeCommandBuffer]++;
    device_data->num_total_objects++;
}

bool ValidateCommandBuffer(VkDevice device, VkCommandPool command_pool, VkCommandBuffer command_buffer) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    bool skip = false;
    uint64_t object_handle = HandleToUint64(command_buffer);
    if (device_data->object_map[kVulkanObjectTypeCommandBuffer].find(object_handle) !=
        device_data->object_map[kVulkanObjectTypeCommandBuffer].end()) {
        ObjTrackState *pNode = device_data->object_map[kVulkanObjectTypeCommandBuffer][HandleToUint64(command_buffer)];

        if (pNode->parent_object != HandleToUint64(command_pool)) {
            skip |= log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            object_handle, __LINE__, VALIDATION_ERROR_28411407, LayerName,
                            "FreeCommandBuffers is attempting to free Command Buffer 0x%" PRIxLEAST64
                            " belonging to Command Pool 0x%" PRIxLEAST64 " from pool 0x%" PRIxLEAST64 "). %s",
                            HandleToUint64(command_buffer), pNode->parent_object, HandleToUint64(command_pool),
                            validation_error_map[VALIDATION_ERROR_28411407]);
        }
    } else {
        skip |=
            log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    object_handle, __LINE__, VALIDATION_ERROR_28400060, LayerName, "Invalid %s Object 0x%" PRIxLEAST64 ". %s",
                    object_string[kVulkanObjectTypeCommandBuffer], object_handle, validation_error_map[VALIDATION_ERROR_28400060]);
    }
    return skip;
}

void AllocateDescriptorSet(VkDevice device, VkDescriptorPool descriptor_pool, VkDescriptorSet descriptor_set) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    log_msg(device_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
            HandleToUint64(descriptor_set), __LINE__, OBJTRACK_NONE, LayerName,
            "OBJ[0x%" PRIxLEAST64 "] : CREATE %s object 0x%" PRIxLEAST64, object_track_index++,
            "VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT", HandleToUint64(descriptor_set));

    ObjTrackState *pNewObjNode = new ObjTrackState;
    pNewObjNode->object_type = kVulkanObjectTypeDescriptorSet;
    pNewObjNode->status = OBJSTATUS_NONE;
    pNewObjNode->handle = HandleToUint64(descriptor_set);
    pNewObjNode->parent_object = HandleToUint64(descriptor_pool);
    device_data->object_map[kVulkanObjectTypeDescriptorSet][HandleToUint64(descriptor_set)] = pNewObjNode;
    device_data->num_objects[kVulkanObjectTypeDescriptorSet]++;
    device_data->num_total_objects++;
}

bool ValidateDescriptorSet(VkDevice device, VkDescriptorPool descriptor_pool, VkDescriptorSet descriptor_set) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    bool skip = false;
    uint64_t object_handle = HandleToUint64(descriptor_set);
    auto dsItem = device_data->object_map[kVulkanObjectTypeDescriptorSet].find(object_handle);
    if (dsItem != device_data->object_map[kVulkanObjectTypeDescriptorSet].end()) {
        ObjTrackState *pNode = dsItem->second;

        if (pNode->parent_object != HandleToUint64(descriptor_pool)) {
            skip |= log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
                            object_handle, __LINE__, VALIDATION_ERROR_28613007, LayerName,
                            "FreeDescriptorSets is attempting to free descriptorSet 0x%" PRIxLEAST64
                            " belonging to Descriptor Pool 0x%" PRIxLEAST64 " from pool 0x%" PRIxLEAST64 "). %s",
                            HandleToUint64(descriptor_set), pNode->parent_object, HandleToUint64(descriptor_pool),
                            validation_error_map[VALIDATION_ERROR_28613007]);
        }
    } else {
        skip |=
            log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
                    object_handle, __LINE__, VALIDATION_ERROR_2860026c, LayerName, "Invalid %s Object 0x%" PRIxLEAST64 ". %s",
                    object_string[kVulkanObjectTypeDescriptorSet], object_handle, validation_error_map[VALIDATION_ERROR_2860026c]);
    }
    return skip;
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                   VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                                   const VkWriteDescriptorSet *pDescriptorWrites) {
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(commandBuffer, commandBuffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_1be02401,
                               VALIDATION_ERROR_1be00009);
        skip |= ValidateObject(commandBuffer, layout, kVulkanObjectTypePipelineLayout, false, VALIDATION_ERROR_1be0be01,
                               VALIDATION_ERROR_1be00009);
        if (pDescriptorWrites) {
            for (uint32_t index0 = 0; index0 < descriptorWriteCount; ++index0) {
                if (pDescriptorWrites[index0].pImageInfo) {
                    for (uint32_t index1 = 0; index1 < pDescriptorWrites[index0].descriptorCount; ++index1) {
                        skip |=
                            ValidateObject(commandBuffer, pDescriptorWrites[index0].pImageInfo[index1].sampler,
                                           kVulkanObjectTypeSampler, false, VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_04600009);
                        skip |= ValidateObject(commandBuffer, pDescriptorWrites[index0].pImageInfo[index1].imageView,
                                               kVulkanObjectTypeImageView, false, VALIDATION_ERROR_UNDEFINED,
                                               VALIDATION_ERROR_04600009);
                    }
                }
                if (pDescriptorWrites[index0].pBufferInfo) {
                    for (uint32_t index1 = 0; index1 < pDescriptorWrites[index0].descriptorCount; ++index1) {
                        skip |=
                            ValidateObject(commandBuffer, pDescriptorWrites[index0].pBufferInfo[index1].buffer,
                                           kVulkanObjectTypeBuffer, false, VALIDATION_ERROR_04401a01, VALIDATION_ERROR_UNDEFINED);
                    }
                }
                if (pDescriptorWrites[index0].pTexelBufferView) {
                    for (uint32_t index1 = 0; index1 < pDescriptorWrites[index0].descriptorCount; ++index1) {
                        skip |= ValidateObject(commandBuffer, pDescriptorWrites[index0].pTexelBufferView[index1],
                                               kVulkanObjectTypeBufferView, false, VALIDATION_ERROR_UNDEFINED,
                                               VALIDATION_ERROR_15c00009);
                    }
                }
            }
        }
    }
    if (skip) return;
    get_dispatch_table(ot_device_table_map, commandBuffer)
        ->CmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
}

void CreateQueue(VkDevice device, VkQueue vkObj) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);

    log_msg(device_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,
            HandleToUint64(vkObj), __LINE__, OBJTRACK_NONE, LayerName, "OBJ[0x%" PRIxLEAST64 "] : CREATE %s object 0x%" PRIxLEAST64,
            object_track_index++, "VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT", HandleToUint64(vkObj));

    ObjTrackState *p_obj_node = NULL;
    auto queue_item = device_data->object_map[kVulkanObjectTypeQueue].find(HandleToUint64(vkObj));
    if (queue_item == device_data->object_map[kVulkanObjectTypeQueue].end()) {
        p_obj_node = new ObjTrackState;
        device_data->object_map[kVulkanObjectTypeQueue][HandleToUint64(vkObj)] = p_obj_node;
        device_data->num_objects[kVulkanObjectTypeQueue]++;
        device_data->num_total_objects++;
    } else {
        p_obj_node = queue_item->second;
    }
    p_obj_node->object_type = kVulkanObjectTypeQueue;
    p_obj_node->status = OBJSTATUS_NONE;
    p_obj_node->handle = HandleToUint64(vkObj);
}

void CreateSwapchainImageObject(VkDevice dispatchable_object, VkImage swapchain_image, VkSwapchainKHR swapchain) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(dispatchable_object), layer_data_map);
    log_msg(device_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
            HandleToUint64(swapchain_image), __LINE__, OBJTRACK_NONE, LayerName,
            "OBJ[0x%" PRIxLEAST64 "] : CREATE %s object 0x%" PRIxLEAST64, object_track_index++, "SwapchainImage",
            HandleToUint64(swapchain_image));

    ObjTrackState *pNewObjNode = new ObjTrackState;
    pNewObjNode->object_type = kVulkanObjectTypeImage;
    pNewObjNode->status = OBJSTATUS_NONE;
    pNewObjNode->handle = HandleToUint64(swapchain_image);
    pNewObjNode->parent_object = HandleToUint64(swapchain);
    device_data->swapchainImageMap[HandleToUint64(swapchain_image)] = pNewObjNode;
}

void DeviceReportUndestroyedObjects(VkDevice device, VulkanObjectType object_type, enum UNIQUE_VALIDATION_ERROR_CODE error_code) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    for (auto item = device_data->object_map[object_type].begin(); item != device_data->object_map[object_type].end();) {
        ObjTrackState *object_info = item->second;
        log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, get_debug_report_enum[object_type], object_info->handle,
                __LINE__, error_code, LayerName,
                "OBJ ERROR : For device 0x%" PRIxLEAST64 ", %s object 0x%" PRIxLEAST64 " has not been destroyed. %s",
                HandleToUint64(device), object_string[object_type], object_info->handle, validation_error_map[error_code]);
        item = device_data->object_map[object_type].erase(item);
    }
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    std::unique_lock<std::mutex> lock(global_lock);

    dispatch_key key = get_dispatch_key(instance);
    layer_data *instance_data = GetLayerDataPtr(key, layer_data_map);

    // Enable the temporary callback(s) here to catch cleanup issues:
    bool callback_setup = false;
    if (instance_data->num_tmp_callbacks > 0) {
        if (!layer_enable_tmp_callbacks(instance_data->report_data, instance_data->num_tmp_callbacks,
                                        instance_data->tmp_dbg_create_infos, instance_data->tmp_callbacks)) {
            callback_setup = true;
        }
    }

    // TODO: The instance handle can not be validated here. The loader will likely have to validate it.
    ValidateObject(instance, instance, kVulkanObjectTypeInstance, true, VALIDATION_ERROR_2580bc01, VALIDATION_ERROR_UNDEFINED);

    // Destroy physical devices
    for (auto iit = instance_data->object_map[kVulkanObjectTypePhysicalDevice].begin();
         iit != instance_data->object_map[kVulkanObjectTypePhysicalDevice].end();) {
        ObjTrackState *pNode = iit->second;

        VkPhysicalDevice physical_device = reinterpret_cast<VkPhysicalDevice>(pNode->handle);
        DestroyObject(instance, physical_device, kVulkanObjectTypePhysicalDevice, nullptr, VALIDATION_ERROR_UNDEFINED,
                      VALIDATION_ERROR_UNDEFINED);
        iit = instance_data->object_map[kVulkanObjectTypePhysicalDevice].begin();
    }

    DestroyObject(instance, instance, kVulkanObjectTypeInstance, pAllocator, VALIDATION_ERROR_258004ec, VALIDATION_ERROR_258004ee);
    // Report any remaining objects in LL

    for (auto iit = instance_data->object_map[kVulkanObjectTypeDevice].begin();
         iit != instance_data->object_map[kVulkanObjectTypeDevice].end();) {
        ObjTrackState *pNode = iit->second;

        VkDevice device = reinterpret_cast<VkDevice>(pNode->handle);
        VkDebugReportObjectTypeEXT debug_object_type = get_debug_report_enum[pNode->object_type];

        log_msg(instance_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, debug_object_type, pNode->handle, __LINE__,
                OBJTRACK_OBJECT_LEAK, LayerName, "OBJ ERROR : %s object 0x%" PRIxLEAST64 " has not been destroyed.",
                string_VkDebugReportObjectTypeEXT(debug_object_type), pNode->handle);

        ReportUndestroyedObjects(device, VALIDATION_ERROR_258004ea);
    }
    instance_data->object_map[kVulkanObjectTypeDevice].clear();

    VkLayerInstanceDispatchTable *pInstanceTable = get_dispatch_table(ot_instance_table_map, instance);
    pInstanceTable->DestroyInstance(instance, pAllocator);

    // Disable and cleanup the temporary callback(s):
    if (callback_setup) {
        layer_disable_tmp_callbacks(instance_data->report_data, instance_data->num_tmp_callbacks, instance_data->tmp_callbacks);
    }
    if (instance_data->num_tmp_callbacks > 0) {
        layer_free_tmp_callbacks(instance_data->tmp_dbg_create_infos, instance_data->tmp_callbacks);
        instance_data->num_tmp_callbacks = 0;
    }

    // Clean up logging callback, if any
    while (instance_data->logging_callback.size() > 0) {
        VkDebugReportCallbackEXT callback = instance_data->logging_callback.back();
        layer_destroy_msg_callback(instance_data->report_data, callback, pAllocator);
        instance_data->logging_callback.pop_back();
    }

    layer_debug_report_destroy_instance(instance_data->report_data);
    FreeLayerDataPtr(key, layer_data_map);

    lock.unlock();
    ot_instance_table_map.erase(key);
    delete pInstanceTable;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    std::unique_lock<std::mutex> lock(global_lock);
    ValidateObject(device, device, kVulkanObjectTypeDevice, true, VALIDATION_ERROR_24a05601, VALIDATION_ERROR_UNDEFINED);
    DestroyObject(device, device, kVulkanObjectTypeDevice, pAllocator, VALIDATION_ERROR_24a002f6, VALIDATION_ERROR_24a002f8);

    // Report any remaining objects associated with this VkDevice object in LL
    ReportUndestroyedObjects(device, VALIDATION_ERROR_24a002f4);

    // Clean up Queue's MemRef Linked Lists
    DestroyQueueDataStructures(device);

    lock.unlock();

    dispatch_key key = get_dispatch_key(device);
    VkLayerDispatchTable *pDisp = get_dispatch_table(ot_device_table_map, device);
    pDisp->DestroyDevice(device, pAllocator);
    ot_device_table_map.erase(key);
    delete pDisp;

    FreeLayerDataPtr(key, layer_data_map);
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue *pQueue) {
    std::unique_lock<std::mutex> lock(global_lock);
    ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_29605601, VALIDATION_ERROR_UNDEFINED);
    lock.unlock();

    get_dispatch_table(ot_device_table_map, device)->GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);

    lock.lock();
    CreateQueue(device, *pQueue);
    AddQueueInfo(device, queueFamilyIndex, *pQueue);
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                const VkWriteDescriptorSet *pDescriptorWrites, uint32_t descriptorCopyCount,
                                                const VkCopyDescriptorSet *pDescriptorCopies) {
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |=
            ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_33c05601, VALIDATION_ERROR_UNDEFINED);
        if (pDescriptorCopies) {
            for (uint32_t idx0 = 0; idx0 < descriptorCopyCount; ++idx0) {
                if (pDescriptorCopies[idx0].dstSet) {
                    skip |= ValidateObject(device, pDescriptorCopies[idx0].dstSet, kVulkanObjectTypeDescriptorSet, false,
                                           VALIDATION_ERROR_03207601, VALIDATION_ERROR_03200009);
                }
                if (pDescriptorCopies[idx0].srcSet) {
                    skip |= ValidateObject(device, pDescriptorCopies[idx0].srcSet, kVulkanObjectTypeDescriptorSet, false,
                                           VALIDATION_ERROR_0322d201, VALIDATION_ERROR_03200009);
                }
            }
        }
        if (pDescriptorWrites) {
            for (uint32_t idx1 = 0; idx1 < descriptorWriteCount; ++idx1) {
                if (pDescriptorWrites[idx1].dstSet) {
                    skip |= ValidateObject(device, pDescriptorWrites[idx1].dstSet, kVulkanObjectTypeDescriptorSet, false,
                                           VALIDATION_ERROR_15c00280, VALIDATION_ERROR_15c00009);
                }
                if ((pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)) {
                    for (uint32_t idx2 = 0; idx2 < pDescriptorWrites[idx1].descriptorCount; ++idx2) {
                        skip |= ValidateObject(device, pDescriptorWrites[idx1].pTexelBufferView[idx2], kVulkanObjectTypeBufferView,
                                               false, VALIDATION_ERROR_15c00286, VALIDATION_ERROR_15c00009);
                    }
                }
                if ((pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT)) {
                    for (uint32_t idx3 = 0; idx3 < pDescriptorWrites[idx1].descriptorCount; ++idx3) {
                        skip |=
                            ValidateObject(device, pDescriptorWrites[idx1].pImageInfo[idx3].imageView, kVulkanObjectTypeImageView,
                                           false, VALIDATION_ERROR_15c0028c, VALIDATION_ERROR_04600009);
                    }
                }
                if ((pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
                    (pDescriptorWrites[idx1].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)) {
                    for (uint32_t idx4 = 0; idx4 < pDescriptorWrites[idx1].descriptorCount; ++idx4) {
                        if (pDescriptorWrites[idx1].pBufferInfo[idx4].buffer) {
                            skip |=
                                ValidateObject(device, pDescriptorWrites[idx1].pBufferInfo[idx4].buffer, kVulkanObjectTypeBuffer,
                                               false, VALIDATION_ERROR_04401a01, VALIDATION_ERROR_UNDEFINED);
                        }
                    }
                }
            }
        }
    }
    if (skip) {
        return;
    }
    get_dispatch_table(ot_device_table_map, device)
        ->UpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines) {
    bool skip = VK_FALSE;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_1f205601, VALIDATION_ERROR_UNDEFINED);
    if (pCreateInfos) {
        for (uint32_t idx0 = 0; idx0 < createInfoCount; ++idx0) {
            if (pCreateInfos[idx0].basePipelineHandle) {
                skip |= ValidateObject(device, pCreateInfos[idx0].basePipelineHandle, kVulkanObjectTypePipeline, true,
                                       VALIDATION_ERROR_03000572, VALIDATION_ERROR_03000009);
            }
            if (pCreateInfos[idx0].layout) {
                skip |= ValidateObject(device, pCreateInfos[idx0].layout, kVulkanObjectTypePipelineLayout, false,
                                       VALIDATION_ERROR_0300be01, VALIDATION_ERROR_03000009);
            }
            if (pCreateInfos[idx0].stage.module) {
                skip |= ValidateObject(device, pCreateInfos[idx0].stage.module, kVulkanObjectTypeShaderModule, false,
                                       VALIDATION_ERROR_1060d201, VALIDATION_ERROR_UNDEFINED);
            }
        }
    }
    if (pipelineCache) {
        skip |= ValidateObject(device, pipelineCache, kVulkanObjectTypePipelineCache, true, VALIDATION_ERROR_1f228001,
                               VALIDATION_ERROR_1f228007);
    }
    lock.unlock();
    if (skip) {
        for (uint32_t i = 0; i < createInfoCount; i++) {
            pPipelines[i] = VK_NULL_HANDLE;
        }
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    VkResult result = get_dispatch_table(ot_device_table_map, device)
                          ->CreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
    lock.lock();
    for (uint32_t idx1 = 0; idx1 < createInfoCount; ++idx1) {
        if (pPipelines[idx1] != VK_NULL_HANDLE) {
            CreateObject(device, pPipelines[idx1], kVulkanObjectTypePipeline, pAllocator);
        }
    }
    lock.unlock();
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                   VkDescriptorPoolResetFlags flags) {
    bool skip = false;
    std::unique_lock<std::mutex> lock(global_lock);
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_32a05601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, descriptorPool, kVulkanObjectTypeDescriptorPool, false, VALIDATION_ERROR_32a04601,
                           VALIDATION_ERROR_32a04607);
    if (skip) {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    // A DescriptorPool's descriptor sets are implicitly deleted when the pool is reset.
    // Remove this pool's descriptor sets from our descriptorSet map.
    auto itr = device_data->object_map[kVulkanObjectTypeDescriptorSet].begin();
    while (itr != device_data->object_map[kVulkanObjectTypeDescriptorSet].end()) {
        ObjTrackState *pNode = (*itr).second;
        auto del_itr = itr++;
        if (pNode->parent_object == HandleToUint64(descriptorPool)) {
            DestroyObject(device, (VkDescriptorSet)((*del_itr).first), kVulkanObjectTypeDescriptorSet, nullptr,
                          VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
        }
    }
    lock.unlock();
    VkResult result = get_dispatch_table(ot_device_table_map, device)->ResetDescriptorPool(device, descriptorPool, flags);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer command_buffer, const VkCommandBufferBeginInfo *begin_info) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(command_buffer), layer_data_map);
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(command_buffer, command_buffer, kVulkanObjectTypeCommandBuffer, false, VALIDATION_ERROR_16e02401,
                               VALIDATION_ERROR_UNDEFINED);
        if (begin_info) {
            ObjTrackState *pNode = device_data->object_map[kVulkanObjectTypeCommandBuffer][HandleToUint64(command_buffer)];
            if ((begin_info->pInheritanceInfo) && (pNode->status & OBJSTATUS_COMMAND_BUFFER_SECONDARY) &&
                (begin_info->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
                skip |= ValidateObject(command_buffer, begin_info->pInheritanceInfo->framebuffer, kVulkanObjectTypeFramebuffer,
                                       true, VALIDATION_ERROR_0280006e, VALIDATION_ERROR_02a00009);
                skip |= ValidateObject(command_buffer, begin_info->pInheritanceInfo->renderPass, kVulkanObjectTypeRenderPass, false,
                                       VALIDATION_ERROR_0280006a, VALIDATION_ERROR_02a00009);
            }
        }
    }
    if (skip) {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    VkResult result = get_dispatch_table(ot_device_table_map, command_buffer)->BeginCommandBuffer(command_buffer, begin_info);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDebugReportCallbackEXT(VkInstance instance,
                                                            const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator,
                                                            VkDebugReportCallbackEXT *pCallback) {
    VkLayerInstanceDispatchTable *pInstanceTable = get_dispatch_table(ot_instance_table_map, instance);
    VkResult result = pInstanceTable->CreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pCallback);
    if (VK_SUCCESS == result) {
        layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(instance), layer_data_map);
        result = layer_create_msg_callback(instance_data->report_data, false, pCreateInfo, pAllocator, pCallback);
        CreateObject(instance, *pCallback, kVulkanObjectTypeDebugReportCallbackEXT, pAllocator);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT msgCallback,
                                                         const VkAllocationCallbacks *pAllocator) {
    VkLayerInstanceDispatchTable *pInstanceTable = get_dispatch_table(ot_instance_table_map, instance);
    pInstanceTable->DestroyDebugReportCallbackEXT(instance, msgCallback, pAllocator);
    layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(instance), layer_data_map);
    layer_destroy_msg_callback(instance_data->report_data, msgCallback, pAllocator);
    DestroyObject(instance, msgCallback, kVulkanObjectTypeDebugReportCallbackEXT, pAllocator, VALIDATION_ERROR_242009b4,
                  VALIDATION_ERROR_242009b6);
}

VKAPI_ATTR void VKAPI_CALL DebugReportMessageEXT(VkInstance instance, VkDebugReportFlagsEXT flags,
                                                 VkDebugReportObjectTypeEXT objType, uint64_t object, size_t location,
                                                 int32_t msgCode, const char *pLayerPrefix, const char *pMsg) {
    VkLayerInstanceDispatchTable *pInstanceTable = get_dispatch_table(ot_instance_table_map, instance);
    pInstanceTable->DebugReportMessageEXT(instance, flags, objType, object, location, msgCode, pLayerPrefix, pMsg);
}

static const VkExtensionProperties instance_extensions[] = {{VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_SPEC_VERSION}};

static const VkLayerProperties globalLayerProps = {"VK_LAYER_LUNARG_object_tracker",
                                                   VK_LAYER_API_VERSION,  // specVersion
                                                   1,                     // implementationVersion
                                                   "LunarG Validation Layer"};

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
        return util_GetExtensionProperties(1, instance_extensions, pCount, pProperties);

    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *pLayerName,
                                                                  uint32_t *pCount, VkExtensionProperties *pProperties) {
    if (pLayerName && !strcmp(pLayerName, globalLayerProps.layerName))
        return util_GetExtensionProperties(0, nullptr, pCount, pProperties);

    assert(physicalDevice);
    VkLayerInstanceDispatchTable *pTable = get_dispatch_table(ot_instance_table_map, physicalDevice);
    return pTable->EnumerateDeviceExtensionProperties(physicalDevice, NULL, pCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    std::lock_guard<std::mutex> lock(global_lock);
    bool skip = ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_1fc27a01,
                               VALIDATION_ERROR_UNDEFINED);
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;

    layer_data *phy_dev_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), layer_data_map);
    VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice)fpGetInstanceProcAddr(phy_dev_data->instance, "vkCreateDevice");
    if (fpCreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(*pDevice), layer_data_map);
    device_data->report_data = layer_debug_report_create_device(phy_dev_data->report_data, *pDevice);
    layer_init_device_dispatch_table(*pDevice, &device_data->dispatch_table, fpGetDeviceProcAddr);

    // Add link back to physDev
    device_data->physical_device = physicalDevice;

    initDeviceTable(*pDevice, fpGetDeviceProcAddr, ot_device_table_map);

    CreateObject(*pDevice, *pDevice, kVulkanObjectTypeDevice, pAllocator);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t *pSwapchainImageCount,
                                                     VkImage *pSwapchainImages) {
    bool skip = false;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_30805601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, false, VALIDATION_ERROR_3082f001,
                           VALIDATION_ERROR_UNDEFINED);
    lock.unlock();
    if (skip) return VK_ERROR_VALIDATION_FAILED_EXT;

    VkResult result = get_dispatch_table(ot_device_table_map, device)
                          ->GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
    if (pSwapchainImages != NULL) {
        lock.lock();
        for (uint32_t i = 0; i < *pSwapchainImageCount; i++) {
            CreateSwapchainImageObject(device, pSwapchainImages[i], swapchain);
        }
        lock.unlock();
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice,
                                                                  uint32_t *pQueueFamilyPropertyCount,
                                                                  VkQueueFamilyProperties *pQueueFamilyProperties) {
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_2da27a01,
                               VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) {
        return;
    }
    get_dispatch_table(ot_instance_table_map, physicalDevice)
        ->GetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
    std::lock_guard<std::mutex> lock(global_lock);
    if (pQueueFamilyProperties != NULL) {
        layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), layer_data_map);
        if (instance_data->queue_family_properties.size() < *pQueueFamilyPropertyCount) {
            instance_data->queue_family_properties.resize(*pQueueFamilyPropertyCount);
        }
        for (uint32_t i = 0; i < *pQueueFamilyPropertyCount; i++) {
            instance_data->queue_family_properties[i] = pQueueFamilyProperties[i];
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

    layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(*pInstance), layer_data_map);
    instance_data->instance = *pInstance;
    initInstanceTable(*pInstance, fpGetInstanceProcAddr, ot_instance_table_map);
    VkLayerInstanceDispatchTable *pInstanceTable = get_dispatch_table(ot_instance_table_map, *pInstance);

    // Look for one or more debug report create info structures, and copy the
    // callback(s) for each one found (for use by vkDestroyInstance)
    layer_copy_tmp_callbacks(pCreateInfo->pNext, &instance_data->num_tmp_callbacks, &instance_data->tmp_dbg_create_infos,
                             &instance_data->tmp_callbacks);

    instance_data->report_data = debug_report_create_instance(pInstanceTable, *pInstance, pCreateInfo->enabledExtensionCount,
                                                              pCreateInfo->ppEnabledExtensionNames);

    InitObjectTracker(instance_data, pAllocator);

    CreateObject(*pInstance, *pInstance, kVulkanObjectTypeInstance, pAllocator);

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                        VkPhysicalDevice *pPhysicalDevices) {
    bool skip = VK_FALSE;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |=
        ValidateObject(instance, instance, kVulkanObjectTypeInstance, false, VALIDATION_ERROR_2800bc01, VALIDATION_ERROR_UNDEFINED);
    lock.unlock();
    if (skip) {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    VkResult result = get_dispatch_table(ot_instance_table_map, instance)
                          ->EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    lock.lock();
    if (result == VK_SUCCESS) {
        if (pPhysicalDevices) {
            for (uint32_t i = 0; i < *pPhysicalDeviceCount; i++) {
                CreateObject(instance, pPhysicalDevices[i], kVulkanObjectTypePhysicalDevice, nullptr);
            }
        }
    }
    lock.unlock();
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
                                                      VkCommandBuffer *pCommandBuffers) {
    bool skip = VK_FALSE;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_16805601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, pAllocateInfo->commandPool, kVulkanObjectTypeCommandPool, false, VALIDATION_ERROR_02602801,
                           VALIDATION_ERROR_UNDEFINED);
    lock.unlock();

    if (skip) {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }

    VkResult result =
        get_dispatch_table(ot_device_table_map, device)->AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);

    lock.lock();
    for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; i++) {
        AllocateCommandBuffer(device, pAllocateInfo->commandPool, pCommandBuffers[i], pAllocateInfo->level);
    }
    lock.unlock();

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                      VkDescriptorSet *pDescriptorSets) {
    bool skip = VK_FALSE;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_16a05601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, pAllocateInfo->descriptorPool, kVulkanObjectTypeDescriptorPool, false, VALIDATION_ERROR_04c04601,
                           VALIDATION_ERROR_04c00009);
    for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
        skip |= ValidateObject(device, pAllocateInfo->pSetLayouts[i], kVulkanObjectTypeDescriptorSetLayout, false,
                               VALIDATION_ERROR_04c22c01, VALIDATION_ERROR_04c00009);
    }
    lock.unlock();
    if (skip) {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }

    VkResult result =
        get_dispatch_table(ot_device_table_map, device)->AllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets);

    if (VK_SUCCESS == result) {
        lock.lock();
        for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
            AllocateDescriptorSet(device, pAllocateInfo->descriptorPool, pDescriptorSets[i]);
        }
        lock.unlock();
    }

    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                              const VkCommandBuffer *pCommandBuffers) {
    bool skip = false;
    std::unique_lock<std::mutex> lock(global_lock);
    ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28405601, VALIDATION_ERROR_UNDEFINED);
    ValidateObject(device, commandPool, kVulkanObjectTypeCommandPool, false, VALIDATION_ERROR_28402801, VALIDATION_ERROR_28402807);
    for (uint32_t i = 0; i < commandBufferCount; i++) {
        if (pCommandBuffers[i] != VK_NULL_HANDLE) {
            skip |= ValidateCommandBuffer(device, commandPool, pCommandBuffers[i]);
        }
    }

    for (uint32_t i = 0; i < commandBufferCount; i++) {
        DestroyObject(device, pCommandBuffers[i], kVulkanObjectTypeCommandBuffer, nullptr, VALIDATION_ERROR_UNDEFINED,
                      VALIDATION_ERROR_UNDEFINED);
    }

    lock.unlock();
    if (!skip) {
        get_dispatch_table(ot_device_table_map, device)
            ->FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
    }
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *pAllocator) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    // A swapchain's images are implicitly deleted when the swapchain is deleted.
    // Remove this swapchain's images from our map of such images.
    std::unordered_map<uint64_t, ObjTrackState *>::iterator itr = device_data->swapchainImageMap.begin();
    while (itr != device_data->swapchainImageMap.end()) {
        ObjTrackState *pNode = (*itr).second;
        if (pNode->parent_object == HandleToUint64(swapchain)) {
            delete pNode;
            auto delete_item = itr++;
            device_data->swapchainImageMap.erase(delete_item);
        } else {
            ++itr;
        }
    }
    DestroyObject(device, swapchain, kVulkanObjectTypeSwapchainKHR, pAllocator, VALIDATION_ERROR_26e00a06,
                  VALIDATION_ERROR_26e00a08);
    lock.unlock();

    get_dispatch_table(ot_device_table_map, device)->DestroySwapchainKHR(device, swapchain, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount,
                                                  const VkDescriptorSet *pDescriptorSets) {
    bool skip = false;
    VkResult result = VK_ERROR_VALIDATION_FAILED_EXT;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_28605601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, descriptorPool, kVulkanObjectTypeDescriptorPool, false, VALIDATION_ERROR_28604601,
                           VALIDATION_ERROR_28604607);
    for (uint32_t i = 0; i < descriptorSetCount; i++) {
        if (pDescriptorSets[i] != VK_NULL_HANDLE) {
            skip |= ValidateDescriptorSet(device, descriptorPool, pDescriptorSets[i]);
        }
    }

    for (uint32_t i = 0; i < descriptorSetCount; i++) {
        DestroyObject(device, pDescriptorSets[i], kVulkanObjectTypeDescriptorSet, nullptr, VALIDATION_ERROR_UNDEFINED,
                      VALIDATION_ERROR_UNDEFINED);
    }

    lock.unlock();
    if (!skip) {
        result = get_dispatch_table(ot_device_table_map, device)
                     ->FreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                 const VkAllocationCallbacks *pAllocator) {
    bool skip = VK_FALSE;
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_24405601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, descriptorPool, kVulkanObjectTypeDescriptorPool, true, VALIDATION_ERROR_24404601,
                           VALIDATION_ERROR_24404607);
    lock.unlock();
    if (skip) {
        return;
    }
    // A DescriptorPool's descriptor sets are implicitly deleted when the pool is deleted.
    // Remove this pool's descriptor sets from our descriptorSet map.
    lock.lock();
    std::unordered_map<uint64_t, ObjTrackState *>::iterator itr = device_data->object_map[kVulkanObjectTypeDescriptorSet].begin();
    while (itr != device_data->object_map[kVulkanObjectTypeDescriptorSet].end()) {
        ObjTrackState *pNode = (*itr).second;
        auto del_itr = itr++;
        if (pNode->parent_object == HandleToUint64(descriptorPool)) {
            DestroyObject(device, (VkDescriptorSet)((*del_itr).first), kVulkanObjectTypeDescriptorSet, nullptr,
                          VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
        }
    }
    DestroyObject(device, descriptorPool, kVulkanObjectTypeDescriptorPool, pAllocator, VALIDATION_ERROR_24400260,
                  VALIDATION_ERROR_24400262);
    lock.unlock();
    get_dispatch_table(ot_device_table_map, device)->DestroyDescriptorPool(device, descriptorPool, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    bool skip = false;
    std::unique_lock<std::mutex> lock(global_lock);
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_24005601, VALIDATION_ERROR_UNDEFINED);
    skip |= ValidateObject(device, commandPool, kVulkanObjectTypeCommandPool, true, VALIDATION_ERROR_24002801,
                           VALIDATION_ERROR_24002807);
    lock.unlock();
    if (skip) {
        return;
    }
    lock.lock();
    // A CommandPool's command buffers are implicitly deleted when the pool is deleted.
    // Remove this pool's cmdBuffers from our cmd buffer map.
    auto itr = device_data->object_map[kVulkanObjectTypeCommandBuffer].begin();
    auto del_itr = itr;
    while (itr != device_data->object_map[kVulkanObjectTypeCommandBuffer].end()) {
        ObjTrackState *pNode = (*itr).second;
        del_itr = itr++;
        if (pNode->parent_object == HandleToUint64(commandPool)) {
            skip |= ValidateCommandBuffer(device, commandPool, reinterpret_cast<VkCommandBuffer>((*del_itr).first));
            DestroyObject(device, reinterpret_cast<VkCommandBuffer>((*del_itr).first), kVulkanObjectTypeCommandBuffer, nullptr,
                          VALIDATION_ERROR_UNDEFINED, VALIDATION_ERROR_UNDEFINED);
        }
    }
    DestroyObject(device, commandPool, kVulkanObjectTypeCommandPool, pAllocator, VALIDATION_ERROR_24000054,
                  VALIDATION_ERROR_24000056);
    lock.unlock();
    get_dispatch_table(ot_device_table_map, device)->DestroyCommandPool(device, commandPool, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties2KHR(VkPhysicalDevice physicalDevice,
                                                                      uint32_t *pQueueFamilyPropertyCount,
                                                                      VkQueueFamilyProperties2KHR *pQueueFamilyProperties) {
    bool skip = false;
    {
        std::lock_guard<std::mutex> lock(global_lock);
        skip |= ValidateObject(physicalDevice, physicalDevice, kVulkanObjectTypePhysicalDevice, false, VALIDATION_ERROR_UNDEFINED,
                               VALIDATION_ERROR_UNDEFINED);
    }
    if (skip) {
        return;
    }
    get_dispatch_table(ot_instance_table_map, physicalDevice)
        ->GetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
    std::lock_guard<std::mutex> lock(global_lock);
    if (pQueueFamilyProperties != NULL) {
        layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(physicalDevice), layer_data_map);
        if (instance_data->queue_family_properties.size() < *pQueueFamilyPropertyCount) {
            instance_data->queue_family_properties.resize(*pQueueFamilyPropertyCount);
        }
        for (uint32_t i = 0; i < *pQueueFamilyPropertyCount; i++) {
            instance_data->queue_family_properties[i] = pQueueFamilyProperties[i].queueFamilyProperties;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectNameEXT(VkDevice device, const VkDebugMarkerObjectNameInfoEXT *pNameInfo) {
    bool skip = VK_FALSE;
    std::unique_lock<std::mutex> lock(global_lock);
    layer_data *dev_data = GetLayerDataPtr(get_dispatch_key(device), layer_data_map);
    if (pNameInfo->pObjectName) {
        dev_data->report_data->debugObjectNameMap->insert(
            std::make_pair<uint64_t, std::string>((uint64_t &&) pNameInfo->object, pNameInfo->pObjectName));
    } else {
        dev_data->report_data->debugObjectNameMap->erase(pNameInfo->object);
    }
    skip |= ValidateObject(device, device, kVulkanObjectTypeDevice, false, VALIDATION_ERROR_23605601, VALIDATION_ERROR_UNDEFINED);
    lock.unlock();
    if (skip) {
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    VkResult result = dev_data->dispatch_table.DebugMarkerSetObjectNameEXT(device, pNameInfo);
    return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetPhysicalDeviceProcAddr(VkInstance instance, const char *funcName) {
    assert(instance);

    if (get_dispatch_table(ot_instance_table_map, instance)->GetPhysicalDeviceProcAddr == NULL) {
        return NULL;
    }
    return get_dispatch_table(ot_instance_table_map, instance)->GetPhysicalDeviceProcAddr(instance, funcName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char *funcName) {
    const auto item = name_to_funcptr_map.find(funcName);
    if (item != name_to_funcptr_map.end()) {
        return reinterpret_cast<PFN_vkVoidFunction>(item->second);
    }

    auto table = get_dispatch_table(ot_device_table_map, device);
    if (!table->GetDeviceProcAddr) return NULL;
    return table->GetDeviceProcAddr(device, funcName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char *funcName) {
    const auto item = name_to_funcptr_map.find(funcName);
    if (item != name_to_funcptr_map.end()) {
        return reinterpret_cast<PFN_vkVoidFunction>(item->second);
    }

    auto table = get_dispatch_table(ot_instance_table_map, instance);
    if (!table->GetInstanceProcAddr) return nullptr;
    return table->GetInstanceProcAddr(instance, funcName);
}

}  // namespace object_tracker

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,
                                                                                      VkExtensionProperties *pProperties) {
    return object_tracker::EnumerateInstanceExtensionProperties(pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pCount,
                                                                                  VkLayerProperties *pProperties) {
    return object_tracker::EnumerateInstanceLayerProperties(pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                                                                VkLayerProperties *pProperties) {
    // The layer command handles VK_NULL_HANDLE just fine internally
    assert(physicalDevice == VK_NULL_HANDLE);
    return object_tracker::EnumerateDeviceLayerProperties(VK_NULL_HANDLE, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice dev, const char *funcName) {
    return object_tracker::GetDeviceProcAddr(dev, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *funcName) {
    return object_tracker::GetInstanceProcAddr(instance, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                                    const char *pLayerName, uint32_t *pCount,
                                                                                    VkExtensionProperties *pProperties) {
    // The layer command handles VK_NULL_HANDLE just fine internally
    assert(physicalDevice == VK_NULL_HANDLE);
    return object_tracker::EnumerateDeviceExtensionProperties(VK_NULL_HANDLE, pLayerName, pCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_layerGetPhysicalDeviceProcAddr(VkInstance instance,
                                                                                           const char *funcName) {
    return object_tracker::GetPhysicalDeviceProcAddr(instance, funcName);
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
        object_tracker::loader_layer_if_version = pVersionStruct->loaderLayerInterfaceVersion;
    } else if (pVersionStruct->loaderLayerInterfaceVersion > CURRENT_LOADER_LAYER_INTERFACE_VERSION) {
        pVersionStruct->loaderLayerInterfaceVersion = CURRENT_LOADER_LAYER_INTERFACE_VERSION;
    }

    return VK_SUCCESS;
}

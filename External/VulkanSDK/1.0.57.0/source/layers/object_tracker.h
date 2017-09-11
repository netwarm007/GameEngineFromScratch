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

#include <mutex>
#include <cinttypes>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

#include "vk_loader_platform.h"
#include "vulkan/vulkan.h"
#include "vk_layer_config.h"
#include "vk_layer_data.h"
#include "vk_layer_logging.h"
#include "vk_layer_table.h"
#include "vk_object_types.h"
#include "vulkan/vk_layer.h"
#include "vk_object_types.h"
#include "vk_enum_string_helper.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_table.h"
#include "vk_layer_utils.h"
#include "vulkan/vk_layer.h"
#include "vk_dispatch_table_helper.h"
#include "vk_validation_error_messages.h"

namespace object_tracker {

// Object Tracker ERROR codes
enum ObjectTrackerError {
    OBJTRACK_NONE,            // Used for INFO & other non-error messages
    OBJTRACK_UNKNOWN_OBJECT,  // Updating uses of object that's not in global object list
    OBJTRACK_INTERNAL_ERROR,  // Bug with data tracking within the layer
    OBJTRACK_OBJECT_LEAK,     // OBJECT was not correctly freed/destroyed
};

// Object Status -- used to track state of individual objects
typedef VkFlags ObjectStatusFlags;
enum ObjectStatusFlagBits {
    OBJSTATUS_NONE = 0x00000000,                      // No status is set
    OBJSTATUS_FENCE_IS_SUBMITTED = 0x00000001,        // Fence has been submitted
    OBJSTATUS_VIEWPORT_BOUND = 0x00000002,            // Viewport state object has been bound
    OBJSTATUS_RASTER_BOUND = 0x00000004,              // Viewport state object has been bound
    OBJSTATUS_COLOR_BLEND_BOUND = 0x00000008,         // Viewport state object has been bound
    OBJSTATUS_DEPTH_STENCIL_BOUND = 0x00000010,       // Viewport state object has been bound
    OBJSTATUS_GPU_MEM_MAPPED = 0x00000020,            // Memory object is currently mapped
    OBJSTATUS_COMMAND_BUFFER_SECONDARY = 0x00000040,  // Command Buffer is of type SECONDARY
    OBJSTATUS_CUSTOM_ALLOCATOR = 0x00000080,          // Allocated with custom allocator
};

// Object and state information structure
struct ObjTrackState {
    uint64_t handle;               // Object handle (new)
    VulkanObjectType object_type;  // Object type identifier
    ObjectStatusFlags status;      // Object state
    uint64_t parent_object;        // Parent object
};

// Track Queue information
struct ObjTrackQueueInfo {
    uint32_t queue_node_index;
    VkQueue queue;
};

// Layer name string to be logged with validation messages.
const char LayerName[] = "ObjectTracker";

typedef std::unordered_map<uint64_t, ObjTrackState *> object_map_type;

struct layer_data {
    VkInstance instance;
    VkPhysicalDevice physical_device;

    uint64_t num_objects[kVulkanObjectTypeMax + 1];
    uint64_t num_total_objects;

    debug_report_data *report_data;
    std::vector<VkDebugReportCallbackEXT> logging_callback;
    // The following are for keeping track of the temporary callbacks that can
    // be used in vkCreateInstance and vkDestroyInstance:
    uint32_t num_tmp_callbacks;
    VkDebugReportCallbackCreateInfoEXT *tmp_dbg_create_infos;
    VkDebugReportCallbackEXT *tmp_callbacks;

    std::vector<VkQueueFamilyProperties> queue_family_properties;

    // Vector of unordered_maps per object type to hold ObjTrackState info
    std::vector<object_map_type> object_map;
    // Special-case map for swapchain images
    std::unordered_map<uint64_t, ObjTrackState *> swapchainImageMap;
    // Map of queue information structures, one per queue
    std::unordered_map<VkQueue, ObjTrackQueueInfo *> queue_info_map;

    VkLayerDispatchTable dispatch_table;
    // Default constructor
    layer_data()
        : instance(nullptr),
          physical_device(nullptr),
          num_objects{},
          num_total_objects(0),
          report_data(nullptr),
          num_tmp_callbacks(0),
          tmp_dbg_create_infos(nullptr),
          tmp_callbacks(nullptr),
          object_map{},
          dispatch_table{} {
        object_map.resize(kVulkanObjectTypeMax + 1);
    }
};

extern std::unordered_map<void *, layer_data *> layer_data_map;
extern device_table_map ot_device_table_map;
extern instance_table_map ot_instance_table_map;
extern std::mutex global_lock;
extern uint64_t object_track_index;
extern uint32_t loader_layer_if_version;
extern const std::unordered_map<std::string, void *> name_to_funcptr_map;

void DeviceReportUndestroyedObjects(VkDevice device, VulkanObjectType object_type, enum UNIQUE_VALIDATION_ERROR_CODE error_code);
void CreateQueue(VkDevice device, VkQueue vkObj);
void AddQueueInfo(VkDevice device, uint32_t queue_node_index, VkQueue queue);
void ValidateQueueFlags(VkQueue queue, const char *function);
void AllocateCommandBuffer(VkDevice device, const VkCommandPool command_pool, const VkCommandBuffer command_buffer,
                           VkCommandBufferLevel level);
void AllocateDescriptorSet(VkDevice device, VkDescriptorPool descriptor_pool, VkDescriptorSet descriptor_set);
void CreateSwapchainImageObject(VkDevice dispatchable_object, VkImage swapchain_image, VkSwapchainKHR swapchain);
void ReportUndestroyedObjects(VkDevice device, UNIQUE_VALIDATION_ERROR_CODE error_code);


template <typename T1, typename T2>
bool ValidateObject(T1 dispatchable_object, T2 object, VulkanObjectType object_type, bool null_allowed,
                    enum UNIQUE_VALIDATION_ERROR_CODE invalid_handle_code, enum UNIQUE_VALIDATION_ERROR_CODE wrong_device_code) {
    if (null_allowed && (object == VK_NULL_HANDLE)) {
        return false;
    }
    auto object_handle = HandleToUint64(object);
    VkDebugReportObjectTypeEXT debug_object_type = get_debug_report_enum[object_type];

    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(dispatchable_object), layer_data_map);
    // Look for object in device object map
    if (device_data->object_map[object_type].find(object_handle) == device_data->object_map[object_type].end()) {
        // If object is an image, also look for it in the swapchain image map
        if ((object_type != kVulkanObjectTypeImage) ||
            (device_data->swapchainImageMap.find(object_handle) == device_data->swapchainImageMap.end())) {
            // Object not found, look for it in other device object maps
            for (auto other_device_data : layer_data_map) {
                if (other_device_data.second != device_data) {
                    if (other_device_data.second->object_map[object_type].find(object_handle) !=
                            other_device_data.second->object_map[object_type].end() ||
                        (object_type == kVulkanObjectTypeImage && other_device_data.second->swapchainImageMap.find(object_handle) !=
                                                                      other_device_data.second->swapchainImageMap.end())) {
                        // Object found on other device, report an error if object has a device parent error code
                        if (wrong_device_code != VALIDATION_ERROR_UNDEFINED) {
                            return log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, debug_object_type,
                                           object_handle, __LINE__, wrong_device_code, LayerName,
                                           "Object 0x%" PRIxLEAST64
                                           " was not created, allocated or retrieved from the correct device. %s",
                                           object_handle, validation_error_map[wrong_device_code]);
                        } else {
                            return false;
                        }
                    }
                }
            }
            // Report an error if object was not found anywhere
            return log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, debug_object_type, object_handle, __LINE__,
                           invalid_handle_code, LayerName, "Invalid %s Object 0x%" PRIxLEAST64 ". %s", object_string[object_type],
                           object_handle, validation_error_map[invalid_handle_code]);
        }
    }
    return false;
}

template <typename T1, typename T2>
void CreateObject(T1 dispatchable_object, T2 object, VulkanObjectType object_type, const VkAllocationCallbacks *pAllocator) {
    layer_data *instance_data = GetLayerDataPtr(get_dispatch_key(dispatchable_object), layer_data_map);

    auto object_handle = HandleToUint64(object);
    bool custom_allocator = pAllocator != nullptr;

    if (!instance_data->object_map[object_type].count(object_handle)) {
        VkDebugReportObjectTypeEXT debug_object_type = get_debug_report_enum[object_type];
        log_msg(instance_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, debug_object_type, object_handle, __LINE__,
                OBJTRACK_NONE, LayerName, "OBJ[0x%" PRIxLEAST64 "] : CREATE %s object 0x%" PRIxLEAST64, object_track_index++,
                object_string[object_type], object_handle);

        ObjTrackState *pNewObjNode = new ObjTrackState;
        pNewObjNode->object_type = object_type;
        pNewObjNode->status = custom_allocator ? OBJSTATUS_CUSTOM_ALLOCATOR : OBJSTATUS_NONE;
        pNewObjNode->handle = object_handle;

        instance_data->object_map[object_type][object_handle] = pNewObjNode;
        instance_data->num_objects[object_type]++;
        instance_data->num_total_objects++;
    }
}

template <typename T1, typename T2>
void DestroyObject(T1 dispatchable_object, T2 object, VulkanObjectType object_type, const VkAllocationCallbacks *pAllocator,
                   enum UNIQUE_VALIDATION_ERROR_CODE expected_custom_allocator_code,
                   enum UNIQUE_VALIDATION_ERROR_CODE expected_default_allocator_code) {
    layer_data *device_data = GetLayerDataPtr(get_dispatch_key(dispatchable_object), layer_data_map);

    auto object_handle = HandleToUint64(object);
    bool custom_allocator = pAllocator != nullptr;
    VkDebugReportObjectTypeEXT debug_object_type = get_debug_report_enum[object_type];

    if (object_handle != VK_NULL_HANDLE) {
        auto item = device_data->object_map[object_type].find(object_handle);
        if (item != device_data->object_map[object_type].end()) {
            ObjTrackState *pNode = item->second;
            assert(device_data->num_total_objects > 0);
            device_data->num_total_objects--;
            assert(device_data->num_objects[pNode->object_type] > 0);
            device_data->num_objects[pNode->object_type]--;

            log_msg(device_data->report_data, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, debug_object_type, object_handle, __LINE__,
                    OBJTRACK_NONE, LayerName,
                    "OBJ_STAT Destroy %s obj 0x%" PRIxLEAST64 " (%" PRIu64 " total objs remain & %" PRIu64 " %s objs).",
                    object_string[object_type], HandleToUint64(object), device_data->num_total_objects,
                    device_data->num_objects[pNode->object_type], object_string[object_type]);

            auto allocated_with_custom = (pNode->status & OBJSTATUS_CUSTOM_ALLOCATOR) ? true : false;
            if (allocated_with_custom && !custom_allocator && expected_custom_allocator_code != VALIDATION_ERROR_UNDEFINED) {
                // This check only verifies that custom allocation callbacks were provided to both Create and Destroy calls,
                // it cannot verify that these allocation callbacks are compatible with each other.
                log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, debug_object_type, object_handle, __LINE__,
                        expected_custom_allocator_code, LayerName,
                        "Custom allocator not specified while destroying %s obj 0x%" PRIxLEAST64 " but specified at creation. %s",
                        object_string[object_type], object_handle, validation_error_map[expected_custom_allocator_code]);
            } else if (!allocated_with_custom && custom_allocator &&
                       expected_default_allocator_code != VALIDATION_ERROR_UNDEFINED) {
                log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, debug_object_type, object_handle, __LINE__,
                        expected_default_allocator_code, LayerName,
                        "Custom allocator specified while destroying %s obj 0x%" PRIxLEAST64 " but not specified at creation. %s",
                        object_string[object_type], object_handle, validation_error_map[expected_default_allocator_code]);
            }

            delete pNode;
            device_data->object_map[object_type].erase(item);
        } else {
            log_msg(device_data->report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, object_handle,
                    __LINE__, OBJTRACK_UNKNOWN_OBJECT, LayerName,
                    "Unable to remove %s obj 0x%" PRIxLEAST64 ". Was it created? Has it already been destroyed?",
                    object_string[object_type], object_handle);
        }
    }
}

}  // namespace object_tracker

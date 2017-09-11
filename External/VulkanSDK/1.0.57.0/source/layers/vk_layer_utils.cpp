/* Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
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
 * Author: Dave Houlton <daveh@lunarg.com>
 *
 */

#include <string.h>
#include <string>
#include <vector>
#include <map>
#include "vulkan/vulkan.h"
#include "vk_layer_config.h"
#include "vk_layer_utils.h"

static const uint8_t UTF8_ONE_BYTE_CODE = 0xC0;
static const uint8_t UTF8_ONE_BYTE_MASK = 0xE0;
static const uint8_t UTF8_TWO_BYTE_CODE = 0xE0;
static const uint8_t UTF8_TWO_BYTE_MASK = 0xF0;
static const uint8_t UTF8_THREE_BYTE_CODE = 0xF0;
static const uint8_t UTF8_THREE_BYTE_MASK = 0xF8;
static const uint8_t UTF8_DATA_BYTE_CODE = 0x80;
static const uint8_t UTF8_DATA_BYTE_MASK = 0xC0;

VK_LAYER_EXPORT VkStringErrorFlags vk_string_validate(const int max_length, const char *utf8) {
    VkStringErrorFlags result = VK_STRING_ERROR_NONE;
    int num_char_bytes = 0;
    int i, j;

    for (i = 0; i <= max_length; i++) {
        if (utf8[i] == 0) {
            break;
        } else if (i == max_length) {
            result = VK_STRING_ERROR_LENGTH;
            break;
        } else if ((utf8[i] >= 0xa) && (utf8[i] < 0x7f)) {
            num_char_bytes = 0;
        } else if ((utf8[i] & UTF8_ONE_BYTE_MASK) == UTF8_ONE_BYTE_CODE) {
            num_char_bytes = 1;
        } else if ((utf8[i] & UTF8_TWO_BYTE_MASK) == UTF8_TWO_BYTE_CODE) {
            num_char_bytes = 2;
        } else if ((utf8[i] & UTF8_THREE_BYTE_MASK) == UTF8_THREE_BYTE_CODE) {
            num_char_bytes = 3;
        } else {
            result = VK_STRING_ERROR_BAD_DATA;
        }

        // Validate the following num_char_bytes of data
        for (j = 0; (j < num_char_bytes) && (i < max_length); j++) {
            if (++i == max_length) {
                result |= VK_STRING_ERROR_LENGTH;
                break;
            }
            if ((utf8[i] & UTF8_DATA_BYTE_MASK) != UTF8_DATA_BYTE_CODE) {
                result |= VK_STRING_ERROR_BAD_DATA;
            }
        }
    }
    return result;
}

// Utility function for finding a text string in another string
VK_LAYER_EXPORT bool white_list(const char *item, const char *list) {
    std::string candidate(item);
    std::string white_list(list);
    return (white_list.find(candidate) != std::string::npos);
}

// Debug callbacks get created in three ways:
//   o  Application-defined debug callbacks
//   o  Through settings in a vk_layer_settings.txt file
//   o  By default, if neither an app-defined debug callback nor a vk_layer_settings.txt file is present
//
// At layer initialization time, default logging callbacks are created to output layer error messages.
// If a vk_layer_settings.txt file is present its settings will override any default settings.
//
// If a vk_layer_settings.txt file is present and an application defines a debug callback, both callbacks
// will be active.  If no vk_layer_settings.txt file is present, creating an application-defined debug
// callback will cause the default callbacks to be unregisterd and removed.
VK_LAYER_EXPORT void layer_debug_actions(debug_report_data *report_data, std::vector<VkDebugReportCallbackEXT> &logging_callback,
                                         const VkAllocationCallbacks *pAllocator, const char *layer_identifier) {
    VkDebugReportCallbackEXT callback = VK_NULL_HANDLE;

    std::string report_flags_key = layer_identifier;
    std::string debug_action_key = layer_identifier;
    std::string log_filename_key = layer_identifier;
    report_flags_key.append(".report_flags");
    debug_action_key.append(".debug_action");
    log_filename_key.append(".log_filename");

    // Initialize layer options
    VkDebugReportFlagsEXT report_flags = GetLayerOptionFlags(report_flags_key, report_flags_option_definitions, 0);
    VkLayerDbgActionFlags debug_action = GetLayerOptionFlags(debug_action_key, debug_actions_option_definitions, 0);
    // Flag as default if these settings are not from a vk_layer_settings.txt file
    bool default_layer_callback = (debug_action & VK_DBG_LAYER_ACTION_DEFAULT) ? true : false;

    if (debug_action & VK_DBG_LAYER_ACTION_LOG_MSG) {
        const char *log_filename = getLayerOption(log_filename_key.c_str());
        FILE *log_output = getLayerLogOutput(log_filename, layer_identifier);
        VkDebugReportCallbackCreateInfoEXT dbgCreateInfo;
        memset(&dbgCreateInfo, 0, sizeof(dbgCreateInfo));
        dbgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
        dbgCreateInfo.flags = report_flags;
        dbgCreateInfo.pfnCallback = log_callback;
        dbgCreateInfo.pUserData = (void *)log_output;
        layer_create_msg_callback(report_data, default_layer_callback, &dbgCreateInfo, pAllocator, &callback);
        logging_callback.push_back(callback);
    }

    callback = VK_NULL_HANDLE;

    if (debug_action & VK_DBG_LAYER_ACTION_DEBUG_OUTPUT) {
        VkDebugReportCallbackCreateInfoEXT dbgCreateInfo;
        memset(&dbgCreateInfo, 0, sizeof(dbgCreateInfo));
        dbgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
        dbgCreateInfo.flags = report_flags;
        dbgCreateInfo.pfnCallback = win32_debug_output_msg;
        dbgCreateInfo.pUserData = NULL;
        layer_create_msg_callback(report_data, default_layer_callback, &dbgCreateInfo, pAllocator, &callback);
        logging_callback.push_back(callback);
    }
}

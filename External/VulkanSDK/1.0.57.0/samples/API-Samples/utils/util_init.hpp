/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google, Inc.
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

#ifndef UTIL_INIT
#define UTIL_INIT

#include "util.hpp"

// Make sure functions start with init, execute, or destroy to assist codegen

VkResult init_global_extension_properties(layer_properties &layer_props);

VkResult init_global_layer_properties(sample_info &info);

VkResult init_device_extension_properties(struct sample_info &info,
                                          layer_properties &layer_props);

void init_instance_extension_names(struct sample_info &info);
VkResult init_instance(struct sample_info &info,
                       char const *const app_short_name);
void init_device_extension_names(struct sample_info &info);
VkResult init_device(struct sample_info &info);
VkResult init_enumerate_device(struct sample_info &info,
                               uint32_t gpu_count = 1);
VkBool32 demo_check_layers(const std::vector<layer_properties> &layer_props,
                           const std::vector<const char *> &layer_names);
void init_connection(struct sample_info &info);
void init_window(struct sample_info &info);
void init_queue_family_index(struct sample_info &info);
void init_presentable_image(struct sample_info &info);
void execute_queue_cmdbuf(struct sample_info &info,
                          const VkCommandBuffer *cmd_bufs, VkFence &fence);
void execute_pre_present_barrier(struct sample_info &info);
void execute_present_image(struct sample_info &info);
void init_swapchain_extension(struct sample_info &info);
void init_command_pool(struct sample_info &info);
void init_command_buffer(struct sample_info &info);
void execute_begin_command_buffer(struct sample_info &info);
void execute_end_command_buffer(struct sample_info &info);
void execute_queue_command_buffer(struct sample_info &info);
void init_device_queue(struct sample_info &info);
void init_swap_chain(
    struct sample_info &info,
    VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
void init_depth_buffer(struct sample_info &info);
void init_uniform_buffer(struct sample_info &info);
void init_descriptor_and_pipeline_layouts(struct sample_info &info, bool use_texture,
                                          VkDescriptorSetLayoutCreateFlags descSetLayoutCreateFlags = 0);
void init_renderpass(
    struct sample_info &info, bool include_depth, bool clear = true,
    VkImageLayout finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
void init_vertex_buffer(struct sample_info &info, const void *vertexData,
                        uint32_t dataSize, uint32_t dataStride,
                        bool use_texture);
void init_framebuffers(struct sample_info &info, bool include_depth);
void init_descriptor_pool(struct sample_info &info, bool use_texture);
void init_descriptor_set(struct sample_info &info, bool use_texture);
void init_shaders(struct sample_info &info, const char *vertShaderText,
                  const char *fragShaderText);
void init_pipeline_cache(struct sample_info &info);
void init_pipeline(struct sample_info &info, VkBool32 include_depth,
                   VkBool32 include_vi = true);
void init_sampler(struct sample_info &info, VkSampler &sampler);
void init_image(struct sample_info &info, texture_object &texObj,
                const char *textureName, VkImageUsageFlags extraUsages = 0,
                VkFormatFeatureFlags extraFeatures = 0);
void init_texture(struct sample_info &info, const char *textureName = nullptr,
                  VkImageUsageFlags extraUsages = 0,
                  VkFormatFeatureFlags extraFeatures = 0);
void init_viewports(struct sample_info &info);
void init_scissors(struct sample_info &info);
void init_fence(struct sample_info &info, VkFence &fence);
void init_submit_info(struct sample_info &info, VkSubmitInfo &submit_info,
                      VkPipelineStageFlags &pipe_stage_flags);
void init_present_info(struct sample_info &info, VkPresentInfoKHR &present);
void init_clear_color_and_depth(struct sample_info &info,
                                VkClearValue *clear_values);
void init_render_pass_begin_info(struct sample_info &info,
                                 VkRenderPassBeginInfo &rp_begin);
void init_window_size(struct sample_info &info, int32_t default_width,
                      int32_t default_height);

VkResult init_debug_report_callback(struct sample_info &info,
                                    PFN_vkDebugReportCallbackEXT dbgFunc);
void destroy_debug_report_callback(struct sample_info &info);
void destroy_pipeline(struct sample_info &info);
void destroy_pipeline_cache(struct sample_info &info);
void destroy_descriptor_pool(struct sample_info &info);
void destroy_vertex_buffer(struct sample_info &info);
void destroy_textures(struct sample_info &info);
void destroy_framebuffers(struct sample_info &info);
void destroy_shaders(struct sample_info &info);
void destroy_renderpass(struct sample_info &info);
void destroy_descriptor_and_pipeline_layouts(struct sample_info &info);
void destroy_uniform_buffer(struct sample_info &info);
void destroy_depth_buffer(struct sample_info &info);
void destroy_swap_chain(struct sample_info &info);
void destroy_command_buffer(struct sample_info &info);
void destroy_command_pool(struct sample_info &info);
void destroy_device(struct sample_info &info);
void destroy_instance(struct sample_info &info);
void destroy_window(struct sample_info &info);

#endif // UTIL_INIT

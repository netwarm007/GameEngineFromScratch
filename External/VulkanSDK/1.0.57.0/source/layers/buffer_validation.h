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
 * Mark Lobodzinski <mark@lunarg.com>
 */
#ifndef CORE_VALIDATION_BUFFER_VALIDATION_H_
#define CORE_VALIDATION_BUFFER_VALIDATION_H_

#include "core_validation_types.h"
#include "core_validation_error_enums.h"
#include "vulkan/vk_layer.h"
#include <limits.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <bitset>

using core_validation::layer_data;

bool PreCallValidateCreateImage(layer_data *device_data, const VkImageCreateInfo *pCreateInfo,
                                const VkAllocationCallbacks *pAllocator, VkImage *pImage);

void PostCallRecordCreateImage(layer_data *device_data, const VkImageCreateInfo *pCreateInfo, VkImage *pImage);

void PostCallRecordDestroyImage(layer_data *device_data, VkImage image, IMAGE_STATE *image_state, VK_OBJECT obj_struct);

bool PreCallValidateDestroyImage(layer_data *device_data, VkImage image, IMAGE_STATE **image_state, VK_OBJECT *obj_struct);

bool ValidateImageAttributes(layer_data *device_data, IMAGE_STATE *image_state, VkImageSubresourceRange range);

uint32_t ResolveRemainingLevels(const VkImageSubresourceRange *range, uint32_t mip_levels);

uint32_t ResolveRemainingLayers(const VkImageSubresourceRange *range, uint32_t layers);

bool VerifyClearImageLayout(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *image_state,
                            VkImageSubresourceRange range, VkImageLayout dest_image_layout, const char *func_name);

bool VerifyImageLayout(layer_data const *device_data, GLOBAL_CB_NODE const *cb_node, IMAGE_STATE *image_state,
                       VkImageSubresourceLayers subLayers, VkImageLayout explicit_layout, VkImageLayout optimal_layout,
                       const char *caller, UNIQUE_VALIDATION_ERROR_CODE msg_code, bool *error);

void RecordClearImageLayout(layer_data *dev_data, GLOBAL_CB_NODE *cb_node, VkImage image, VkImageSubresourceRange range,
                            VkImageLayout dest_image_layout);

bool PreCallValidateCmdClearColorImage(layer_data *dev_data, VkCommandBuffer commandBuffer, VkImage image,
                                       VkImageLayout imageLayout, uint32_t rangeCount, const VkImageSubresourceRange *pRanges);

void PreCallRecordCmdClearImage(layer_data *dev_data, VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout,
                                uint32_t rangeCount, const VkImageSubresourceRange *pRanges);

bool PreCallValidateCmdClearDepthStencilImage(layer_data *dev_data, VkCommandBuffer commandBuffer, VkImage image,
                                              VkImageLayout imageLayout, uint32_t rangeCount,
                                              const VkImageSubresourceRange *pRanges);

bool FindLayoutVerifyNode(layer_data const *device_data, GLOBAL_CB_NODE const *pCB, ImageSubresourcePair imgpair,
                          IMAGE_CMD_BUF_LAYOUT_NODE &node, const VkImageAspectFlags aspectMask);

bool FindLayoutVerifyLayout(layer_data const *device_data, ImageSubresourcePair imgpair, VkImageLayout &layout,
                            const VkImageAspectFlags aspectMask);

bool FindCmdBufLayout(layer_data const *device_data, GLOBAL_CB_NODE const *pCB, VkImage image, VkImageSubresource range,
                      IMAGE_CMD_BUF_LAYOUT_NODE &node);

bool FindGlobalLayout(layer_data *device_data, ImageSubresourcePair imgpair, VkImageLayout &layout);

bool FindLayouts(layer_data *device_data, VkImage image, std::vector<VkImageLayout> &layouts);

bool FindLayout(const std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &imageLayoutMap, ImageSubresourcePair imgpair,
                VkImageLayout &layout, const VkImageAspectFlags aspectMask);

bool FindLayout(const std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &imageLayoutMap, ImageSubresourcePair imgpair,
                VkImageLayout &layout);

void SetGlobalLayout(layer_data *device_data, ImageSubresourcePair imgpair, const VkImageLayout &layout);

void SetLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, ImageSubresourcePair imgpair, const IMAGE_CMD_BUF_LAYOUT_NODE &node);

void SetLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, ImageSubresourcePair imgpair, const VkImageLayout &layout);

void SetLayout(std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &imageLayoutMap, ImageSubresourcePair imgpair,
               VkImageLayout layout);

void SetImageViewLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, VkImageView imageView,
                        const VkImageLayout &layout);

bool VerifyFramebufferAndRenderPassLayouts(layer_data *dev_data, GLOBAL_CB_NODE *pCB, const VkRenderPassBeginInfo *pRenderPassBegin,
                                           const FRAMEBUFFER_STATE *framebuffer_state);

void TransitionAttachmentRefLayout(layer_data *dev_data, GLOBAL_CB_NODE *pCB, FRAMEBUFFER_STATE *pFramebuffer,
                                   VkAttachmentReference ref);

void TransitionSubpassLayouts(layer_data *, GLOBAL_CB_NODE *, const RENDER_PASS_STATE *, const int, FRAMEBUFFER_STATE *);

void TransitionBeginRenderPassLayouts(layer_data *, GLOBAL_CB_NODE *, const RENDER_PASS_STATE *, FRAMEBUFFER_STATE *);

bool ValidateImageAspectLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, const VkImageMemoryBarrier *mem_barrier,
                               uint32_t level, uint32_t layer, VkImageAspectFlags aspect);

void TransitionImageAspectLayout(layer_data *dev_data, GLOBAL_CB_NODE *pCB, const VkImageMemoryBarrier *mem_barrier, uint32_t level,
                                 uint32_t layer, VkImageAspectFlags aspect);

bool ValidateBarrierLayoutToImageUsage(layer_data *device_data, const VkImageMemoryBarrier *img_barrier, bool new_not_old,
                                       VkImageUsageFlags usage, const char *func_name);

bool ValidateBarriersToImages(layer_data *device_data, GLOBAL_CB_NODE const *cb_state, uint32_t imageMemoryBarrierCount,
                              const VkImageMemoryBarrier *pImageMemoryBarriers, const char *func_name);

void TransitionImageLayouts(layer_data *device_data, VkCommandBuffer cmdBuffer, uint32_t memBarrierCount,
                            const VkImageMemoryBarrier *pImgMemBarriers);

bool VerifySourceImageLayout(layer_data *dev_data, GLOBAL_CB_NODE *cb_node, VkImage srcImage, VkImageSubresourceLayers subLayers,
                             VkImageLayout srcImageLayout, UNIQUE_VALIDATION_ERROR_CODE msgCode);

bool VerifyDestImageLayout(layer_data *dev_data, GLOBAL_CB_NODE *cb_node, VkImage destImage, VkImageSubresourceLayers subLayers,
                           VkImageLayout destImageLayout, UNIQUE_VALIDATION_ERROR_CODE msgCode);

void TransitionFinalSubpassLayouts(layer_data *dev_data, GLOBAL_CB_NODE *pCB, const VkRenderPassBeginInfo *pRenderPassBegin,
                                   FRAMEBUFFER_STATE *framebuffer_state);

bool PreCallValidateCmdCopyImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                 IMAGE_STATE *dst_image_state, uint32_t region_count, const VkImageCopy *regions,
                                 VkImageLayout src_image_layout, VkImageLayout dst_image_layout);

bool PreCallValidateCmdClearAttachments(layer_data *device_data, VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                        const VkClearAttachment *pAttachments, uint32_t rectCount, const VkClearRect *pRects);

bool PreCallValidateCmdResolveImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                    IMAGE_STATE *dst_image_state, uint32_t regionCount, const VkImageResolve *pRegions);

void PreCallRecordCmdResolveImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                  IMAGE_STATE *dst_image_state);

bool PreCallValidateCmdBlitImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                 IMAGE_STATE *dst_image_state, uint32_t regionCount, const VkImageBlit *pRegions, VkFilter filter);

void PreCallRecordCmdBlitImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                               IMAGE_STATE *dst_image_state);

bool ValidateCmdBufImageLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB,
                                std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> const &globalImageLayoutMap,
                                std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &overlayLayoutMap);

void UpdateCmdBufImageLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB);

bool ValidateMaskBitsFromLayouts(core_validation::layer_data *device_data, VkCommandBuffer cmdBuffer,
                                 const VkAccessFlags &accessMask, const VkImageLayout &layout, const char *type);

bool ValidateLayoutVsAttachmentDescription(const debug_report_data *report_data, const VkImageLayout first_layout,
                                           const uint32_t attachment, const VkAttachmentDescription &attachment_description);

bool ValidateLayouts(core_validation::layer_data *dev_data, VkDevice device, const VkRenderPassCreateInfo *pCreateInfo);

bool ValidateMapImageLayouts(core_validation::layer_data *dev_data, VkDevice device, DEVICE_MEM_INFO const *mem_info,
                             VkDeviceSize offset, VkDeviceSize end_offset);

bool ValidateImageUsageFlags(layer_data *dev_data, IMAGE_STATE const *image_state, VkFlags desired, bool strict,
                             int32_t const msgCode, char const *func_name, char const *usage_string);

bool ValidateBufferUsageFlags(layer_data *dev_data, BUFFER_STATE const *buffer_state, VkFlags desired, bool strict,
                              int32_t const msgCode, char const *func_name, char const *usage_string);

bool PreCallValidateCreateBuffer(layer_data *dev_data, const VkBufferCreateInfo *pCreateInfo);

void PostCallRecordCreateBuffer(layer_data *device_data, const VkBufferCreateInfo *pCreateInfo, VkBuffer *pBuffer);

bool PreCallValidateCreateBufferView(layer_data *dev_data, const VkBufferViewCreateInfo *pCreateInfo);

void PostCallRecordCreateBufferView(layer_data *device_data, const VkBufferViewCreateInfo *pCreateInfo, VkBufferView *pView);

bool ValidateImageAspectMask(layer_data *device_data, VkImage image, VkFormat format, VkImageAspectFlags aspect_mask,
                             const char *func_name);

bool ValidateImageSubresourceRange(const layer_data *device_data, const IMAGE_STATE *image_state, const bool is_imageview_2d_array,
                                   const VkImageSubresourceRange &subresourceRange, const char *cmd_name, const char *param_name);

bool PreCallValidateCreateImageView(layer_data *device_data, const VkImageViewCreateInfo *create_info);

void PostCallRecordCreateImageView(layer_data *device_data, const VkImageViewCreateInfo *create_info, VkImageView view);

bool ValidateCopyBufferImageTransferGranularityRequirements(layer_data *device_data, const GLOBAL_CB_NODE *cb_node,
                                                            const IMAGE_STATE *img, const VkBufferImageCopy *region,
                                                            const uint32_t i, const char *function);

void PreCallRecordCmdCopyImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                               IMAGE_STATE *dst_image_state, uint32_t region_count, const VkImageCopy *regions,
                               VkImageLayout src_image_layout, VkImageLayout dst_image_layout);

bool PreCallValidateCmdCopyBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *src_buffer_state,
                                  BUFFER_STATE *dst_buffer_state);

void PreCallRecordCmdCopyBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *src_buffer_state,
                                BUFFER_STATE *dst_buffer_state);

bool PreCallValidateDestroyImageView(layer_data *device_data, VkImageView image_view, IMAGE_VIEW_STATE **image_view_state,
                                     VK_OBJECT *obj_struct);

void PostCallRecordDestroyImageView(layer_data *device_data, VkImageView image_view, IMAGE_VIEW_STATE *image_view_state,
                                    VK_OBJECT obj_struct);

bool PreCallValidateDestroyBuffer(layer_data *device_data, VkBuffer buffer, BUFFER_STATE **buffer_state, VK_OBJECT *obj_struct);

void PostCallRecordDestroyBuffer(layer_data *device_data, VkBuffer buffer, BUFFER_STATE *buffer_state, VK_OBJECT obj_struct);

bool PreCallValidateDestroyBufferView(layer_data *device_data, VkBufferView buffer_view, BUFFER_VIEW_STATE **buffer_view_state,
                                      VK_OBJECT *obj_struct);

void PostCallRecordDestroyBufferView(layer_data *device_data, VkBufferView buffer_view, BUFFER_VIEW_STATE *buffer_view_state,
                                     VK_OBJECT obj_struct);

bool PreCallValidateCmdFillBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *buffer_state);

void PreCallRecordCmdFillBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *buffer_state);

bool PreCallValidateCmdCopyImageToBuffer(layer_data *device_data, VkImageLayout srcImageLayout, GLOBAL_CB_NODE *cb_node,
                                         IMAGE_STATE *src_image_state, BUFFER_STATE *dst_buff_state, uint32_t regionCount,
                                         const VkBufferImageCopy *pRegions, const char *func_name);

void PreCallRecordCmdCopyImageToBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                       BUFFER_STATE *dst_buff_state, uint32_t region_count, const VkBufferImageCopy *regions,
                                       VkImageLayout src_image_layout);

bool PreCallValidateCmdCopyBufferToImage(layer_data *dev_data, VkImageLayout dstImageLayout, GLOBAL_CB_NODE *cb_node,
                                         BUFFER_STATE *src_buff_state, IMAGE_STATE *dst_image_state, uint32_t regionCount,
                                         const VkBufferImageCopy *pRegions, const char *func_name);

void PreCallRecordCmdCopyBufferToImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *src_buff_state,
                                       IMAGE_STATE *dst_image_state, uint32_t region_count, const VkBufferImageCopy *regions,
                                       VkImageLayout dst_image_layout);

bool PreCallValidateGetImageSubresourceLayout(layer_data *device_data, VkImage image, const VkImageSubresource *pSubresource);

#endif  // CORE_VALIDATION_BUFFER_VALIDATION_H_

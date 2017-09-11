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
 * Author: Dave Houlton <daveh@lunarg.com>
 */

// Allow use of STL min and max functions in Windows
#define NOMINMAX

#include <inttypes.h>
#include <sstream>
#include <string>

#include "vk_enum_string_helper.h"
#include "vk_layer_data.h"
#include "vk_layer_utils.h"
#include "vk_layer_logging.h"

#include "buffer_validation.h"

// TODO: remove on NDK update (r15 will probably have proper STL impl)
#ifdef __ANDROID__
namespace std {

template <typename T>
std::string to_string(T var) {
    std::ostringstream ss;
    ss << var;
    return ss.str();
}
}  // namespace std
#endif

void SetLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, ImageSubresourcePair imgpair, const VkImageLayout &layout) {
    if (pCB->imageLayoutMap.find(imgpair) != pCB->imageLayoutMap.end()) {
        pCB->imageLayoutMap[imgpair].layout = layout;
    } else {
        assert(imgpair.hasSubresource);
        IMAGE_CMD_BUF_LAYOUT_NODE node;
        if (!FindCmdBufLayout(device_data, pCB, imgpair.image, imgpair.subresource, node)) {
            node.initialLayout = layout;
        }
        SetLayout(device_data, pCB, imgpair, {node.initialLayout, layout});
    }
}
template <class OBJECT, class LAYOUT>
void SetLayout(layer_data *device_data, OBJECT *pObject, VkImage image, VkImageSubresource range, const LAYOUT &layout) {
    ImageSubresourcePair imgpair = {image, true, range};
    SetLayout(device_data, pObject, imgpair, layout, VK_IMAGE_ASPECT_COLOR_BIT);
    SetLayout(device_data, pObject, imgpair, layout, VK_IMAGE_ASPECT_DEPTH_BIT);
    SetLayout(device_data, pObject, imgpair, layout, VK_IMAGE_ASPECT_STENCIL_BIT);
    SetLayout(device_data, pObject, imgpair, layout, VK_IMAGE_ASPECT_METADATA_BIT);
}

template <class OBJECT, class LAYOUT>
void SetLayout(layer_data *device_data, OBJECT *pObject, ImageSubresourcePair imgpair, const LAYOUT &layout,
               VkImageAspectFlags aspectMask) {
    if (imgpair.subresource.aspectMask & aspectMask) {
        imgpair.subresource.aspectMask = aspectMask;
        SetLayout(device_data, pObject, imgpair, layout);
    }
}

// Set the layout in supplied map
void SetLayout(std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &imageLayoutMap, ImageSubresourcePair imgpair,
               VkImageLayout layout) {
    imageLayoutMap[imgpair].layout = layout;
}

bool FindLayoutVerifyNode(layer_data const *device_data, GLOBAL_CB_NODE const *pCB, ImageSubresourcePair imgpair,
                          IMAGE_CMD_BUF_LAYOUT_NODE &node, const VkImageAspectFlags aspectMask) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    if (!(imgpair.subresource.aspectMask & aspectMask)) {
        return false;
    }
    VkImageAspectFlags oldAspectMask = imgpair.subresource.aspectMask;
    imgpair.subresource.aspectMask = aspectMask;
    auto imgsubIt = pCB->imageLayoutMap.find(imgpair);
    if (imgsubIt == pCB->imageLayoutMap.end()) {
        return false;
    }
    if (node.layout != VK_IMAGE_LAYOUT_MAX_ENUM && node.layout != imgsubIt->second.layout) {
        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(imgpair.image),
                __LINE__, DRAWSTATE_INVALID_LAYOUT, "DS",
                "Cannot query for VkImage 0x%" PRIx64 " layout when combined aspect mask %d has multiple layout types: %s and %s",
                HandleToUint64(imgpair.image), oldAspectMask, string_VkImageLayout(node.layout),
                string_VkImageLayout(imgsubIt->second.layout));
    }
    if (node.initialLayout != VK_IMAGE_LAYOUT_MAX_ENUM && node.initialLayout != imgsubIt->second.initialLayout) {
        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(imgpair.image),
                __LINE__, DRAWSTATE_INVALID_LAYOUT, "DS",
                "Cannot query for VkImage 0x%" PRIx64
                " layout when combined aspect mask %d has multiple initial layout types: %s and %s",
                HandleToUint64(imgpair.image), oldAspectMask, string_VkImageLayout(node.initialLayout),
                string_VkImageLayout(imgsubIt->second.initialLayout));
    }
    node = imgsubIt->second;
    return true;
}

bool FindLayoutVerifyLayout(layer_data const *device_data, ImageSubresourcePair imgpair, VkImageLayout &layout,
                            const VkImageAspectFlags aspectMask) {
    if (!(imgpair.subresource.aspectMask & aspectMask)) {
        return false;
    }
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    VkImageAspectFlags oldAspectMask = imgpair.subresource.aspectMask;
    imgpair.subresource.aspectMask = aspectMask;
    auto imgsubIt = (*core_validation::GetImageLayoutMap(device_data)).find(imgpair);
    if (imgsubIt == (*core_validation::GetImageLayoutMap(device_data)).end()) {
        return false;
    }
    if (layout != VK_IMAGE_LAYOUT_MAX_ENUM && layout != imgsubIt->second.layout) {
        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(imgpair.image),
                __LINE__, DRAWSTATE_INVALID_LAYOUT, "DS",
                "Cannot query for VkImage 0x%" PRIx64 " layout when combined aspect mask %d has multiple layout types: %s and %s",
                HandleToUint64(imgpair.image), oldAspectMask, string_VkImageLayout(layout),
                string_VkImageLayout(imgsubIt->second.layout));
    }
    layout = imgsubIt->second.layout;
    return true;
}

// Find layout(s) on the command buffer level
bool FindCmdBufLayout(layer_data const *device_data, GLOBAL_CB_NODE const *pCB, VkImage image, VkImageSubresource range,
                      IMAGE_CMD_BUF_LAYOUT_NODE &node) {
    ImageSubresourcePair imgpair = {image, true, range};
    node = IMAGE_CMD_BUF_LAYOUT_NODE(VK_IMAGE_LAYOUT_MAX_ENUM, VK_IMAGE_LAYOUT_MAX_ENUM);
    FindLayoutVerifyNode(device_data, pCB, imgpair, node, VK_IMAGE_ASPECT_COLOR_BIT);
    FindLayoutVerifyNode(device_data, pCB, imgpair, node, VK_IMAGE_ASPECT_DEPTH_BIT);
    FindLayoutVerifyNode(device_data, pCB, imgpair, node, VK_IMAGE_ASPECT_STENCIL_BIT);
    FindLayoutVerifyNode(device_data, pCB, imgpair, node, VK_IMAGE_ASPECT_METADATA_BIT);
    if (node.layout == VK_IMAGE_LAYOUT_MAX_ENUM) {
        imgpair = {image, false, VkImageSubresource()};
        auto imgsubIt = pCB->imageLayoutMap.find(imgpair);
        if (imgsubIt == pCB->imageLayoutMap.end()) return false;
        // TODO: This is ostensibly a find function but it changes state here
        node = imgsubIt->second;
    }
    return true;
}

// Find layout(s) on the global level
bool FindGlobalLayout(layer_data *device_data, ImageSubresourcePair imgpair, VkImageLayout &layout) {
    layout = VK_IMAGE_LAYOUT_MAX_ENUM;
    FindLayoutVerifyLayout(device_data, imgpair, layout, VK_IMAGE_ASPECT_COLOR_BIT);
    FindLayoutVerifyLayout(device_data, imgpair, layout, VK_IMAGE_ASPECT_DEPTH_BIT);
    FindLayoutVerifyLayout(device_data, imgpair, layout, VK_IMAGE_ASPECT_STENCIL_BIT);
    FindLayoutVerifyLayout(device_data, imgpair, layout, VK_IMAGE_ASPECT_METADATA_BIT);
    if (layout == VK_IMAGE_LAYOUT_MAX_ENUM) {
        imgpair = {imgpair.image, false, VkImageSubresource()};
        auto imgsubIt = (*core_validation::GetImageLayoutMap(device_data)).find(imgpair);
        if (imgsubIt == (*core_validation::GetImageLayoutMap(device_data)).end()) return false;
        layout = imgsubIt->second.layout;
    }
    return true;
}

bool FindLayouts(layer_data *device_data, VkImage image, std::vector<VkImageLayout> &layouts) {
    auto sub_data = (*core_validation::GetImageSubresourceMap(device_data)).find(image);
    if (sub_data == (*core_validation::GetImageSubresourceMap(device_data)).end()) return false;
    auto image_state = GetImageState(device_data, image);
    if (!image_state) return false;
    bool ignoreGlobal = false;
    // TODO: Make this robust for >1 aspect mask. Now it will just say ignore potential errors in this case.
    if (sub_data->second.size() >= (image_state->createInfo.arrayLayers * image_state->createInfo.mipLevels + 1)) {
        ignoreGlobal = true;
    }
    for (auto imgsubpair : sub_data->second) {
        if (ignoreGlobal && !imgsubpair.hasSubresource) continue;
        auto img_data = (*core_validation::GetImageLayoutMap(device_data)).find(imgsubpair);
        if (img_data != (*core_validation::GetImageLayoutMap(device_data)).end()) {
            layouts.push_back(img_data->second.layout);
        }
    }
    return true;
}
bool FindLayout(const std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &imageLayoutMap, ImageSubresourcePair imgpair,
                VkImageLayout &layout, const VkImageAspectFlags aspectMask) {
    if (!(imgpair.subresource.aspectMask & aspectMask)) {
        return false;
    }
    imgpair.subresource.aspectMask = aspectMask;
    auto imgsubIt = imageLayoutMap.find(imgpair);
    if (imgsubIt == imageLayoutMap.end()) {
        return false;
    }
    layout = imgsubIt->second.layout;
    return true;
}

// find layout in supplied map
bool FindLayout(const std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> &imageLayoutMap, ImageSubresourcePair imgpair,
                VkImageLayout &layout) {
    layout = VK_IMAGE_LAYOUT_MAX_ENUM;
    FindLayout(imageLayoutMap, imgpair, layout, VK_IMAGE_ASPECT_COLOR_BIT);
    FindLayout(imageLayoutMap, imgpair, layout, VK_IMAGE_ASPECT_DEPTH_BIT);
    FindLayout(imageLayoutMap, imgpair, layout, VK_IMAGE_ASPECT_STENCIL_BIT);
    FindLayout(imageLayoutMap, imgpair, layout, VK_IMAGE_ASPECT_METADATA_BIT);
    if (layout == VK_IMAGE_LAYOUT_MAX_ENUM) {
        imgpair = {imgpair.image, false, VkImageSubresource()};
        auto imgsubIt = imageLayoutMap.find(imgpair);
        if (imgsubIt == imageLayoutMap.end()) return false;
        layout = imgsubIt->second.layout;
    }
    return true;
}

// Set the layout on the global level
void SetGlobalLayout(layer_data *device_data, ImageSubresourcePair imgpair, const VkImageLayout &layout) {
    VkImage &image = imgpair.image;
    (*core_validation::GetImageLayoutMap(device_data))[imgpair].layout = layout;
    auto &image_subresources = (*core_validation::GetImageSubresourceMap(device_data))[image];
    auto subresource = std::find(image_subresources.begin(), image_subresources.end(), imgpair);
    if (subresource == image_subresources.end()) {
        image_subresources.push_back(imgpair);
    }
}

// Set the layout on the cmdbuf level
void SetLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, ImageSubresourcePair imgpair, const IMAGE_CMD_BUF_LAYOUT_NODE &node) {
    pCB->imageLayoutMap[imgpair] = node;
}
// Set image layout for given VkImageSubresourceRange struct
void SetImageLayout(layer_data *device_data, GLOBAL_CB_NODE *cb_node, const IMAGE_STATE *image_state,
                    VkImageSubresourceRange image_subresource_range, const VkImageLayout &layout) {
    assert(image_state);
    for (uint32_t level_index = 0; level_index < image_subresource_range.levelCount; ++level_index) {
        uint32_t level = image_subresource_range.baseMipLevel + level_index;
        for (uint32_t layer_index = 0; layer_index < image_subresource_range.layerCount; layer_index++) {
            uint32_t layer = image_subresource_range.baseArrayLayer + layer_index;
            VkImageSubresource sub = {image_subresource_range.aspectMask, level, layer};
            // TODO: If ImageView was created with depth or stencil, transition both layouts as the aspectMask is ignored and both
            // are used. Verify that the extra implicit layout is OK for descriptor set layout validation
            if (image_subresource_range.aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
                if (FormatIsDepthAndStencil(image_state->createInfo.format)) {
                    sub.aspectMask |= (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
                }
            }
            SetLayout(device_data, cb_node, image_state->image, sub, layout);
        }
    }
}
// Set image layout for given VkImageSubresourceLayers struct
void SetImageLayout(layer_data *device_data, GLOBAL_CB_NODE *cb_node, const IMAGE_STATE *image_state,
                    VkImageSubresourceLayers image_subresource_layers, const VkImageLayout &layout) {
    // Transfer VkImageSubresourceLayers into VkImageSubresourceRange struct
    VkImageSubresourceRange image_subresource_range;
    image_subresource_range.aspectMask = image_subresource_layers.aspectMask;
    image_subresource_range.baseArrayLayer = image_subresource_layers.baseArrayLayer;
    image_subresource_range.layerCount = image_subresource_layers.layerCount;
    image_subresource_range.baseMipLevel = image_subresource_layers.mipLevel;
    image_subresource_range.levelCount = 1;
    SetImageLayout(device_data, cb_node, image_state, image_subresource_range, layout);
}
// Set image layout for all slices of an image view
void SetImageViewLayout(layer_data *device_data, GLOBAL_CB_NODE *cb_node, VkImageView imageView, const VkImageLayout &layout) {
    auto view_state = GetImageViewState(device_data, imageView);
    assert(view_state);

    SetImageLayout(device_data, cb_node, GetImageState(device_data, view_state->create_info.image),
                   view_state->create_info.subresourceRange, layout);
}

bool VerifyFramebufferAndRenderPassLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB,
                                           const VkRenderPassBeginInfo *pRenderPassBegin,
                                           const FRAMEBUFFER_STATE *framebuffer_state) {
    bool skip = false;
    auto const pRenderPassInfo = GetRenderPassState(device_data, pRenderPassBegin->renderPass)->createInfo.ptr();
    auto const &framebufferInfo = framebuffer_state->createInfo;
    const auto report_data = core_validation::GetReportData(device_data);
    if (pRenderPassInfo->attachmentCount != framebufferInfo.attachmentCount) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(pCB->commandBuffer), __LINE__, DRAWSTATE_INVALID_RENDERPASS, "DS",
                        "You cannot start a render pass using a framebuffer "
                        "with a different number of attachments.");
    }
    for (uint32_t i = 0; i < pRenderPassInfo->attachmentCount; ++i) {
        const VkImageView &image_view = framebufferInfo.pAttachments[i];
        auto view_state = GetImageViewState(device_data, image_view);
        assert(view_state);
        const VkImage &image = view_state->create_info.image;
        const VkImageSubresourceRange &subRange = view_state->create_info.subresourceRange;
        auto initial_layout = pRenderPassInfo->pAttachments[i].initialLayout;
        // TODO: Do not iterate over every possibility - consolidate where possible
        for (uint32_t j = 0; j < subRange.levelCount; j++) {
            uint32_t level = subRange.baseMipLevel + j;
            for (uint32_t k = 0; k < subRange.layerCount; k++) {
                uint32_t layer = subRange.baseArrayLayer + k;
                VkImageSubresource sub = {subRange.aspectMask, level, layer};
                IMAGE_CMD_BUF_LAYOUT_NODE node;
                if (!FindCmdBufLayout(device_data, pCB, image, sub, node)) {
                    // Missing layouts will be added during state update
                    continue;
                }
                if (initial_layout != VK_IMAGE_LAYOUT_UNDEFINED && initial_layout != node.layout) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, DRAWSTATE_INVALID_RENDERPASS, "DS",
                                    "You cannot start a render pass using attachment %u "
                                    "where the render pass initial layout is %s and the previous "
                                    "known layout of the attachment is %s. The layouts must match, or "
                                    "the render pass initial layout for the attachment must be "
                                    "VK_IMAGE_LAYOUT_UNDEFINED",
                                    i, string_VkImageLayout(initial_layout), string_VkImageLayout(node.layout));
                }
            }
        }
    }
    return skip;
}

void TransitionAttachmentRefLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, FRAMEBUFFER_STATE *pFramebuffer,
                                   VkAttachmentReference ref) {
    if (ref.attachment != VK_ATTACHMENT_UNUSED) {
        auto image_view = pFramebuffer->createInfo.pAttachments[ref.attachment];
        SetImageViewLayout(device_data, pCB, image_view, ref.layout);
    }
}

void TransitionSubpassLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB, const RENDER_PASS_STATE *render_pass_state,
                              const int subpass_index, FRAMEBUFFER_STATE *framebuffer_state) {
    assert(render_pass_state);

    if (framebuffer_state) {
        auto const &subpass = render_pass_state->createInfo.pSubpasses[subpass_index];
        for (uint32_t j = 0; j < subpass.inputAttachmentCount; ++j) {
            TransitionAttachmentRefLayout(device_data, pCB, framebuffer_state, subpass.pInputAttachments[j]);
        }
        for (uint32_t j = 0; j < subpass.colorAttachmentCount; ++j) {
            TransitionAttachmentRefLayout(device_data, pCB, framebuffer_state, subpass.pColorAttachments[j]);
        }
        if (subpass.pDepthStencilAttachment) {
            TransitionAttachmentRefLayout(device_data, pCB, framebuffer_state, *subpass.pDepthStencilAttachment);
        }
    }
}

bool ValidateImageAspectLayout(layer_data *device_data, GLOBAL_CB_NODE const *pCB, const VkImageMemoryBarrier *mem_barrier,
                               uint32_t level, uint32_t layer, VkImageAspectFlags aspect) {
    if (!(mem_barrier->subresourceRange.aspectMask & aspect)) {
        return false;
    }
    VkImageSubresource sub = {aspect, level, layer};
    IMAGE_CMD_BUF_LAYOUT_NODE node;
    if (!FindCmdBufLayout(device_data, pCB, mem_barrier->image, sub, node)) {
        return false;
    }
    bool skip = false;
    if (mem_barrier->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        // TODO: Set memory invalid which is in mem_tracker currently
    } else if (node.layout != mem_barrier->oldLayout) {
        skip |=
            log_msg(core_validation::GetReportData(device_data), VK_DEBUG_REPORT_ERROR_BIT_EXT,
                    VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, HandleToUint64(pCB->commandBuffer), __LINE__,
                    DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                    "For image 0x%" PRIxLEAST64 " you cannot transition the layout of aspect %d from %s when current layout is %s.",
                    HandleToUint64(mem_barrier->image), aspect, string_VkImageLayout(mem_barrier->oldLayout),
                    string_VkImageLayout(node.layout));
    }
    return skip;
}

// Transition the layout state for renderpass attachments based on the BeginRenderPass() call. This includes:
// 1. Transition into initialLayout state
// 2. Transition from initialLayout to layout used in subpass 0
void TransitionBeginRenderPassLayouts(layer_data *device_data, GLOBAL_CB_NODE *cb_state, const RENDER_PASS_STATE *render_pass_state,
                                      FRAMEBUFFER_STATE *framebuffer_state) {
    // First transition into initialLayout
    auto const rpci = render_pass_state->createInfo.ptr();
    for (uint32_t i = 0; i < rpci->attachmentCount; ++i) {
        VkImageView image_view = framebuffer_state->createInfo.pAttachments[i];
        SetImageViewLayout(device_data, cb_state, image_view, rpci->pAttachments[i].initialLayout);
    }
    // Now transition for first subpass (index 0)
    TransitionSubpassLayouts(device_data, cb_state, render_pass_state, 0, framebuffer_state);
}

void TransitionImageAspectLayout(layer_data *device_data, GLOBAL_CB_NODE *pCB, const VkImageMemoryBarrier *mem_barrier,
                                 uint32_t level, uint32_t layer, VkImageAspectFlags aspect) {
    if (!(mem_barrier->subresourceRange.aspectMask & aspect)) {
        return;
    }
    VkImageSubresource sub = {aspect, level, layer};
    IMAGE_CMD_BUF_LAYOUT_NODE node;
    if (!FindCmdBufLayout(device_data, pCB, mem_barrier->image, sub, node)) {
        SetLayout(device_data, pCB, mem_barrier->image, sub,
                  IMAGE_CMD_BUF_LAYOUT_NODE(mem_barrier->oldLayout, mem_barrier->newLayout));
        return;
    }
    if (mem_barrier->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        // TODO: Set memory invalid
    }
    SetLayout(device_data, pCB, mem_barrier->image, sub, mem_barrier->newLayout);
}

bool VerifyAspectsPresent(VkImageAspectFlags aspect_mask, VkFormat format) {
    if ((aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
        if (!FormatIsColor(format)) return false;
    }
    if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
        if (!FormatHasDepth(format)) return false;
    }
    if ((aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) {
        if (!FormatHasStencil(format)) return false;
    }
    return true;
}

// Verify an ImageMemoryBarrier's old/new ImageLayouts are compatible with the Image's ImageUsageFlags.
bool ValidateBarrierLayoutToImageUsage(layer_data *device_data, const VkImageMemoryBarrier *img_barrier, bool new_not_old,
                                       VkImageUsageFlags usage_flags, const char *func_name) {
    const auto report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    const VkImageLayout layout = (new_not_old) ? img_barrier->newLayout : img_barrier->oldLayout;
    UNIQUE_VALIDATION_ERROR_CODE msg_code = VALIDATION_ERROR_UNDEFINED;  // sentinel value meaning "no error"

    switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            if ((usage_flags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
                msg_code = VALIDATION_ERROR_0a000970;
            }
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            if ((usage_flags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0) {
                msg_code = VALIDATION_ERROR_0a000972;
            }
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            if ((usage_flags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0) {
                msg_code = VALIDATION_ERROR_0a000974;
            }
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            if ((usage_flags & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) == 0) {
                msg_code = VALIDATION_ERROR_0a000976;
            }
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            if ((usage_flags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
                msg_code = VALIDATION_ERROR_0a000978;
            }
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            if ((usage_flags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
                msg_code = VALIDATION_ERROR_0a00097a;
            }
            break;
        default:
            // Other VkImageLayout values do not have VUs defined in this context.
            break;
    }

    if (msg_code != VALIDATION_ERROR_UNDEFINED) {
        skip |=
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                    HandleToUint64(img_barrier->image), __LINE__, msg_code, "DS",
                    "%s: Image barrier 0x%p %sLayout=%s is not compatible with image 0x%" PRIx64 " usage flags 0x%" PRIx32 ". %s",
                    func_name, img_barrier, ((new_not_old) ? "new" : "old"), string_VkImageLayout(layout),
                    HandleToUint64(img_barrier->image), usage_flags, validation_error_map[msg_code]);
    }
    return skip;
}

// Verify image barriers are compatible with the images they reference.
bool ValidateBarriersToImages(layer_data *device_data, GLOBAL_CB_NODE const *cb_state, uint32_t imageMemoryBarrierCount,
                              const VkImageMemoryBarrier *pImageMemoryBarriers, const char *func_name) {
    bool skip = false;

    for (uint32_t i = 0; i < imageMemoryBarrierCount; ++i) {
        auto img_barrier = &pImageMemoryBarriers[i];
        if (!img_barrier) continue;

        auto image_state = GetImageState(device_data, img_barrier->image);
        if (image_state) {
            VkImageUsageFlags usage_flags = image_state->createInfo.usage;
            skip |= ValidateBarrierLayoutToImageUsage(device_data, img_barrier, false, usage_flags, func_name);
            skip |= ValidateBarrierLayoutToImageUsage(device_data, img_barrier, true, usage_flags, func_name);

            // Make sure layout is able to be transitioned, currently only presented shared presentable images are locked
            if (image_state->layout_locked) {
                // TODO: Add unique id for error when available
                skip |= log_msg(
                    core_validation::GetReportData(device_data), VK_DEBUG_REPORT_ERROR_BIT_EXT,
                    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, 0, "DS",
                    "Attempting to transition shared presentable image 0x%" PRIxLEAST64
                    " from layout %s to layout %s, but image has already been presented and cannot have its layout transitioned.",
                    reinterpret_cast<const uint64_t &>(img_barrier->image), string_VkImageLayout(img_barrier->oldLayout),
                    string_VkImageLayout(img_barrier->newLayout));
            }
        }

        VkImageCreateInfo *image_create_info = &(GetImageState(device_data, img_barrier->image)->createInfo);
        // For a Depth/Stencil image both aspects MUST be set
        if (FormatIsDepthAndStencil(image_create_info->format)) {
            auto const aspect_mask = img_barrier->subresourceRange.aspectMask;
            auto const ds_mask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            if ((aspect_mask & ds_mask) != (ds_mask)) {
                skip |=
                    log_msg(core_validation::GetReportData(device_data), VK_DEBUG_REPORT_ERROR_BIT_EXT,
                            VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(img_barrier->image), __LINE__,
                            VALIDATION_ERROR_0a00096e, "DS",
                            "%s: Image barrier 0x%p references image 0x%" PRIx64
                            " of format %s that must have the depth and stencil aspects set, but its "
                            "aspectMask is 0x%" PRIx32 ". %s",
                            func_name, img_barrier, HandleToUint64(img_barrier->image), string_VkFormat(image_create_info->format),
                            aspect_mask, validation_error_map[VALIDATION_ERROR_0a00096e]);
            }
        }
        uint32_t level_count = ResolveRemainingLevels(&img_barrier->subresourceRange, image_create_info->mipLevels);
        uint32_t layer_count = ResolveRemainingLayers(&img_barrier->subresourceRange, image_create_info->arrayLayers);

        for (uint32_t j = 0; j < level_count; j++) {
            uint32_t level = img_barrier->subresourceRange.baseMipLevel + j;
            for (uint32_t k = 0; k < layer_count; k++) {
                uint32_t layer = img_barrier->subresourceRange.baseArrayLayer + k;
                skip |= ValidateImageAspectLayout(device_data, cb_state, img_barrier, level, layer, VK_IMAGE_ASPECT_COLOR_BIT);
                skip |= ValidateImageAspectLayout(device_data, cb_state, img_barrier, level, layer, VK_IMAGE_ASPECT_DEPTH_BIT);
                skip |= ValidateImageAspectLayout(device_data, cb_state, img_barrier, level, layer, VK_IMAGE_ASPECT_STENCIL_BIT);
                skip |= ValidateImageAspectLayout(device_data, cb_state, img_barrier, level, layer, VK_IMAGE_ASPECT_METADATA_BIT);
            }
        }
    }
    return skip;
}

void TransitionImageLayouts(layer_data *device_data, VkCommandBuffer cmdBuffer, uint32_t memBarrierCount,
                            const VkImageMemoryBarrier *pImgMemBarriers) {
    GLOBAL_CB_NODE *pCB = GetCBNode(device_data, cmdBuffer);

    for (uint32_t i = 0; i < memBarrierCount; ++i) {
        auto mem_barrier = &pImgMemBarriers[i];
        if (!mem_barrier) continue;

        VkImageCreateInfo *image_create_info = &(GetImageState(device_data, mem_barrier->image)->createInfo);
        uint32_t level_count = ResolveRemainingLevels(&mem_barrier->subresourceRange, image_create_info->mipLevels);
        uint32_t layer_count = ResolveRemainingLayers(&mem_barrier->subresourceRange, image_create_info->arrayLayers);

        for (uint32_t j = 0; j < level_count; j++) {
            uint32_t level = mem_barrier->subresourceRange.baseMipLevel + j;
            for (uint32_t k = 0; k < layer_count; k++) {
                uint32_t layer = mem_barrier->subresourceRange.baseArrayLayer + k;
                TransitionImageAspectLayout(device_data, pCB, mem_barrier, level, layer, VK_IMAGE_ASPECT_COLOR_BIT);
                TransitionImageAspectLayout(device_data, pCB, mem_barrier, level, layer, VK_IMAGE_ASPECT_DEPTH_BIT);
                TransitionImageAspectLayout(device_data, pCB, mem_barrier, level, layer, VK_IMAGE_ASPECT_STENCIL_BIT);
                TransitionImageAspectLayout(device_data, pCB, mem_barrier, level, layer, VK_IMAGE_ASPECT_METADATA_BIT);
            }
        }
    }
}

bool VerifyImageLayout(layer_data const *device_data, GLOBAL_CB_NODE const *cb_node, IMAGE_STATE *image_state,
                       VkImageSubresourceLayers subLayers, VkImageLayout explicit_layout, VkImageLayout optimal_layout,
                       const char *caller, UNIQUE_VALIDATION_ERROR_CODE msg_code, bool *error) {
    const auto report_data = core_validation::GetReportData(device_data);
    const auto image = image_state->image;
    bool skip = false;

    for (uint32_t i = 0; i < subLayers.layerCount; ++i) {
        uint32_t layer = i + subLayers.baseArrayLayer;
        VkImageSubresource sub = {subLayers.aspectMask, subLayers.mipLevel, layer};
        IMAGE_CMD_BUF_LAYOUT_NODE node;
        if (FindCmdBufLayout(device_data, cb_node, image, sub, node)) {
            if (node.layout != explicit_layout) {
                *error = true;
                // TODO: Improve log message in the next pass
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                "%s: Cannot use image 0x%" PRIxLEAST64
                                " with specific layout %s that doesn't match the actual current layout %s.",
                                caller, HandleToUint64(image), string_VkImageLayout(explicit_layout),
                                string_VkImageLayout(node.layout));
            }
        }
    }
    // If optimal_layout is not UNDEFINED, check that layout matches optimal for this case
    if ((VK_IMAGE_LAYOUT_UNDEFINED != optimal_layout) && (explicit_layout != optimal_layout)) {
        if (VK_IMAGE_LAYOUT_GENERAL == explicit_layout) {
            if (image_state->createInfo.tiling != VK_IMAGE_TILING_LINEAR) {
                // LAYOUT_GENERAL is allowed, but may not be performance optimal, flag as perf warning.
                skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                                VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, HandleToUint64(cb_node->commandBuffer), __LINE__,
                                DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                "%s: For optimal performance image 0x%" PRIxLEAST64 " layout should be %s instead of GENERAL.",
                                caller, HandleToUint64(image), string_VkImageLayout(optimal_layout));
            }
        } else if (GetDeviceExtensions(device_data)->vk_khr_shared_presentable_image) {
            if (image_state->shared_presentable) {
                if (VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR != explicit_layout) {
                    // TODO: Add unique error id when available.
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, msg_code, "DS",
                                    "Layout for shared presentable image is %s but must be VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR.",
                                    string_VkImageLayout(optimal_layout));
                }
            }
        } else {
            *error = true;
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, msg_code, "DS",
                            "%s: Layout for image 0x%" PRIxLEAST64 " is %s but can only be %s or VK_IMAGE_LAYOUT_GENERAL. %s",
                            caller, HandleToUint64(image), string_VkImageLayout(explicit_layout),
                            string_VkImageLayout(optimal_layout), validation_error_map[msg_code]);
        }
    }
    return skip;
}

void TransitionFinalSubpassLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB, const VkRenderPassBeginInfo *pRenderPassBegin,
                                   FRAMEBUFFER_STATE *framebuffer_state) {
    auto renderPass = GetRenderPassState(device_data, pRenderPassBegin->renderPass);
    if (!renderPass) return;

    const VkRenderPassCreateInfo *pRenderPassInfo = renderPass->createInfo.ptr();
    if (framebuffer_state) {
        for (uint32_t i = 0; i < pRenderPassInfo->attachmentCount; ++i) {
            auto image_view = framebuffer_state->createInfo.pAttachments[i];
            SetImageViewLayout(device_data, pCB, image_view, pRenderPassInfo->pAttachments[i].finalLayout);
        }
    }
}

bool PreCallValidateCreateImage(layer_data *device_data, const VkImageCreateInfo *pCreateInfo,
                                const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    if (pCreateInfo->format == VK_FORMAT_UNDEFINED) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0075e, "IMAGE", "vkCreateImage: VkFormat for image must not be VK_FORMAT_UNDEFINED. %s",
                        validation_error_map[VALIDATION_ERROR_09e0075e]);

        return skip;
    }

    const VkFormatProperties *properties = GetFormatProperties(device_data, pCreateInfo->format);

    if ((pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) && (properties->linearTilingFeatures == 0)) {
        std::stringstream ss;
        ss << "vkCreateImage format parameter (" << string_VkFormat(pCreateInfo->format) << ") is an unsupported format";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e007a2, "IMAGE", "%s. %s", ss.str().c_str(),
                        validation_error_map[VALIDATION_ERROR_09e007a2]);

        return skip;
    }

    if ((pCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL) && (properties->optimalTilingFeatures == 0)) {
        std::stringstream ss;
        ss << "vkCreateImage format parameter (" << string_VkFormat(pCreateInfo->format) << ") is an unsupported format";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e007ac, "IMAGE", "%s. %s", ss.str().c_str(),
                        validation_error_map[VALIDATION_ERROR_09e007ac]);

        return skip;
    }

    // Validate that format supports usage as color attachment
    if (pCreateInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        if ((pCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL) &&
            ((properties->optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) == 0)) {
            std::stringstream ss;
            ss << "vkCreateImage: VkFormat for TILING_OPTIMAL image (" << string_VkFormat(pCreateInfo->format)
               << ") does not support requested Image usage type VK_IMAGE_USAGE_COLOR_ATTACHMENT";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_09e007b2, "IMAGE", "%s. %s", ss.str().c_str(),
                            validation_error_map[VALIDATION_ERROR_09e007b2]);
        }
        if ((pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) &&
            ((properties->linearTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) == 0)) {
            std::stringstream ss;
            ss << "vkCreateImage: VkFormat for TILING_LINEAR image (" << string_VkFormat(pCreateInfo->format)
               << ") does not support requested Image usage type VK_IMAGE_USAGE_COLOR_ATTACHMENT";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_09e007a8, "IMAGE", "%s. %s", ss.str().c_str(),
                            validation_error_map[VALIDATION_ERROR_09e007a8]);
        }
    }

    // Validate that format supports usage as depth/stencil attachment
    if (pCreateInfo->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        if ((pCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL) &&
            ((properties->optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0)) {
            std::stringstream ss;
            ss << "vkCreateImage: VkFormat for TILING_OPTIMAL image (" << string_VkFormat(pCreateInfo->format)
               << ") does not support requested Image usage type VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_09e007b4, "IMAGE", "%s. %s", ss.str().c_str(),
                            validation_error_map[VALIDATION_ERROR_09e007b4]);
        }
        if ((pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR) &&
            ((properties->linearTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0)) {
            std::stringstream ss;
            ss << "vkCreateImage: VkFormat for TILING_LINEAR image (" << string_VkFormat(pCreateInfo->format)
               << ") does not support requested Image usage type VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_09e007aa, "IMAGE", "%s. %s", ss.str().c_str(),
                            validation_error_map[VALIDATION_ERROR_09e007aa]);
        }
    }

    const VkImageFormatProperties *ImageFormatProperties = GetImageFormatProperties(
        device_data, pCreateInfo->format, pCreateInfo->imageType, pCreateInfo->tiling, pCreateInfo->usage, pCreateInfo->flags);

    VkDeviceSize imageGranularity = GetPhysicalDeviceProperties(device_data)->limits.bufferImageGranularity;
    imageGranularity = imageGranularity == 1 ? 0 : imageGranularity;
    // TODO : This is also covering 2918 & 2919. Break out into separate checks
    if ((pCreateInfo->extent.width <= 0) || (pCreateInfo->extent.height <= 0) || (pCreateInfo->extent.depth <= 0)) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e007b8, "Image",
                        "CreateImage extent is 0 for at least one required dimension for image: "
                        "Width = %d Height = %d Depth = %d. %s",
                        pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->extent.depth,
                        validation_error_map[VALIDATION_ERROR_09e007b8]);
    }

    // TODO: VALIDATION_ERROR_09e00770 VALIDATION_ERROR_09e00772 VALIDATION_ERROR_09e00776 VALIDATION_ERROR_09e0076e
    // All these extent-related VUs should be checked here
    if ((pCreateInfo->extent.depth > ImageFormatProperties->maxExtent.depth) ||
        (pCreateInfo->extent.width > ImageFormatProperties->maxExtent.width) ||
        (pCreateInfo->extent.height > ImageFormatProperties->maxExtent.height)) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                        IMAGE_INVALID_FORMAT_LIMITS_VIOLATION, "Image",
                        "CreateImage extents exceed allowable limits for format: "
                        "Width = %d Height = %d Depth = %d:  Limits for Width = %d Height = %d Depth = %d for format %s.",
                        pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->extent.depth,
                        ImageFormatProperties->maxExtent.width, ImageFormatProperties->maxExtent.height,
                        ImageFormatProperties->maxExtent.depth, string_VkFormat(pCreateInfo->format));
    }

    uint64_t totalSize =
        ((uint64_t)pCreateInfo->extent.width * (uint64_t)pCreateInfo->extent.height * (uint64_t)pCreateInfo->extent.depth *
             (uint64_t)pCreateInfo->arrayLayers * (uint64_t)pCreateInfo->samples * (uint64_t)FormatSize(pCreateInfo->format) +
         (uint64_t)imageGranularity) &
        ~(uint64_t)imageGranularity;

    if (totalSize > ImageFormatProperties->maxResourceSize) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                        IMAGE_INVALID_FORMAT_LIMITS_VIOLATION, "Image",
                        "CreateImage resource size exceeds allowable maximum "
                        "Image resource size = 0x%" PRIxLEAST64 ", maximum resource size = 0x%" PRIxLEAST64 " ",
                        totalSize, ImageFormatProperties->maxResourceSize);
    }

    // TODO: VALIDATION_ERROR_09e0077e
    if (pCreateInfo->mipLevels > ImageFormatProperties->maxMipLevels) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                        IMAGE_INVALID_FORMAT_LIMITS_VIOLATION, "Image",
                        "CreateImage mipLevels=%d exceeds allowable maximum supported by format of %d", pCreateInfo->mipLevels,
                        ImageFormatProperties->maxMipLevels);
    }

    if (pCreateInfo->arrayLayers > ImageFormatProperties->maxArrayLayers) {
        skip |=
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                    VALIDATION_ERROR_09e00780, "Image",
                    "CreateImage arrayLayers=%d exceeds allowable maximum supported by format of %d. %s", pCreateInfo->arrayLayers,
                    ImageFormatProperties->maxArrayLayers, validation_error_map[VALIDATION_ERROR_09e00780]);
    }

    if ((pCreateInfo->samples & ImageFormatProperties->sampleCounts) == 0) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0078e, "Image", "CreateImage samples %s is not supported by format 0x%.8X. %s",
                        string_VkSampleCountFlagBits(pCreateInfo->samples), ImageFormatProperties->sampleCounts,
                        validation_error_map[VALIDATION_ERROR_09e0078e]);
    }

    if (pCreateInfo->initialLayout != VK_IMAGE_LAYOUT_UNDEFINED && pCreateInfo->initialLayout != VK_IMAGE_LAYOUT_PREINITIALIZED) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e0b801, "Image",
                        "vkCreateImage parameter, pCreateInfo->initialLayout, must be VK_IMAGE_LAYOUT_UNDEFINED or "
                        "VK_IMAGE_LAYOUT_PREINITIALIZED. %s",
                        validation_error_map[VALIDATION_ERROR_09e0b801]);
    }

    if ((pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) && (!GetEnabledFeatures(device_data)->sparseBinding)) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_09e00792, "DS",
                        "vkCreateImage(): the sparseBinding device feature is disabled: Images cannot be created with the "
                        "VK_IMAGE_CREATE_SPARSE_BINDING_BIT set. %s",
                        validation_error_map[VALIDATION_ERROR_09e00792]);
    }

    if ((pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_ALIASED_BIT) && (!GetEnabledFeatures(device_data)->sparseResidencyAliased)) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        DRAWSTATE_INVALID_FEATURE, "DS",
                        "vkCreateImage(): the sparseResidencyAliased device feature is disabled: Images cannot be created with the "
                        "VK_IMAGE_CREATE_SPARSE_ALIASED_BIT set.");
    }

    return skip;
}

void PostCallRecordCreateImage(layer_data *device_data, const VkImageCreateInfo *pCreateInfo, VkImage *pImage) {
    IMAGE_LAYOUT_NODE image_state;
    image_state.layout = pCreateInfo->initialLayout;
    image_state.format = pCreateInfo->format;
    GetImageMap(device_data)->insert(std::make_pair(*pImage, std::unique_ptr<IMAGE_STATE>(new IMAGE_STATE(*pImage, pCreateInfo))));
    ImageSubresourcePair subpair{*pImage, false, VkImageSubresource()};
    (*core_validation::GetImageSubresourceMap(device_data))[*pImage].push_back(subpair);
    (*core_validation::GetImageLayoutMap(device_data))[subpair] = image_state;
}

bool PreCallValidateDestroyImage(layer_data *device_data, VkImage image, IMAGE_STATE **image_state, VK_OBJECT *obj_struct) {
    const CHECK_DISABLED *disabled = core_validation::GetDisables(device_data);
    *image_state = core_validation::GetImageState(device_data, image);
    *obj_struct = {HandleToUint64(image), kVulkanObjectTypeImage};
    if (disabled->destroy_image) return false;
    bool skip = false;
    if (*image_state) {
        skip |= core_validation::ValidateObjectNotInUse(device_data, *image_state, *obj_struct, VALIDATION_ERROR_252007d0);
    }
    return skip;
}

void PostCallRecordDestroyImage(layer_data *device_data, VkImage image, IMAGE_STATE *image_state, VK_OBJECT obj_struct) {
    core_validation::invalidateCommandBuffers(device_data, image_state->cb_bindings, obj_struct);
    // Clean up memory mapping, bindings and range references for image
    for (auto mem_binding : image_state->GetBoundMemory()) {
        auto mem_info = core_validation::GetMemObjInfo(device_data, mem_binding);
        if (mem_info) {
            core_validation::RemoveImageMemoryRange(obj_struct.handle, mem_info);
        }
    }
    core_validation::ClearMemoryObjectBindings(device_data, obj_struct.handle, kVulkanObjectTypeImage);
    // Remove image from imageMap
    core_validation::GetImageMap(device_data)->erase(image);
    std::unordered_map<VkImage, std::vector<ImageSubresourcePair>> *imageSubresourceMap =
        core_validation::GetImageSubresourceMap(device_data);

    const auto &sub_entry = imageSubresourceMap->find(image);
    if (sub_entry != imageSubresourceMap->end()) {
        for (const auto &pair : sub_entry->second) {
            core_validation::GetImageLayoutMap(device_data)->erase(pair);
        }
        imageSubresourceMap->erase(sub_entry);
    }
}

bool ValidateImageAttributes(layer_data *device_data, IMAGE_STATE *image_state, VkImageSubresourceRange range) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    if (range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT) {
        char const str[] = "vkCmdClearColorImage aspectMasks for all subresource ranges must be set to VK_IMAGE_ASPECT_COLOR_BIT";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE", str);
    }

    if (FormatIsDepthOrStencil(image_state->createInfo.format)) {
        char const str[] = "vkCmdClearColorImage called with depth/stencil image.";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_1880000e, "IMAGE", "%s. %s", str,
                        validation_error_map[VALIDATION_ERROR_1880000e]);
    } else if (FormatIsCompressed(image_state->createInfo.format)) {
        char const str[] = "vkCmdClearColorImage called with compressed image.";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_1880000e, "IMAGE", "%s. %s", str,
                        validation_error_map[VALIDATION_ERROR_1880000e]);
    }

    if (!(image_state->createInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        char const str[] = "vkCmdClearColorImage called with image created without VK_IMAGE_USAGE_TRANSFER_DST_BIT.";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_18800004, "IMAGE", "%s. %s", str,
                        validation_error_map[VALIDATION_ERROR_18800004]);
    }
    return skip;
}

uint32_t ResolveRemainingLevels(const VkImageSubresourceRange *range, uint32_t mip_levels) {
    // Return correct number of mip levels taking into account VK_REMAINING_MIP_LEVELS
    uint32_t mip_level_count = range->levelCount;
    if (range->levelCount == VK_REMAINING_MIP_LEVELS) {
        mip_level_count = mip_levels - range->baseMipLevel;
    }
    return mip_level_count;
}

uint32_t ResolveRemainingLayers(const VkImageSubresourceRange *range, uint32_t layers) {
    // Return correct number of layers taking into account VK_REMAINING_ARRAY_LAYERS
    uint32_t array_layer_count = range->layerCount;
    if (range->layerCount == VK_REMAINING_ARRAY_LAYERS) {
        array_layer_count = layers - range->baseArrayLayer;
    }
    return array_layer_count;
}

bool VerifyClearImageLayout(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *image_state,
                            VkImageSubresourceRange range, VkImageLayout dest_image_layout, const char *func_name) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    uint32_t level_count = ResolveRemainingLevels(&range, image_state->createInfo.mipLevels);
    uint32_t layer_count = ResolveRemainingLayers(&range, image_state->createInfo.arrayLayers);

    if (dest_image_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        if (dest_image_layout == VK_IMAGE_LAYOUT_GENERAL) {
            if (image_state->createInfo.tiling != VK_IMAGE_TILING_LINEAR) {
                // LAYOUT_GENERAL is allowed, but may not be performance optimal, flag as perf warning.
                skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                "%s: Layout for cleared image should be TRANSFER_DST_OPTIMAL instead of GENERAL.", func_name);
            }
        } else if (VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR == dest_image_layout) {
            if (!GetDeviceExtensions(device_data)->vk_khr_shared_presentable_image) {
                // TODO: Add unique error id when available.
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, 0, "DS",
                                "Must enable VK_KHR_shared_presentable_image extension before creating images with a layout type "
                                "of VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR.");

            } else {
                if (image_state->shared_presentable) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, 0, "DS",
                        "Layout for shared presentable cleared image is %s but can only be VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR.",
                        string_VkImageLayout(dest_image_layout));
                }
            }
        } else {
            UNIQUE_VALIDATION_ERROR_CODE error_code = VALIDATION_ERROR_1880000a;
            if (strcmp(func_name, "vkCmdClearDepthStencilImage()") == 0) {
                error_code = VALIDATION_ERROR_18a00018;
            } else {
                assert(strcmp(func_name, "vkCmdClearColorImage()") == 0);
            }
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, error_code, "DS",
                            "%s: Layout for cleared image is %s but can only be "
                            "TRANSFER_DST_OPTIMAL or GENERAL. %s",
                            func_name, string_VkImageLayout(dest_image_layout), validation_error_map[error_code]);
        }
    }

    for (uint32_t level_index = 0; level_index < level_count; ++level_index) {
        uint32_t level = level_index + range.baseMipLevel;
        for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            uint32_t layer = layer_index + range.baseArrayLayer;
            VkImageSubresource sub = {range.aspectMask, level, layer};
            IMAGE_CMD_BUF_LAYOUT_NODE node;
            if (FindCmdBufLayout(device_data, cb_node, image_state->image, sub, node)) {
                if (node.layout != dest_image_layout) {
                    UNIQUE_VALIDATION_ERROR_CODE error_code = VALIDATION_ERROR_18800008;
                    if (strcmp(func_name, "vkCmdClearDepthStencilImage()") == 0) {
                        error_code = VALIDATION_ERROR_18a00016;
                    } else {
                        assert(strcmp(func_name, "vkCmdClearColorImage()") == 0);
                    }
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, 0,
                                    __LINE__, error_code, "DS",
                                    "%s: Cannot clear an image whose layout is %s and "
                                    "doesn't match the current layout %s. %s",
                                    func_name, string_VkImageLayout(dest_image_layout), string_VkImageLayout(node.layout),
                                    validation_error_map[error_code]);
                }
            }
        }
    }

    return skip;
}

void RecordClearImageLayout(layer_data *device_data, GLOBAL_CB_NODE *cb_node, VkImage image, VkImageSubresourceRange range,
                            VkImageLayout dest_image_layout) {
    VkImageCreateInfo *image_create_info = &(GetImageState(device_data, image)->createInfo);
    uint32_t level_count = ResolveRemainingLevels(&range, image_create_info->mipLevels);
    uint32_t layer_count = ResolveRemainingLayers(&range, image_create_info->arrayLayers);

    for (uint32_t level_index = 0; level_index < level_count; ++level_index) {
        uint32_t level = level_index + range.baseMipLevel;
        for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            uint32_t layer = layer_index + range.baseArrayLayer;
            VkImageSubresource sub = {range.aspectMask, level, layer};
            IMAGE_CMD_BUF_LAYOUT_NODE node;
            if (!FindCmdBufLayout(device_data, cb_node, image, sub, node)) {
                SetLayout(device_data, cb_node, image, sub, IMAGE_CMD_BUF_LAYOUT_NODE(dest_image_layout, dest_image_layout));
            }
        }
    }
}

bool PreCallValidateCmdClearColorImage(layer_data *dev_data, VkCommandBuffer commandBuffer, VkImage image,
                                       VkImageLayout imageLayout, uint32_t rangeCount, const VkImageSubresourceRange *pRanges) {
    bool skip = false;
    // TODO : Verify memory is in VK_IMAGE_STATE_CLEAR state
    auto cb_node = GetCBNode(dev_data, commandBuffer);
    auto image_state = GetImageState(dev_data, image);
    if (cb_node && image_state) {
        skip |= ValidateMemoryIsBoundToImage(dev_data, image_state, "vkCmdClearColorImage()", VALIDATION_ERROR_18800006);
        skip |= ValidateCmdQueueFlags(dev_data, cb_node, "vkCmdClearColorImage()", VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
                                      VALIDATION_ERROR_18802415);
        skip |= ValidateCmd(dev_data, cb_node, CMD_CLEARCOLORIMAGE, "vkCmdClearColorImage()");
        skip |= insideRenderPass(dev_data, cb_node, "vkCmdClearColorImage()", VALIDATION_ERROR_18800017);
        for (uint32_t i = 0; i < rangeCount; ++i) {
            std::string param_name = "pRanges[" + std::to_string(i) + "]";
            skip |=
                ValidateImageSubresourceRange(dev_data, image_state, false, pRanges[i], "vkCmdClearColorImage", param_name.c_str());
            skip |= ValidateImageAttributes(dev_data, image_state, pRanges[i]);
            skip |= VerifyClearImageLayout(dev_data, cb_node, image_state, pRanges[i], imageLayout, "vkCmdClearColorImage()");
        }
    }
    return skip;
}

// This state recording routine is shared between ClearColorImage and ClearDepthStencilImage
void PreCallRecordCmdClearImage(layer_data *dev_data, VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout,
                                uint32_t rangeCount, const VkImageSubresourceRange *pRanges) {
    auto cb_node = GetCBNode(dev_data, commandBuffer);
    auto image_state = GetImageState(dev_data, image);
    if (cb_node && image_state) {
        AddCommandBufferBindingImage(dev_data, cb_node, image_state);
        std::function<bool()> function = [=]() {
            SetImageMemoryValid(dev_data, image_state, true);
            return false;
        };
        cb_node->queue_submit_functions.push_back(function);
        for (uint32_t i = 0; i < rangeCount; ++i) {
            RecordClearImageLayout(dev_data, cb_node, image, pRanges[i], imageLayout);
        }
    }
}

bool PreCallValidateCmdClearDepthStencilImage(layer_data *device_data, VkCommandBuffer commandBuffer, VkImage image,
                                              VkImageLayout imageLayout, uint32_t rangeCount,
                                              const VkImageSubresourceRange *pRanges) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    // TODO : Verify memory is in VK_IMAGE_STATE_CLEAR state
    auto cb_node = GetCBNode(device_data, commandBuffer);
    auto image_state = GetImageState(device_data, image);
    if (cb_node && image_state) {
        skip |= ValidateMemoryIsBoundToImage(device_data, image_state, "vkCmdClearDepthStencilImage()", VALIDATION_ERROR_18a00014);
        skip |= ValidateCmdQueueFlags(device_data, cb_node, "vkCmdClearDepthStencilImage()", VK_QUEUE_GRAPHICS_BIT,
                                      VALIDATION_ERROR_18a02415);
        skip |= ValidateCmd(device_data, cb_node, CMD_CLEARDEPTHSTENCILIMAGE, "vkCmdClearDepthStencilImage()");
        skip |= insideRenderPass(device_data, cb_node, "vkCmdClearDepthStencilImage()", VALIDATION_ERROR_18a00017);
        for (uint32_t i = 0; i < rangeCount; ++i) {
            std::string param_name = "pRanges[" + std::to_string(i) + "]";
            skip |= ValidateImageSubresourceRange(device_data, image_state, false, pRanges[i], "vkCmdClearDepthStencilImage",
                                                  param_name.c_str());
            skip |=
                VerifyClearImageLayout(device_data, cb_node, image_state, pRanges[i], imageLayout, "vkCmdClearDepthStencilImage()");
            // Image aspect must be depth or stencil or both
            if (((pRanges[i].aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != VK_IMAGE_ASPECT_DEPTH_BIT) &&
                ((pRanges[i].aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != VK_IMAGE_ASPECT_STENCIL_BIT)) {
                char const str[] =
                    "vkCmdClearDepthStencilImage aspectMasks for all subresource ranges must be "
                    "set to VK_IMAGE_ASPECT_DEPTH_BIT and/or VK_IMAGE_ASPECT_STENCIL_BIT";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE", str);
            }
        }
        if (image_state && !FormatIsDepthOrStencil(image_state->createInfo.format)) {
            char const str[] = "vkCmdClearDepthStencilImage called without a depth/stencil image.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_18a0001c, "IMAGE", "%s. %s", str,
                            validation_error_map[VALIDATION_ERROR_18a0001c]);
        }
    }
    return skip;
}

// Returns true if [x, xoffset] and [y, yoffset] overlap
static bool RangesIntersect(int32_t start, uint32_t start_offset, int32_t end, uint32_t end_offset) {
    bool result = false;
    uint32_t intersection_min = std::max(static_cast<uint32_t>(start), static_cast<uint32_t>(end));
    uint32_t intersection_max = std::min(static_cast<uint32_t>(start) + start_offset, static_cast<uint32_t>(end) + end_offset);

    if (intersection_max > intersection_min) {
        result = true;
    }
    return result;
}

// Returns true if two VkImageCopy structures overlap
static bool RegionIntersects(const VkImageCopy *src, const VkImageCopy *dst, VkImageType type) {
    bool result = false;
    if ((src->srcSubresource.mipLevel == dst->dstSubresource.mipLevel) &&
        (RangesIntersect(src->srcSubresource.baseArrayLayer, src->srcSubresource.layerCount, dst->dstSubresource.baseArrayLayer,
                         dst->dstSubresource.layerCount))) {
        result = true;
        switch (type) {
            case VK_IMAGE_TYPE_3D:
                result &= RangesIntersect(src->srcOffset.z, src->extent.depth, dst->dstOffset.z, dst->extent.depth);
            // Intentionally fall through to 2D case
            case VK_IMAGE_TYPE_2D:
                result &= RangesIntersect(src->srcOffset.y, src->extent.height, dst->dstOffset.y, dst->extent.height);
            // Intentionally fall through to 1D case
            case VK_IMAGE_TYPE_1D:
                result &= RangesIntersect(src->srcOffset.x, src->extent.width, dst->dstOffset.x, dst->extent.width);
                break;
            default:
                // Unrecognized or new IMAGE_TYPE enums will be caught in parameter_validation
                assert(false);
        }
    }
    return result;
}

// Returns non-zero if offset and extent exceed image extents
static const uint32_t x_bit = 1;
static const uint32_t y_bit = 2;
static const uint32_t z_bit = 4;
static uint32_t ExceedsBounds(const VkOffset3D *offset, const VkExtent3D *extent, const VkExtent3D *image_extent) {
    uint32_t result = 0;
    // Extents/depths cannot be negative but checks left in for clarity
    if ((offset->z + extent->depth > image_extent->depth) || (offset->z < 0) ||
        ((offset->z + static_cast<int32_t>(extent->depth)) < 0)) {
        result |= z_bit;
    }
    if ((offset->y + extent->height > image_extent->height) || (offset->y < 0) ||
        ((offset->y + static_cast<int32_t>(extent->height)) < 0)) {
        result |= y_bit;
    }
    if ((offset->x + extent->width > image_extent->width) || (offset->x < 0) ||
        ((offset->x + static_cast<int32_t>(extent->width)) < 0)) {
        result |= x_bit;
    }
    return result;
}

// Test if two VkExtent3D structs are equivalent
static inline bool IsExtentEqual(const VkExtent3D *extent, const VkExtent3D *other_extent) {
    bool result = true;
    if ((extent->width != other_extent->width) || (extent->height != other_extent->height) ||
        (extent->depth != other_extent->depth)) {
        result = false;
    }
    return result;
}

// Returns the effective extent of an image subresource, adjusted for mip level and array depth.
static inline VkExtent3D GetImageSubresourceExtent(const IMAGE_STATE *img, const VkImageSubresourceLayers *subresource) {
    const uint32_t mip = subresource->mipLevel;

    // Return zero extent if mip level doesn't exist
    if (mip >= img->createInfo.mipLevels) {
        return VkExtent3D{0, 0, 0};
    }

    // Don't allow mip adjustment to create 0 dim, but pass along a 0 if that's what subresource specified
    VkExtent3D extent = img->createInfo.extent;
    extent.width = (0 == extent.width ? 0 : std::max(1U, extent.width >> mip));
    extent.height = (0 == extent.height ? 0 : std::max(1U, extent.height >> mip));
    extent.depth = (0 == extent.depth ? 0 : std::max(1U, extent.depth >> mip));

    // Image arrays have an effective z extent that isn't diminished by mip level
    if (VK_IMAGE_TYPE_3D != img->createInfo.imageType) {
        extent.depth = img->createInfo.arrayLayers;
    }

    return extent;
}

// Test if the extent argument has all dimensions set to 0.
static inline bool IsExtentAllZeroes(const VkExtent3D *extent) {
    return ((extent->width == 0) && (extent->height == 0) && (extent->depth == 0));
}

// Test if the extent argument has any dimensions set to 0.
static inline bool IsExtentSizeZero(const VkExtent3D *extent) {
    return ((extent->width == 0) || (extent->height == 0) || (extent->depth == 0));
}

// Returns the image transfer granularity for a specific image scaled by compressed block size if necessary.
static inline VkExtent3D GetScaledItg(layer_data *device_data, const GLOBAL_CB_NODE *cb_node, const IMAGE_STATE *img) {
    // Default to (0, 0, 0) granularity in case we can't find the real granularity for the physical device.
    VkExtent3D granularity = {0, 0, 0};
    auto pPool = GetCommandPoolNode(device_data, cb_node->createInfo.commandPool);
    if (pPool) {
        granularity =
            GetPhysDevProperties(device_data)->queue_family_properties[pPool->queueFamilyIndex].minImageTransferGranularity;
        if (FormatIsCompressed(img->createInfo.format)) {
            auto block_size = FormatCompressedTexelBlockExtent(img->createInfo.format);
            granularity.width *= block_size.width;
            granularity.height *= block_size.height;
        }
    }
    return granularity;
}

// Test elements of a VkExtent3D structure against alignment constraints contained in another VkExtent3D structure
static inline bool IsExtentAligned(const VkExtent3D *extent, const VkExtent3D *granularity) {
    bool valid = true;
    if ((SafeModulo(extent->depth, granularity->depth) != 0) || (SafeModulo(extent->width, granularity->width) != 0) ||
        (SafeModulo(extent->height, granularity->height) != 0)) {
        valid = false;
    }
    return valid;
}

// Check elements of a VkOffset3D structure against a queue family's Image Transfer Granularity values
static inline bool CheckItgOffset(layer_data *device_data, const GLOBAL_CB_NODE *cb_node, const VkOffset3D *offset,
                                  const VkExtent3D *granularity, const uint32_t i, const char *function, const char *member) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    VkExtent3D offset_extent = {};
    offset_extent.width = static_cast<uint32_t>(abs(offset->x));
    offset_extent.height = static_cast<uint32_t>(abs(offset->y));
    offset_extent.depth = static_cast<uint32_t>(abs(offset->z));
    if (IsExtentAllZeroes(granularity)) {
        // If the queue family image transfer granularity is (0, 0, 0), then the offset must always be (0, 0, 0)
        if (IsExtentAllZeroes(&offset_extent) == false) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_IMAGE_TRANSFER_GRANULARITY, "DS",
                            "%s: pRegion[%d].%s (x=%d, y=%d, z=%d) must be (x=0, y=0, z=0) "
                            "when the command buffer's queue family image transfer granularity is (w=0, h=0, d=0).",
                            function, i, member, offset->x, offset->y, offset->z);
        }
    } else {
        // If the queue family image transfer granularity is not (0, 0, 0), then the offset dimensions must always be even
        // integer multiples of the image transfer granularity.
        if (IsExtentAligned(&offset_extent, granularity) == false) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_IMAGE_TRANSFER_GRANULARITY, "DS",
                            "%s: pRegion[%d].%s (x=%d, y=%d, z=%d) dimensions must be even integer "
                            "multiples of this command buffer's queue family image transfer granularity (w=%d, h=%d, d=%d).",
                            function, i, member, offset->x, offset->y, offset->z, granularity->width, granularity->height,
                            granularity->depth);
        }
    }
    return skip;
}

// Check elements of a VkExtent3D structure against a queue family's Image Transfer Granularity values
static inline bool CheckItgExtent(layer_data *device_data, const GLOBAL_CB_NODE *cb_node, const VkExtent3D *extent,
                                  const VkOffset3D *offset, const VkExtent3D *granularity, const VkExtent3D *subresource_extent,
                                  const uint32_t i, const char *function, const char *member) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    if (IsExtentAllZeroes(granularity)) {
        // If the queue family image transfer granularity is (0, 0, 0), then the extent must always match the image
        // subresource extent.
        if (IsExtentEqual(extent, subresource_extent) == false) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_IMAGE_TRANSFER_GRANULARITY, "DS",
                            "%s: pRegion[%d].%s (w=%d, h=%d, d=%d) must match the image subresource extents (w=%d, h=%d, d=%d) "
                            "when the command buffer's queue family image transfer granularity is (w=0, h=0, d=0).",
                            function, i, member, extent->width, extent->height, extent->depth, subresource_extent->width,
                            subresource_extent->height, subresource_extent->depth);
        }
    } else {
        // If the queue family image transfer granularity is not (0, 0, 0), then the extent dimensions must always be even
        // integer multiples of the image transfer granularity or the offset + extent dimensions must always match the image
        // subresource extent dimensions.
        VkExtent3D offset_extent_sum = {};
        offset_extent_sum.width = static_cast<uint32_t>(abs(offset->x)) + extent->width;
        offset_extent_sum.height = static_cast<uint32_t>(abs(offset->y)) + extent->height;
        offset_extent_sum.depth = static_cast<uint32_t>(abs(offset->z)) + extent->depth;

        bool x_ok =
            ((0 == SafeModulo(extent->width, granularity->width)) || (subresource_extent->width == offset_extent_sum.width));
        bool y_ok =
            ((0 == SafeModulo(extent->height, granularity->height)) || (subresource_extent->height == offset_extent_sum.height));
        bool z_ok =
            ((0 == SafeModulo(extent->depth, granularity->depth)) || (subresource_extent->depth == offset_extent_sum.depth));

        if (!(x_ok && y_ok && z_ok)) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_IMAGE_TRANSFER_GRANULARITY, "DS",
                        "%s: pRegion[%d].%s (w=%d, h=%d, d=%d) dimensions must be even integer multiples of this command buffer's "
                        "queue family image transfer granularity (w=%d, h=%d, d=%d) or offset (x=%d, y=%d, z=%d) + "
                        "extent (w=%d, h=%d, d=%d) must match the image subresource extents (w=%d, h=%d, d=%d).",
                        function, i, member, extent->width, extent->height, extent->depth, granularity->width, granularity->height,
                        granularity->depth, offset->x, offset->y, offset->z, extent->width, extent->height, extent->depth,
                        subresource_extent->width, subresource_extent->height, subresource_extent->depth);
        }
    }
    return skip;
}

// Check a uint32_t width or stride value against a queue family's Image Transfer Granularity width value
static inline bool CheckItgInt(layer_data *device_data, const GLOBAL_CB_NODE *cb_node, const uint32_t value,
                               const uint32_t granularity, const uint32_t i, const char *function, const char *member) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    bool skip = false;
    if (SafeModulo(value, granularity) != 0) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_IMAGE_TRANSFER_GRANULARITY, "DS",
                        "%s: pRegion[%d].%s (%d) must be an even integer multiple of this command buffer's queue family image "
                        "transfer granularity width (%d).",
                        function, i, member, value, granularity);
    }
    return skip;
}

// Check a VkDeviceSize value against a queue family's Image Transfer Granularity width value
static inline bool CheckItgSize(layer_data *device_data, const GLOBAL_CB_NODE *cb_node, const VkDeviceSize value,
                                const uint32_t granularity, const uint32_t i, const char *function, const char *member) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    if (SafeModulo(value, granularity) != 0) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_IMAGE_TRANSFER_GRANULARITY, "DS",
                        "%s: pRegion[%d].%s (%" PRIdLEAST64
                        ") must be an even integer multiple of this command buffer's queue family image transfer "
                        "granularity width (%d).",
                        function, i, member, value, granularity);
    }
    return skip;
}

// Check valid usage Image Tranfer Granularity requirements for elements of a VkBufferImageCopy structure
bool ValidateCopyBufferImageTransferGranularityRequirements(layer_data *device_data, const GLOBAL_CB_NODE *cb_node,
                                                            const IMAGE_STATE *img, const VkBufferImageCopy *region,
                                                            const uint32_t i, const char *function) {
    bool skip = false;
    if (FormatIsCompressed(img->createInfo.format) == true) {
        // TODO: Add granularity checking for compressed formats

        // bufferRowLength must be a multiple of the compressed texel block width
        // bufferImageHeight must be a multiple of the compressed texel block height
        // all members of imageOffset must be a multiple of the corresponding dimensions of the compressed texel block
        // bufferOffset must be a multiple of the compressed texel block size in bytes
        // imageExtent.width must be a multiple of the compressed texel block width or (imageExtent.width + imageOffset.x)
        //     must equal the image subresource width
        // imageExtent.height must be a multiple of the compressed texel block height or (imageExtent.height + imageOffset.y)
        //     must equal the image subresource height
        // imageExtent.depth must be a multiple of the compressed texel block depth or (imageExtent.depth + imageOffset.z)
        //     must equal the image subresource depth
    } else {
        VkExtent3D granularity = GetScaledItg(device_data, cb_node, img);
        skip |= CheckItgSize(device_data, cb_node, region->bufferOffset, granularity.width, i, function, "bufferOffset");
        skip |= CheckItgInt(device_data, cb_node, region->bufferRowLength, granularity.width, i, function, "bufferRowLength");
        skip |= CheckItgInt(device_data, cb_node, region->bufferImageHeight, granularity.width, i, function, "bufferImageHeight");
        skip |= CheckItgOffset(device_data, cb_node, &region->imageOffset, &granularity, i, function, "imageOffset");
        VkExtent3D subresource_extent = GetImageSubresourceExtent(img, &region->imageSubresource);
        skip |= CheckItgExtent(device_data, cb_node, &region->imageExtent, &region->imageOffset, &granularity, &subresource_extent,
                               i, function, "imageExtent");
    }
    return skip;
}

// Check valid usage Image Tranfer Granularity requirements for elements of a VkImageCopy structure
bool ValidateCopyImageTransferGranularityRequirements(layer_data *device_data, const GLOBAL_CB_NODE *cb_node,
                                                      const IMAGE_STATE *src_img, const IMAGE_STATE *dst_img,
                                                      const VkImageCopy *region, const uint32_t i, const char *function) {
    bool skip = false;
    VkExtent3D granularity = GetScaledItg(device_data, cb_node, src_img);
    skip |= CheckItgOffset(device_data, cb_node, &region->srcOffset, &granularity, i, function, "srcOffset");
    VkExtent3D subresource_extent = GetImageSubresourceExtent(src_img, &region->srcSubresource);
    skip |= CheckItgExtent(device_data, cb_node, &region->extent, &region->srcOffset, &granularity, &subresource_extent, i,
                           function, "extent");

    granularity = GetScaledItg(device_data, cb_node, dst_img);
    skip |= CheckItgOffset(device_data, cb_node, &region->dstOffset, &granularity, i, function, "dstOffset");
    subresource_extent = GetImageSubresourceExtent(dst_img, &region->dstSubresource);
    skip |= CheckItgExtent(device_data, cb_node, &region->extent, &region->dstOffset, &granularity, &subresource_extent, i,
                           function, "extent");
    return skip;
}

// Validate contents of a VkImageCopy struct
bool ValidateImageCopyData(const layer_data *device_data, const debug_report_data *report_data, const uint32_t regionCount,
                           const VkImageCopy *ic_regions, const IMAGE_STATE *src_state, const IMAGE_STATE *dst_state) {
    bool skip = false;

    for (uint32_t i = 0; i < regionCount; i++) {
        VkImageCopy image_copy = ic_regions[i];
        bool slice_override = false;
        uint32_t depth_slices = 0;

        // Special case for copying between a 1D/2D array and a 3D image
        // TBD: This seems like the only way to reconcile 3 mutually-exclusive VU checks for 2D/3D copies. Heads up.
        if ((VK_IMAGE_TYPE_3D == src_state->createInfo.imageType) && (VK_IMAGE_TYPE_3D != dst_state->createInfo.imageType)) {
            depth_slices = image_copy.dstSubresource.layerCount;  // Slice count from 2D subresource
            slice_override = (depth_slices != 1);
        } else if ((VK_IMAGE_TYPE_3D == dst_state->createInfo.imageType) && (VK_IMAGE_TYPE_3D != src_state->createInfo.imageType)) {
            depth_slices = image_copy.srcSubresource.layerCount;  // Slice count from 2D subresource
            slice_override = (depth_slices != 1);
        }

        // Do all checks on source image
        //
        if (src_state->createInfo.imageType == VK_IMAGE_TYPE_1D) {
            if ((0 != image_copy.srcOffset.y) || (1 != image_copy.extent.height)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c00124, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] srcOffset.y is %d and extent.height is %d. For 1D images these must "
                            "be 0 and 1, respectively. %s",
                            i, image_copy.srcOffset.y, image_copy.extent.height, validation_error_map[VALIDATION_ERROR_09c00124]);
            }
        }

        if ((src_state->createInfo.imageType == VK_IMAGE_TYPE_1D) || (src_state->createInfo.imageType == VK_IMAGE_TYPE_2D)) {
            if ((0 != image_copy.srcOffset.z) || (1 != image_copy.extent.depth)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c00128, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] srcOffset.z is %d and extent.depth is %d. For 1D and 2D images "
                            "these must be 0 and 1, respectively. %s",
                            i, image_copy.srcOffset.z, image_copy.extent.depth, validation_error_map[VALIDATION_ERROR_09c00128]);
            }
        }

        // VU01199 changed with mnt1
        if (GetDeviceExtensions(device_data)->vk_khr_maintenance1) {
            if (src_state->createInfo.imageType == VK_IMAGE_TYPE_3D) {
                if ((0 != image_copy.srcSubresource.baseArrayLayer) || (1 != image_copy.srcSubresource.layerCount)) {
                    skip |=
                        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c0011a, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] srcSubresource.baseArrayLayer is %d and srcSubresource.layerCount "
                                "is %d. For VK_IMAGE_TYPE_3D images these must be 0 and 1, respectively. %s",
                                i, image_copy.srcSubresource.baseArrayLayer, image_copy.srcSubresource.layerCount,
                                validation_error_map[VALIDATION_ERROR_09c0011a]);
                }
            }
        } else {  // Pre maint 1
            if (src_state->createInfo.imageType == VK_IMAGE_TYPE_3D || dst_state->createInfo.imageType == VK_IMAGE_TYPE_3D) {
                if ((0 != image_copy.srcSubresource.baseArrayLayer) || (1 != image_copy.srcSubresource.layerCount)) {
                    skip |=
                        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c0011a, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] srcSubresource.baseArrayLayer is %d and "
                                "srcSubresource.layerCount is %d. For copies with either source or dest of type "
                                "VK_IMAGE_TYPE_3D, these must be 0 and 1, respectively. %s",
                                i, image_copy.srcSubresource.baseArrayLayer, image_copy.srcSubresource.layerCount,
                                validation_error_map[VALIDATION_ERROR_09c0011a]);
                }
            }
        }

        // TODO: this VU is redundant with VU01224. Gitlab issue 812 submitted to get it removed from the spec.
        if ((image_copy.srcSubresource.baseArrayLayer >= src_state->createInfo.arrayLayers) ||
            (image_copy.srcSubresource.baseArrayLayer + image_copy.srcSubresource.layerCount > src_state->createInfo.arrayLayers)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c0012a, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] srcSubresource.baseArrayLayer (%d) must be less than the source image's "
                            "arrayLayers (%d), and the sum of baseArrayLayer and srcSubresource.layerCount (%d) must be less than "
                            "or equal to the source image's arrayLayers. %s",
                            i, image_copy.srcSubresource.baseArrayLayer, src_state->createInfo.arrayLayers,
                            image_copy.srcSubresource.layerCount, validation_error_map[VALIDATION_ERROR_09c0012a]);
        }

        // Checks that apply only to compressed images
        if (FormatIsCompressed(src_state->createInfo.format)) {
            VkExtent3D block_size = FormatCompressedTexelBlockExtent(src_state->createInfo.format);

            //  image offsets must be multiples of block dimensions
            if ((SafeModulo(image_copy.srcOffset.x, block_size.width) != 0) ||
                (SafeModulo(image_copy.srcOffset.y, block_size.height) != 0) ||
                (SafeModulo(image_copy.srcOffset.z, block_size.depth) != 0)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c0013a, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] srcOffset (%d, %d) must be multiples of the compressed image's "
                                "texel width & height (%d, %d). %s.",
                                i, image_copy.srcOffset.x, image_copy.srcOffset.y, block_size.width, block_size.height,
                                validation_error_map[VALIDATION_ERROR_09c0013a]);
            }

            // extent width must be a multiple of block width, or extent+offset width must equal subresource width
            VkExtent3D mip_extent = GetImageSubresourceExtent(src_state, &(image_copy.srcSubresource));
            if ((SafeModulo(image_copy.extent.width, block_size.width) != 0) &&
                (image_copy.extent.width + image_copy.srcOffset.x != mip_extent.width)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c0013c, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] extent width (%d) must be a multiple of the compressed texture block "
                            "width (%d), or when added to srcOffset.x (%d) must equal the image subresource width (%d). %s.",
                            i, image_copy.extent.width, block_size.width, image_copy.srcOffset.x, mip_extent.width,
                            validation_error_map[VALIDATION_ERROR_09c0013c]);
            }

            // extent height must be a multiple of block height, or extent+offset height must equal subresource height
            if ((SafeModulo(image_copy.extent.height, block_size.height) != 0) &&
                (image_copy.extent.height + image_copy.srcOffset.y != mip_extent.height)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c0013e, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] extent height (%d) must be a multiple of the compressed texture block "
                            "height (%d), or when added to srcOffset.y (%d) must equal the image subresource height (%d). %s.",
                            i, image_copy.extent.height, block_size.height, image_copy.srcOffset.y, mip_extent.height,
                            validation_error_map[VALIDATION_ERROR_09c0013e]);
            }

            // extent depth must be a multiple of block depth, or extent+offset depth must equal subresource depth
            uint32_t copy_depth = (slice_override ? depth_slices : image_copy.extent.depth);
            if ((SafeModulo(copy_depth, block_size.depth) != 0) && (copy_depth + image_copy.srcOffset.z != mip_extent.depth)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(src_state->image), __LINE__, VALIDATION_ERROR_09c00140, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] extent width (%d) must be a multiple of the compressed texture block "
                            "depth (%d), or when added to srcOffset.z (%d) must equal the image subresource depth (%d). %s.",
                            i, image_copy.extent.depth, block_size.depth, image_copy.srcOffset.z, mip_extent.depth,
                            validation_error_map[VALIDATION_ERROR_09c00140]);
            }
        }  // Compressed

        // Do all checks on dest image
        //
        if (dst_state->createInfo.imageType == VK_IMAGE_TYPE_1D) {
            if ((0 != image_copy.dstOffset.y) || (1 != image_copy.extent.height)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c00130, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] dstOffset.y is %d and extent.height is %d. For 1D images these must "
                            "be 0 and 1, respectively. %s",
                            i, image_copy.dstOffset.y, image_copy.extent.height, validation_error_map[VALIDATION_ERROR_09c00130]);
            }
        }

        if ((dst_state->createInfo.imageType == VK_IMAGE_TYPE_1D) || (dst_state->createInfo.imageType == VK_IMAGE_TYPE_2D)) {
            if ((0 != image_copy.dstOffset.z) || (1 != image_copy.extent.depth)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c00134, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] dstOffset.z is %d and extent.depth is %d. For 1D and 2D images "
                            "these must be 0 and 1, respectively. %s",
                            i, image_copy.dstOffset.z, image_copy.extent.depth, validation_error_map[VALIDATION_ERROR_09c00134]);
            }
        }

        if (dst_state->createInfo.imageType == VK_IMAGE_TYPE_3D) {
            if ((0 != image_copy.dstSubresource.baseArrayLayer) || (1 != image_copy.dstSubresource.layerCount)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c0011a, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] dstSubresource.baseArrayLayer is %d and dstSubresource.layerCount "
                                "is %d. For VK_IMAGE_TYPE_3D images these must be 0 and 1, respectively. %s",
                                i, image_copy.dstSubresource.baseArrayLayer, image_copy.dstSubresource.layerCount,
                                validation_error_map[VALIDATION_ERROR_09c0011a]);
            }
        }
        // VU01199 changed with mnt1
        if (GetDeviceExtensions(device_data)->vk_khr_maintenance1) {
            if (dst_state->createInfo.imageType == VK_IMAGE_TYPE_3D) {
                if ((0 != image_copy.dstSubresource.baseArrayLayer) || (1 != image_copy.dstSubresource.layerCount)) {
                    skip |=
                        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c0011a, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] dstSubresource.baseArrayLayer is %d and dstSubresource.layerCount "
                                "is %d. For VK_IMAGE_TYPE_3D images these must be 0 and 1, respectively. %s",
                                i, image_copy.dstSubresource.baseArrayLayer, image_copy.dstSubresource.layerCount,
                                validation_error_map[VALIDATION_ERROR_09c0011a]);
                }
            }
        } else {  // Pre maint 1
            if (src_state->createInfo.imageType == VK_IMAGE_TYPE_3D || dst_state->createInfo.imageType == VK_IMAGE_TYPE_3D) {
                if ((0 != image_copy.dstSubresource.baseArrayLayer) || (1 != image_copy.dstSubresource.layerCount)) {
                    skip |=
                        log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c0011a, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] dstSubresource.baseArrayLayer is %d and "
                                "dstSubresource.layerCount is %d. For copies with either source or dest of type "
                                "VK_IMAGE_TYPE_3D, these must be 0 and 1, respectively. %s",
                                i, image_copy.dstSubresource.baseArrayLayer, image_copy.dstSubresource.layerCount,
                                validation_error_map[VALIDATION_ERROR_09c0011a]);
                }
            }
        }

        // TODO: this VU is redundant with VU01224. Gitlab issue 812 submitted to get it removed from the spec.
        if ((image_copy.dstSubresource.baseArrayLayer >= dst_state->createInfo.arrayLayers) ||
            (image_copy.dstSubresource.baseArrayLayer + image_copy.dstSubresource.layerCount > dst_state->createInfo.arrayLayers)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c00136, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] dstSubresource.baseArrayLayer (%d) must be less than the dest image's "
                            "arrayLayers (%d), and the sum of baseArrayLayer and dstSubresource.layerCount (%d) must be less than "
                            "or equal to the dest image's arrayLayers. %s",
                            i, image_copy.dstSubresource.baseArrayLayer, dst_state->createInfo.arrayLayers,
                            image_copy.dstSubresource.layerCount, validation_error_map[VALIDATION_ERROR_09c00136]);
        }

        // Checks that apply only to compressed images
        if (FormatIsCompressed(dst_state->createInfo.format)) {
            VkExtent3D block_size = FormatCompressedTexelBlockExtent(dst_state->createInfo.format);

            //  image offsets must be multiples of block dimensions
            if ((SafeModulo(image_copy.dstOffset.x, block_size.width) != 0) ||
                (SafeModulo(image_copy.dstOffset.y, block_size.height) != 0) ||
                (SafeModulo(image_copy.dstOffset.z, block_size.depth) != 0)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c00144, "IMAGE",
                                "vkCmdCopyImage(): pRegion[%d] dstOffset (%d, %d) must be multiples of the compressed image's "
                                "texel width & height (%d, %d). %s.",
                                i, image_copy.dstOffset.x, image_copy.dstOffset.y, block_size.width, block_size.height,
                                validation_error_map[VALIDATION_ERROR_09c00144]);
            }

            // extent width must be a multiple of block width, or extent+offset width must equal subresource width
            VkExtent3D mip_extent = GetImageSubresourceExtent(dst_state, &(image_copy.dstSubresource));
            if ((SafeModulo(image_copy.extent.width, block_size.width) != 0) &&
                (image_copy.extent.width + image_copy.dstOffset.x != mip_extent.width)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c00146, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] extent width (%d) must be a multiple of the compressed texture block "
                            "width (%d), or when added to dstOffset.x (%d) must equal the image subresource width (%d). %s.",
                            i, image_copy.extent.width, block_size.width, image_copy.dstOffset.x, mip_extent.width,
                            validation_error_map[VALIDATION_ERROR_09c00146]);
            }

            // extent height must be a multiple of block height, or extent+offset height must equal subresource height
            if ((SafeModulo(image_copy.extent.height, block_size.height) != 0) &&
                (image_copy.extent.height + image_copy.dstOffset.y != mip_extent.height)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c00148, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] extent height (%d) must be a multiple of the compressed texture block "
                            "height (%d), or when added to dstOffset.y (%d) must equal the image subresource height (%d). %s.",
                            i, image_copy.extent.height, block_size.height, image_copy.dstOffset.y, mip_extent.height,
                            validation_error_map[VALIDATION_ERROR_09c00148]);
            }

            // extent depth must be a multiple of block depth, or extent+offset depth must equal subresource depth
            uint32_t copy_depth = (slice_override ? depth_slices : image_copy.extent.depth);
            if ((SafeModulo(copy_depth, block_size.depth) != 0) && (copy_depth + image_copy.dstOffset.z != mip_extent.depth)) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            reinterpret_cast<const uint64_t &>(dst_state->image), __LINE__, VALIDATION_ERROR_09c0014a, "IMAGE",
                            "vkCmdCopyImage(): pRegion[%d] extent width (%d) must be a multiple of the compressed texture block "
                            "depth (%d), or when added to dstOffset.z (%d) must equal the image subresource depth (%d). %s.",
                            i, image_copy.extent.depth, block_size.depth, image_copy.dstOffset.z, mip_extent.depth,
                            validation_error_map[VALIDATION_ERROR_09c0014a]);
            }
        }  // Compressed
    }
    return skip;
}

bool PreCallValidateCmdCopyImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                 IMAGE_STATE *dst_image_state, uint32_t region_count, const VkImageCopy *regions,
                                 VkImageLayout src_image_layout, VkImageLayout dst_image_layout) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    skip = ValidateImageCopyData(device_data, report_data, region_count, regions, src_image_state, dst_image_state);

    VkCommandBuffer command_buffer = cb_node->commandBuffer;

    for (uint32_t i = 0; i < region_count; i++) {
        bool slice_override = false;
        uint32_t depth_slices = 0;

        // Special case for copying between a 1D/2D array and a 3D image
        // TBD: This seems like the only way to reconcile 3 mutually-exclusive VU checks for 2D/3D copies. Heads up.
        if ((VK_IMAGE_TYPE_3D == src_image_state->createInfo.imageType) &&
            (VK_IMAGE_TYPE_3D != dst_image_state->createInfo.imageType)) {
            depth_slices = regions[i].dstSubresource.layerCount;  // Slice count from 2D subresource
            slice_override = (depth_slices != 1);
        } else if ((VK_IMAGE_TYPE_3D == dst_image_state->createInfo.imageType) &&
                   (VK_IMAGE_TYPE_3D != src_image_state->createInfo.imageType)) {
            depth_slices = regions[i].srcSubresource.layerCount;  // Slice count from 2D subresource
            slice_override = (depth_slices != 1);
        }

        if (regions[i].srcSubresource.layerCount == 0) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: number of layers in pRegions[" << i << "] srcSubresource is zero";
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(command_buffer), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE", "%s", ss.str().c_str());
        }

        if (regions[i].dstSubresource.layerCount == 0) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: number of layers in pRegions[" << i << "] dstSubresource is zero";
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(command_buffer), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE", "%s", ss.str().c_str());
        }

        if (GetDeviceExtensions(device_data)->vk_khr_maintenance1) {
            // No chance of mismatch if we're overriding depth slice count
            if (!slice_override) {
                // The number of depth slices in srcSubresource and dstSubresource must match
                // Depth comes from layerCount for 1D,2D resources, from extent.depth for 3D
                uint32_t src_slices =
                    (VK_IMAGE_TYPE_3D == src_image_state->createInfo.imageType ? regions[i].extent.depth
                                                                               : regions[i].srcSubresource.layerCount);
                uint32_t dst_slices =
                    (VK_IMAGE_TYPE_3D == dst_image_state->createInfo.imageType ? regions[i].extent.depth
                                                                               : regions[i].dstSubresource.layerCount);
                if (src_slices != dst_slices) {
                    std::stringstream ss;
                    ss << "vkCmdCopyImage: number of depth slices in source and destination subresources for pRegions[" << i
                       << "] do not match";
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    reinterpret_cast<uint64_t &>(command_buffer), __LINE__, VALIDATION_ERROR_09c00118, "IMAGE",
                                    "%s. %s", ss.str().c_str(), validation_error_map[VALIDATION_ERROR_09c00118]);
                }
            }
        } else {
            // For each region the layerCount member of srcSubresource and dstSubresource must match
            if (regions[i].srcSubresource.layerCount != regions[i].dstSubresource.layerCount) {
                std::stringstream ss;
                ss << "vkCmdCopyImage: number of layers in source and destination subresources for pRegions[" << i
                   << "] do not match";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c00118, "IMAGE", "%s. %s",
                                ss.str().c_str(), validation_error_map[VALIDATION_ERROR_09c00118]);
            }
        }

        // For each region, the aspectMask member of srcSubresource and dstSubresource must match
        if (regions[i].srcSubresource.aspectMask != regions[i].dstSubresource.aspectMask) {
            char const str[] = "vkCmdCopyImage: Src and dest aspectMasks for each region must match";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c00112, "IMAGE", "%s. %s", str,
                            validation_error_map[VALIDATION_ERROR_09c00112]);
        }

        // For each region, the aspectMask member of srcSubresource must be present in the source image
        if (!VerifyAspectsPresent(regions[i].srcSubresource.aspectMask, src_image_state->createInfo.format)) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: pRegion[" << i
               << "] srcSubresource.aspectMask cannot specify aspects not present in source image";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c0011c, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_09c0011c]);
        }

        // For each region, the aspectMask member of dstSubresource must be present in the destination image
        if (!VerifyAspectsPresent(regions[i].dstSubresource.aspectMask, dst_image_state->createInfo.format)) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: pRegion[" << i << "] dstSubresource.aspectMask cannot specify aspects not present in dest image";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c0011e, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_09c0011e]);
        }

        // AspectMask must not contain VK_IMAGE_ASPECT_METADATA_BIT
        if ((regions[i].srcSubresource.aspectMask & VK_IMAGE_ASPECT_METADATA_BIT) ||
            (regions[i].dstSubresource.aspectMask & VK_IMAGE_ASPECT_METADATA_BIT)) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: pRegions[" << i << "] may not specify aspectMask containing VK_IMAGE_ASPECT_METADATA_BIT";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_0a600150, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_0a600150]);
        }

        // For each region, if aspectMask contains VK_IMAGE_ASPECT_COLOR_BIT, it must not contain either of
        // VK_IMAGE_ASPECT_DEPTH_BIT or VK_IMAGE_ASPECT_STENCIL_BIT
        if ((regions[i].srcSubresource.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) &&
            (regions[i].srcSubresource.aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
            char const str[] = "vkCmdCopyImage aspectMask cannot specify both COLOR and DEPTH/STENCIL aspects";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_0a60014e, "IMAGE", "%s. %s", str,
                            validation_error_map[VALIDATION_ERROR_0a60014e]);
        }

        // MipLevel must be less than the mipLevels specified in VkImageCreateInfo when the image was created
        if (regions[i].srcSubresource.mipLevel >= src_image_state->createInfo.mipLevels) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: pRegions[" << i
               << "] specifies a src mipLevel greater than the number specified when the srcImage was created.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_0a600152, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_0a600152]);
        }
        if (regions[i].dstSubresource.mipLevel >= dst_image_state->createInfo.mipLevels) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: pRegions[" << i
               << "] specifies a dst mipLevel greater than the number specified when the dstImage was created.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_0a600152, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_0a600152]);
        }

        // (baseArrayLayer + layerCount) must be less than or equal to the arrayLayers specified in VkImageCreateInfo when the
        // image was created
        if ((regions[i].srcSubresource.baseArrayLayer + regions[i].srcSubresource.layerCount) >
            src_image_state->createInfo.arrayLayers) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: srcImage arrayLayers was " << src_image_state->createInfo.arrayLayers << " but subRegion[" << i
               << "] baseArrayLayer + layerCount is "
               << (regions[i].srcSubresource.baseArrayLayer + regions[i].srcSubresource.layerCount);
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_0a600154, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_0a600154]);
        }
        if ((regions[i].dstSubresource.baseArrayLayer + regions[i].dstSubresource.layerCount) >
            dst_image_state->createInfo.arrayLayers) {
            std::stringstream ss;
            ss << "vkCmdCopyImage: dstImage arrayLayers was " << dst_image_state->createInfo.arrayLayers << " but subRegion[" << i
               << "] baseArrayLayer + layerCount is "
               << (regions[i].dstSubresource.baseArrayLayer + regions[i].dstSubresource.layerCount);
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_0a600154, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_0a600154]);
        }

        // Check region extents for 1D-1D, 2D-2D, and 3D-3D copies
        if (src_image_state->createInfo.imageType == dst_image_state->createInfo.imageType) {
            // The source region specified by a given element of regions must be a region that is contained within srcImage
            VkExtent3D img_extent = GetImageSubresourceExtent(src_image_state, &(regions[i].srcSubresource));
            if (0 != ExceedsBounds(&regions[i].srcOffset, &regions[i].extent, &img_extent)) {
                std::stringstream ss;
                ss << "vkCmdCopyImage: Source pRegion[" << i << "] with mipLevel [ " << regions[i].srcSubresource.mipLevel
                   << " ], offset [ " << regions[i].srcOffset.x << ", " << regions[i].srcOffset.y << ", " << regions[i].srcOffset.z
                   << " ], extent [ " << regions[i].extent.width << ", " << regions[i].extent.height << ", "
                   << regions[i].extent.depth << " ] exceeds the source image dimensions";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_190000f4, "IMAGE", "%s. %s",
                                ss.str().c_str(), validation_error_map[VALIDATION_ERROR_190000f4]);
            }

            // The destination region specified by a given element of regions must be a region that is contained within dst_image
            img_extent = GetImageSubresourceExtent(dst_image_state, &(regions[i].dstSubresource));
            if (0 != ExceedsBounds(&regions[i].dstOffset, &regions[i].extent, &img_extent)) {
                std::stringstream ss;
                ss << "vkCmdCopyImage: Dest pRegion[" << i << "] with mipLevel [ " << regions[i].dstSubresource.mipLevel
                   << " ], offset [ " << regions[i].dstOffset.x << ", " << regions[i].dstOffset.y << ", " << regions[i].dstOffset.z
                   << " ], extent [ " << regions[i].extent.width << ", " << regions[i].extent.height << ", "
                   << regions[i].extent.depth << " ] exceeds the destination image dimensions";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_190000f6, "IMAGE", "%s. %s",
                                ss.str().c_str(), validation_error_map[VALIDATION_ERROR_190000f6]);
            }
        }

        // Each dimension offset + extent limits must fall with image subresource extent
        VkExtent3D subresource_extent = GetImageSubresourceExtent(src_image_state, &(regions[i].srcSubresource));
        VkExtent3D copy_extent = regions[i].extent;
        if (slice_override) copy_extent.depth = depth_slices;
        uint32_t extent_check = ExceedsBounds(&(regions[i].srcOffset), &copy_extent, &subresource_extent);
        if (extent_check & x_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c00120, "IMAGE",
                            "vkCmdCopyImage: Source image pRegion %1d x-dimension offset [%1d] + extent [%1d] exceeds subResource "
                            "width [%1d]. %s",
                            i, regions[i].srcOffset.x, regions[i].extent.width, subresource_extent.width,
                            validation_error_map[VALIDATION_ERROR_09c00120]);
        }

        if (extent_check & y_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c00122, "IMAGE",
                            "vkCmdCopyImage: Source image pRegion %1d y-dimension offset [%1d] + extent [%1d] exceeds subResource "
                            "height [%1d]. %s",
                            i, regions[i].srcOffset.y, regions[i].extent.height, subresource_extent.height,
                            validation_error_map[VALIDATION_ERROR_09c00122]);
        }
        if (extent_check & z_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c00126, "IMAGE",
                            "vkCmdCopyImage: Source image pRegion %1d z-dimension offset [%1d] + extent [%1d] exceeds subResource "
                            "depth [%1d]. %s",
                            i, regions[i].srcOffset.z, copy_extent.depth, subresource_extent.depth,
                            validation_error_map[VALIDATION_ERROR_09c00126]);
        }

        subresource_extent = GetImageSubresourceExtent(dst_image_state, &(regions[i].dstSubresource));
        copy_extent = regions[i].extent;
        if (slice_override) copy_extent.depth = depth_slices;
        extent_check = ExceedsBounds(&(regions[i].dstOffset), &copy_extent, &subresource_extent);
        if (extent_check & x_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c0012c, "IMAGE",
                            "vkCmdCopyImage: Dest image pRegion %1d x-dimension offset [%1d] + extent [%1d] exceeds subResource "
                            "width [%1d]. %s",
                            i, regions[i].dstOffset.x, regions[i].extent.width, subresource_extent.width,
                            validation_error_map[VALIDATION_ERROR_09c0012c]);
        }
        if (extent_check & y_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c0012e, "IMAGE",
                            "vkCmdCopyImage: Dest image pRegion %1d y-dimension offset [%1d] + extent [%1d] exceeds subResource "
                            "height [%1d]. %s",
                            i, regions[i].dstOffset.y, regions[i].extent.height, subresource_extent.height,
                            validation_error_map[VALIDATION_ERROR_09c0012e]);
        }
        if (extent_check & z_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_09c00132, "IMAGE",
                            "vkCmdCopyImage: Dest image pRegion %1d z-dimension offset [%1d] + extent [%1d] exceeds subResource "
                            "depth [%1d]. %s",
                            i, regions[i].dstOffset.z, copy_extent.depth, subresource_extent.depth,
                            validation_error_map[VALIDATION_ERROR_09c00132]);
        }

        // The union of all source regions, and the union of all destination regions, specified by the elements of regions,
        // must not overlap in memory
        if (src_image_state->image == dst_image_state->image) {
            for (uint32_t j = 0; j < region_count; j++) {
                if (RegionIntersects(&regions[i], &regions[j], src_image_state->createInfo.imageType)) {
                    std::stringstream ss;
                    ss << "vkCmdCopyImage: pRegions[" << i << "] src overlaps with pRegions[" << j << "].";
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_190000f8, "IMAGE", "%s. %s",
                                    ss.str().c_str(), validation_error_map[VALIDATION_ERROR_190000f8]);
                }
            }
        }
    }

    // The formats of src_image and dst_image must be compatible. Formats are considered compatible if their texel size in bytes
    // is the same between both formats. For example, VK_FORMAT_R8G8B8A8_UNORM is compatible with VK_FORMAT_R32_UINT because
    // because both texels are 4 bytes in size. Depth/stencil formats must match exactly.
    if (FormatIsDepthOrStencil(src_image_state->createInfo.format) || FormatIsDepthOrStencil(dst_image_state->createInfo.format)) {
        if (src_image_state->createInfo.format != dst_image_state->createInfo.format) {
            char const str[] = "vkCmdCopyImage called with unmatched source and dest image depth/stencil formats.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_FORMAT, "IMAGE", str);
        }
    } else {
        size_t srcSize = FormatSize(src_image_state->createInfo.format);
        size_t destSize = FormatSize(dst_image_state->createInfo.format);
        if (srcSize != destSize) {
            char const str[] = "vkCmdCopyImage called with unmatched source and dest image format sizes.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_1900010e, "IMAGE", "%s. %s", str,
                            validation_error_map[VALIDATION_ERROR_1900010e]);
        }
    }

    // Source and dest image sample counts must match
    if (src_image_state->createInfo.samples != dst_image_state->createInfo.samples) {
        char const str[] = "vkCmdCopyImage() called on image pair with non-identical sample counts.";
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(command_buffer), __LINE__, VALIDATION_ERROR_19000110, "IMAGE", "%s %s", str,
                        validation_error_map[VALIDATION_ERROR_19000110]);
    }

    skip |= ValidateMemoryIsBoundToImage(device_data, src_image_state, "vkCmdCopyImage()", VALIDATION_ERROR_190000fe);
    skip |= ValidateMemoryIsBoundToImage(device_data, dst_image_state, "vkCmdCopyImage()", VALIDATION_ERROR_19000108);
    // Validate that SRC & DST images have correct usage flags set
    skip |= ValidateImageUsageFlags(device_data, src_image_state, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true, VALIDATION_ERROR_190000fc,
                                    "vkCmdCopyImage()", "VK_IMAGE_USAGE_TRANSFER_SRC_BIT");
    skip |= ValidateImageUsageFlags(device_data, dst_image_state, VK_IMAGE_USAGE_TRANSFER_DST_BIT, true, VALIDATION_ERROR_19000106,
                                    "vkCmdCopyImage()", "VK_IMAGE_USAGE_TRANSFER_DST_BIT");
    skip |= ValidateCmdQueueFlags(device_data, cb_node, "vkCmdCopyImage()",
                                  VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, VALIDATION_ERROR_19002415);
    skip |= ValidateCmd(device_data, cb_node, CMD_COPYIMAGE, "vkCmdCopyImage()");
    skip |= insideRenderPass(device_data, cb_node, "vkCmdCopyImage()", VALIDATION_ERROR_19000017);
    bool hit_error = false;
    for (uint32_t i = 0; i < region_count; ++i) {
        skip |= VerifyImageLayout(device_data, cb_node, src_image_state, regions[i].srcSubresource, src_image_layout,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, "vkCmdCopyImage()", VALIDATION_ERROR_19000102, &hit_error);
        skip |= VerifyImageLayout(device_data, cb_node, dst_image_state, regions[i].dstSubresource, dst_image_layout,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, "vkCmdCopyImage()", VALIDATION_ERROR_1900010c, &hit_error);
        skip |= ValidateCopyImageTransferGranularityRequirements(device_data, cb_node, src_image_state, dst_image_state,
                                                                 &regions[i], i, "vkCmdCopyImage()");
    }

    return skip;
}

void PreCallRecordCmdCopyImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                               IMAGE_STATE *dst_image_state, uint32_t region_count, const VkImageCopy *regions,
                               VkImageLayout src_image_layout, VkImageLayout dst_image_layout) {
    // Make sure that all image slices are updated to correct layout
    for (uint32_t i = 0; i < region_count; ++i) {
        SetImageLayout(device_data, cb_node, src_image_state, regions[i].srcSubresource, src_image_layout);
        SetImageLayout(device_data, cb_node, dst_image_state, regions[i].dstSubresource, dst_image_layout);
    }
    // Update bindings between images and cmd buffer
    AddCommandBufferBindingImage(device_data, cb_node, src_image_state);
    AddCommandBufferBindingImage(device_data, cb_node, dst_image_state);
    std::function<bool()> function = [=]() { return ValidateImageMemoryIsValid(device_data, src_image_state, "vkCmdCopyImage()"); };
    cb_node->queue_submit_functions.push_back(function);
    function = [=]() {
        SetImageMemoryValid(device_data, dst_image_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
}

// Returns true if sub_rect is entirely contained within rect
static inline bool ContainsRect(VkRect2D rect, VkRect2D sub_rect) {
    if ((sub_rect.offset.x < rect.offset.x) || (sub_rect.offset.x + sub_rect.extent.width > rect.offset.x + rect.extent.width) ||
        (sub_rect.offset.y < rect.offset.y) || (sub_rect.offset.y + sub_rect.extent.height > rect.offset.y + rect.extent.height))
        return false;
    return true;
}

bool PreCallValidateCmdClearAttachments(layer_data *device_data, VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                        const VkClearAttachment *pAttachments, uint32_t rectCount, const VkClearRect *pRects) {
    GLOBAL_CB_NODE *cb_node = GetCBNode(device_data, commandBuffer);
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    bool skip = false;
    if (cb_node) {
        skip |= ValidateCmdQueueFlags(device_data, cb_node, "vkCmdClearAttachments()", VK_QUEUE_GRAPHICS_BIT,
                                      VALIDATION_ERROR_18602415);
        skip |= ValidateCmd(device_data, cb_node, CMD_CLEARATTACHMENTS, "vkCmdClearAttachments()");
        // Warn if this is issued prior to Draw Cmd and clearing the entire attachment
        if (!cb_node->hasDrawCmd && (cb_node->activeRenderPassBeginInfo.renderArea.extent.width == pRects[0].rect.extent.width) &&
            (cb_node->activeRenderPassBeginInfo.renderArea.extent.height == pRects[0].rect.extent.height)) {
            // There are times where app needs to use ClearAttachments (generally when reusing a buffer inside of a render pass)
            // This warning should be made more specific. It'd be best to avoid triggering this test if it's a use that must call
            // CmdClearAttachments.
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(commandBuffer), 0, DRAWSTATE_CLEAR_CMD_BEFORE_DRAW, "DS",
                        "vkCmdClearAttachments() issued on command buffer object 0x%p prior to any Draw Cmds."
                        " It is recommended you use RenderPass LOAD_OP_CLEAR on Attachments prior to any Draw.",
                        commandBuffer);
        }
        skip |= outsideRenderPass(device_data, cb_node, "vkCmdClearAttachments()", VALIDATION_ERROR_18600017);
    }

    // Validate that attachment is in reference list of active subpass
    if (cb_node->activeRenderPass) {
        const VkRenderPassCreateInfo *renderpass_create_info = cb_node->activeRenderPass->createInfo.ptr();
        const VkSubpassDescription *subpass_desc = &renderpass_create_info->pSubpasses[cb_node->activeSubpass];
        auto framebuffer = GetFramebufferState(device_data, cb_node->activeFramebuffer);

        for (uint32_t i = 0; i < attachmentCount; i++) {
            auto clear_desc = &pAttachments[i];
            VkImageView image_view = VK_NULL_HANDLE;

            if (0 == clear_desc->aspectMask) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_01c00c03, "IMAGE", "%s",
                                validation_error_map[VALIDATION_ERROR_01c00c03]);
            } else if (clear_desc->aspectMask & VK_IMAGE_ASPECT_METADATA_BIT) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_01c00028, "IMAGE", "%s",
                                validation_error_map[VALIDATION_ERROR_01c00028]);
            } else if (clear_desc->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
                if (clear_desc->colorAttachment >= subpass_desc->colorAttachmentCount) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_1860001e, "DS",
                                    "vkCmdClearAttachments() color attachment index %d out of range for active subpass %d. %s",
                                    clear_desc->colorAttachment, cb_node->activeSubpass,
                                    validation_error_map[VALIDATION_ERROR_1860001e]);
                } else if (subpass_desc->pColorAttachments[clear_desc->colorAttachment].attachment == VK_ATTACHMENT_UNUSED) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                                    VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, HandleToUint64(commandBuffer), __LINE__,
                                    DRAWSTATE_MISSING_ATTACHMENT_REFERENCE, "DS",
                                    "vkCmdClearAttachments() color attachment index %d is VK_ATTACHMENT_UNUSED; ignored.",
                                    clear_desc->colorAttachment);
                } else {
                    image_view = framebuffer->createInfo
                                     .pAttachments[subpass_desc->pColorAttachments[clear_desc->colorAttachment].attachment];
                }
                if ((clear_desc->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) ||
                    (clear_desc->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)) {
                    char const str[] =
                        "vkCmdClearAttachments aspectMask [%d] must set only VK_IMAGE_ASPECT_COLOR_BIT of a color attachment. %s";
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_01c00026, "IMAGE", str, i,
                                    validation_error_map[VALIDATION_ERROR_01c00026]);
                }
            } else {  // Must be depth and/or stencil
                if (((clear_desc->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != VK_IMAGE_ASPECT_DEPTH_BIT) &&
                    ((clear_desc->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != VK_IMAGE_ASPECT_STENCIL_BIT)) {
                    char const str[] = "vkCmdClearAttachments aspectMask [%d] is not a valid combination of bits. %s";
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_01c00c01, "IMAGE", str, i,
                                    validation_error_map[VALIDATION_ERROR_01c00c01]);
                }
                if (!subpass_desc->pDepthStencilAttachment ||
                    (subpass_desc->pDepthStencilAttachment->attachment == VK_ATTACHMENT_UNUSED)) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(commandBuffer), __LINE__, DRAWSTATE_MISSING_ATTACHMENT_REFERENCE, "DS",
                        "vkCmdClearAttachments() depth/stencil clear with no depth/stencil attachment in subpass; ignored");
                } else {
                    image_view = framebuffer->createInfo.pAttachments[subpass_desc->pDepthStencilAttachment->attachment];
                }
            }
            if (image_view) {
                auto image_view_state = GetImageViewState(device_data, image_view);
                for (uint32_t j = 0; j < rectCount; j++) {
                    // The rectangular region specified by a given element of pRects must be contained within the render area of
                    // the current render pass instance
                    // TODO: This check should be moved to CmdExecuteCommands or QueueSubmit to cover secondary CB cases
                    if ((cb_node->createInfo.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY) &&
                        (false == ContainsRect(cb_node->activeRenderPassBeginInfo.renderArea, pRects[j].rect))) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                        HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_18600020, "DS",
                                        "vkCmdClearAttachments(): The area defined by pRects[%d] is not contained in the area of "
                                        "the current render pass instance. %s",
                                        j, validation_error_map[VALIDATION_ERROR_18600020]);
                    }
                    // The layers specified by a given element of pRects must be contained within every attachment that
                    // pAttachments refers to
                    auto attachment_layer_count = image_view_state->create_info.subresourceRange.layerCount;
                    if ((pRects[j].baseArrayLayer >= attachment_layer_count) ||
                        (pRects[j].baseArrayLayer + pRects[j].layerCount > attachment_layer_count)) {
                        skip |=
                            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(commandBuffer), __LINE__, VALIDATION_ERROR_18600022, "DS",
                                    "vkCmdClearAttachments(): The layers defined in pRects[%d] are not contained in the layers of "
                                    "pAttachment[%d]. %s",
                                    j, i, validation_error_map[VALIDATION_ERROR_18600022]);
                    }
                }
            }
        }
    }
    return skip;
}

bool PreCallValidateCmdResolveImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                    IMAGE_STATE *dst_image_state, uint32_t regionCount, const VkImageResolve *pRegions) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    if (cb_node && src_image_state && dst_image_state) {
        skip |= ValidateMemoryIsBoundToImage(device_data, src_image_state, "vkCmdResolveImage()", VALIDATION_ERROR_1c800200);
        skip |= ValidateMemoryIsBoundToImage(device_data, dst_image_state, "vkCmdResolveImage()", VALIDATION_ERROR_1c800204);
        skip |=
            ValidateCmdQueueFlags(device_data, cb_node, "vkCmdResolveImage()", VK_QUEUE_GRAPHICS_BIT, VALIDATION_ERROR_1c802415);
        skip |= ValidateCmd(device_data, cb_node, CMD_RESOLVEIMAGE, "vkCmdResolveImage()");
        skip |= insideRenderPass(device_data, cb_node, "vkCmdResolveImage()", VALIDATION_ERROR_1c800017);

        // For each region, the number of layers in the image subresource should not be zero
        // For each region, src and dest image aspect must be color only
        for (uint32_t i = 0; i < regionCount; i++) {
            if (pRegions[i].srcSubresource.layerCount == 0) {
                char const str[] = "vkCmdResolveImage: number of layers in source subresource is zero";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_ASPECT, "IMAGE", str);
            }
            if (pRegions[i].dstSubresource.layerCount == 0) {
                char const str[] = "vkCmdResolveImage: number of layers in destination subresource is zero";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_ASPECT, "IMAGE", str);
            }
            if (pRegions[i].srcSubresource.layerCount != pRegions[i].dstSubresource.layerCount) {
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_0a200216, "IMAGE",
                    "vkCmdResolveImage: layerCount in source and destination subresource of pRegions[%d] does not match. %s", i,
                    validation_error_map[VALIDATION_ERROR_0a200216]);
            }
            if ((pRegions[i].srcSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT) ||
                (pRegions[i].dstSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT)) {
                char const str[] =
                    "vkCmdResolveImage: src and dest aspectMasks for each region must specify only VK_IMAGE_ASPECT_COLOR_BIT";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_0a200214, "IMAGE", "%s. %s", str,
                                validation_error_map[VALIDATION_ERROR_0a200214]);
            }
        }

        if (src_image_state->createInfo.format != dst_image_state->createInfo.format) {
            char const str[] = "vkCmdResolveImage called with unmatched source and dest formats.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_FORMAT, "IMAGE", str);
        }
        if (src_image_state->createInfo.imageType != dst_image_state->createInfo.imageType) {
            char const str[] = "vkCmdResolveImage called with unmatched source and dest image types.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_TYPE, "IMAGE", str);
        }
        if (src_image_state->createInfo.samples == VK_SAMPLE_COUNT_1_BIT) {
            char const str[] = "vkCmdResolveImage called with source sample count less than 2.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_1c800202, "IMAGE", "%s. %s", str,
                            validation_error_map[VALIDATION_ERROR_1c800202]);
        }
        if (dst_image_state->createInfo.samples != VK_SAMPLE_COUNT_1_BIT) {
            char const str[] = "vkCmdResolveImage called with dest sample count greater than 1.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_1c800206, "IMAGE", "%s. %s", str,
                            validation_error_map[VALIDATION_ERROR_1c800206]);
        }
        // TODO: Need to validate image layouts, which will include layout validation for shared presentable images
    } else {
        assert(0);
    }
    return skip;
}

void PreCallRecordCmdResolveImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                  IMAGE_STATE *dst_image_state) {
    // Update bindings between images and cmd buffer
    AddCommandBufferBindingImage(device_data, cb_node, src_image_state);
    AddCommandBufferBindingImage(device_data, cb_node, dst_image_state);

    std::function<bool()> function = [=]() {
        return ValidateImageMemoryIsValid(device_data, src_image_state, "vkCmdResolveImage()");
    };
    cb_node->queue_submit_functions.push_back(function);
    function = [=]() {
        SetImageMemoryValid(device_data, dst_image_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
}

bool PreCallValidateCmdBlitImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                 IMAGE_STATE *dst_image_state, uint32_t regionCount, const VkImageBlit *pRegions, VkFilter filter) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    bool skip = false;
    if (cb_node && src_image_state && dst_image_state) {
        skip |= ValidateImageSampleCount(device_data, src_image_state, VK_SAMPLE_COUNT_1_BIT, "vkCmdBlitImage(): srcImage",
                                         VALIDATION_ERROR_184001d2);
        skip |= ValidateImageSampleCount(device_data, dst_image_state, VK_SAMPLE_COUNT_1_BIT, "vkCmdBlitImage(): dstImage",
                                         VALIDATION_ERROR_184001d4);
        skip |= ValidateMemoryIsBoundToImage(device_data, src_image_state, "vkCmdBlitImage()", VALIDATION_ERROR_184001b8);
        skip |= ValidateMemoryIsBoundToImage(device_data, dst_image_state, "vkCmdBlitImage()", VALIDATION_ERROR_184001c2);
        skip |= ValidateImageUsageFlags(device_data, src_image_state, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true,
                                        VALIDATION_ERROR_184001b6, "vkCmdBlitImage()", "VK_IMAGE_USAGE_TRANSFER_SRC_BIT");
        skip |= ValidateImageUsageFlags(device_data, dst_image_state, VK_IMAGE_USAGE_TRANSFER_DST_BIT, true,
                                        VALIDATION_ERROR_184001c0, "vkCmdBlitImage()", "VK_IMAGE_USAGE_TRANSFER_DST_BIT");
        skip |= ValidateCmdQueueFlags(device_data, cb_node, "vkCmdBlitImage()", VK_QUEUE_GRAPHICS_BIT, VALIDATION_ERROR_18402415);
        skip |= ValidateCmd(device_data, cb_node, CMD_BLITIMAGE, "vkCmdBlitImage()");
        skip |= insideRenderPass(device_data, cb_node, "vkCmdBlitImage()", VALIDATION_ERROR_18400017);
        // TODO: Need to validate image layouts, which will include layout validation for shared presentable images

        VkFormat src_format = src_image_state->createInfo.format;
        VkFormat dst_format = dst_image_state->createInfo.format;
        VkImageType src_type = src_image_state->createInfo.imageType;
        VkImageType dst_type = dst_image_state->createInfo.imageType;

        const VkFormatProperties *props = GetFormatProperties(device_data, src_format);
        VkImageTiling tiling = src_image_state->createInfo.tiling;
        VkFormatFeatureFlags flags =
            (tiling == VK_IMAGE_TILING_LINEAR ? props->linearTilingFeatures : props->optimalTilingFeatures);
        if (VK_FORMAT_FEATURE_BLIT_SRC_BIT != (flags & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001b4, "IMAGE",
                            "vkCmdBlitImage: source image format %s does not support VK_FORMAT_FEATURE_BLIT_SRC_BIT feature. %s",
                            string_VkFormat(src_format), validation_error_map[VALIDATION_ERROR_184001b4]);
        }

        if ((VK_FILTER_LINEAR == filter) &&
            (VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT != (flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001d6, "IMAGE",
                            "vkCmdBlitImage: source image format %s does not support linear filtering. %s",
                            string_VkFormat(src_format), validation_error_map[VALIDATION_ERROR_184001d6]);
        }

        if ((VK_FILTER_CUBIC_IMG == filter) && (VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG !=
                                                (flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG))) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001d8, "IMAGE",
                            "vkCmdBlitImage: source image format %s does not support cubic filtering. %s",
                            string_VkFormat(src_format), validation_error_map[VALIDATION_ERROR_184001d8]);
        }

        if ((VK_FILTER_CUBIC_IMG == filter) && (VK_IMAGE_TYPE_3D != src_type)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001da, "IMAGE",
                            "vkCmdBlitImage: source image type must be VK_IMAGE_TYPE_3D when cubic filtering is specified. %s",
                            validation_error_map[VALIDATION_ERROR_184001da]);
        }

        props = GetFormatProperties(device_data, dst_format);
        tiling = dst_image_state->createInfo.tiling;
        flags = (tiling == VK_IMAGE_TILING_LINEAR ? props->linearTilingFeatures : props->optimalTilingFeatures);
        if (VK_FORMAT_FEATURE_BLIT_DST_BIT != (flags & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001be, "IMAGE",
                        "vkCmdBlitImage: destination image format %s does not support VK_FORMAT_FEATURE_BLIT_DST_BIT feature. %s",
                        string_VkFormat(dst_format), validation_error_map[VALIDATION_ERROR_184001be]);
        }

        if ((VK_SAMPLE_COUNT_1_BIT != src_image_state->createInfo.samples) ||
            (VK_SAMPLE_COUNT_1_BIT != dst_image_state->createInfo.samples)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001c8, "IMAGE",
                            "vkCmdBlitImage: source or dest image has sample count other than VK_SAMPLE_COUNT_1_BIT. %s",
                            validation_error_map[VALIDATION_ERROR_184001c8]);
        }

        // Validate consistency for unsigned formats
        if (FormatIsUInt(src_format) != FormatIsUInt(dst_format)) {
            std::stringstream ss;
            ss << "vkCmdBlitImage: If one of srcImage and dstImage images has unsigned integer format, "
               << "the other one must also have unsigned integer format.  "
               << "Source format is " << string_VkFormat(src_format) << " Destination format is " << string_VkFormat(dst_format);
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001cc, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_184001cc]);
        }

        // Validate consistency for signed formats
        if (FormatIsSInt(src_format) != FormatIsSInt(dst_format)) {
            std::stringstream ss;
            ss << "vkCmdBlitImage: If one of srcImage and dstImage images has signed integer format, "
               << "the other one must also have signed integer format.  "
               << "Source format is " << string_VkFormat(src_format) << " Destination format is " << string_VkFormat(dst_format);
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001ca, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_184001ca]);
        }

        // Validate filter for Depth/Stencil formats
        if (FormatIsDepthOrStencil(src_format) && (filter != VK_FILTER_NEAREST)) {
            std::stringstream ss;
            ss << "vkCmdBlitImage: If the format of srcImage is a depth, stencil, or depth stencil "
               << "then filter must be VK_FILTER_NEAREST.";
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001d0, "IMAGE", "%s. %s",
                            ss.str().c_str(), validation_error_map[VALIDATION_ERROR_184001d0]);
        }

        // Validate aspect bits and formats for depth/stencil images
        if (FormatIsDepthOrStencil(src_format) || FormatIsDepthOrStencil(dst_format)) {
            if (src_format != dst_format) {
                std::stringstream ss;
                ss << "vkCmdBlitImage: If one of srcImage and dstImage images has a format of depth, stencil or depth "
                   << "stencil, the other one must have exactly the same format.  "
                   << "Source format is " << string_VkFormat(src_format) << " Destination format is "
                   << string_VkFormat(dst_format);
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001ce, "IMAGE", "%s. %s",
                                ss.str().c_str(), validation_error_map[VALIDATION_ERROR_184001ce]);
            }

#if 0  // TODO: Cannot find VU statements or spec language for these in CmdBlitImage. Verify or remove.
            for (uint32_t i = 0; i < regionCount; i++) {
                VkImageAspectFlags srcAspect = pRegions[i].srcSubresource.aspectMask;

                if (FormatIsDepthAndStencil(src_format)) {
                    if ((srcAspect != VK_IMAGE_ASPECT_DEPTH_BIT) && (srcAspect != VK_IMAGE_ASPECT_STENCIL_BIT)) {
                        std::stringstream ss;
                        ss << "vkCmdBlitImage: Combination depth/stencil image formats must have only one of "
                            "VK_IMAGE_ASPECT_DEPTH_BIT "
                            << "and VK_IMAGE_ASPECT_STENCIL_BIT set in srcImage and dstImage";
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE",
                            "%s", ss.str().c_str());
                    }
                }
                else if (FormatIsStencilOnly(src_format)) {
                    if (srcAspect != VK_IMAGE_ASPECT_STENCIL_BIT) {
                        std::stringstream ss;
                        ss << "vkCmdBlitImage: Stencil-only image formats must have only the VK_IMAGE_ASPECT_STENCIL_BIT "
                            << "set in both the srcImage and dstImage";
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE",
                            "%s", ss.str().c_str());
                    }
                }
                else if (FormatIsDepthOnly(src_format)) {
                    if (srcAspect != VK_IMAGE_ASPECT_DEPTH_BIT) {
                        std::stringstream ss;
                        ss << "vkCmdBlitImage: Depth-only image formats must have only the VK_IMAGE_ASPECT_DEPTH "
                            << "set in both the srcImage and dstImage";
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_ASPECT, "IMAGE",
                            "%s", ss.str().c_str());
                    }
                }
            }
#endif
        }  // Depth or Stencil

        // Do per-region checks
        for (uint32_t i = 0; i < regionCount; i++) {
            VkImageBlit rgn = pRegions[i];

            // Warn for zero-sized regions
            if ((rgn.srcOffsets[0].x == rgn.srcOffsets[1].x) || (rgn.srcOffsets[0].y == rgn.srcOffsets[1].y) ||
                (rgn.srcOffsets[0].z == rgn.srcOffsets[1].z)) {
                std::stringstream ss;
                ss << "vkCmdBlitImage: pRegions[" << i << "].srcOffsets specify a zero-volume area.";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_INVALID_EXTENTS, "IMAGE", "%s",
                                ss.str().c_str());
            }
            if ((rgn.dstOffsets[0].x == rgn.dstOffsets[1].x) || (rgn.dstOffsets[0].y == rgn.dstOffsets[1].y) ||
                (rgn.dstOffsets[0].z == rgn.dstOffsets[1].z)) {
                std::stringstream ss;
                ss << "vkCmdBlitImage: pRegions[" << i << "].dstOffsets specify a zero-volume area.";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_INVALID_EXTENTS, "IMAGE", "%s",
                                ss.str().c_str());
            }
            if (rgn.srcSubresource.layerCount == 0) {
                char const str[] = "vkCmdBlitImage: number of layers in source subresource is zero";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_ASPECT, "IMAGE", str);
            }
            if (rgn.dstSubresource.layerCount == 0) {
                char const str[] = "vkCmdBlitImage: number of layers in destination subresource is zero";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, DRAWSTATE_MISMATCHED_IMAGE_ASPECT, "IMAGE", str);
            }

            // Check that src/dst layercounts match
            if (rgn.srcSubresource.layerCount != rgn.dstSubresource.layerCount) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001de, "IMAGE",
                            "vkCmdBlitImage: layerCount in source and destination subresource of pRegions[%d] does not match. %s",
                            i, validation_error_map[VALIDATION_ERROR_09a001de]);
            }

            if (rgn.srcSubresource.aspectMask != rgn.dstSubresource.aspectMask) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001dc, "IMAGE",
                                "vkCmdBlitImage: aspectMask members for pRegion[%d] do not match. %s", i,
                                validation_error_map[VALIDATION_ERROR_09a001dc]);
            }

            if (!VerifyAspectsPresent(rgn.srcSubresource.aspectMask, src_format)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001e2, "IMAGE",
                                "vkCmdBlitImage: region [%d] source aspectMask (0x%x) specifies aspects not present in source "
                                "image format %s. %s",
                                i, rgn.srcSubresource.aspectMask, string_VkFormat(src_format),
                                validation_error_map[VALIDATION_ERROR_09a001e2]);
            }

            if (!VerifyAspectsPresent(rgn.dstSubresource.aspectMask, dst_format)) {
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001e4, "IMAGE",
                    "vkCmdBlitImage: region [%d] dest aspectMask (0x%x) specifies aspects not present in dest image format %s. %s",
                    i, rgn.dstSubresource.aspectMask, string_VkFormat(dst_format), validation_error_map[VALIDATION_ERROR_09a001e4]);
            }

            // Validate source image offsets
            VkExtent3D src_extent = GetImageSubresourceExtent(src_image_state, &(rgn.srcSubresource));
            if (VK_IMAGE_TYPE_1D == src_type) {
                if ((0 != rgn.srcOffsets[0].y) || (1 != rgn.srcOffsets[1].y)) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001ea, "IMAGE",
                                    "vkCmdBlitImage: region [%d], source image of type VK_IMAGE_TYPE_1D with srcOffset[].y values "
                                    "of (%1d, %1d). These must be (0, 1). %s",
                                    i, rgn.srcOffsets[0].y, rgn.srcOffsets[1].y, validation_error_map[VALIDATION_ERROR_09a001ea]);
                }
            }

            if ((VK_IMAGE_TYPE_1D == src_type) || (VK_IMAGE_TYPE_2D == src_type)) {
                if ((0 != rgn.srcOffsets[0].z) || (1 != rgn.srcOffsets[1].z)) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001ee, "IMAGE",
                                    "vkCmdBlitImage: region [%d], source image of type VK_IMAGE_TYPE_1D or VK_IMAGE_TYPE_2D with "
                                    "srcOffset[].z values of (%1d, %1d). These must be (0, 1). %s",
                                    i, rgn.srcOffsets[0].z, rgn.srcOffsets[1].z, validation_error_map[VALIDATION_ERROR_09a001ee]);
                }
            }

            bool oob = false;
            if ((rgn.srcOffsets[0].x < 0) || (rgn.srcOffsets[0].x > static_cast<int32_t>(src_extent.width)) ||
                (rgn.srcOffsets[1].x < 0) || (rgn.srcOffsets[1].x > static_cast<int32_t>(src_extent.width))) {
                oob = true;
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001e6, "IMAGE",
                    "vkCmdBlitImage: region [%d] srcOffset[].x values (%1d, %1d) exceed srcSubresource width extent (%1d). %s", i,
                    rgn.srcOffsets[0].x, rgn.srcOffsets[1].x, src_extent.width, validation_error_map[VALIDATION_ERROR_09a001e6]);
            }
            if ((rgn.srcOffsets[0].y < 0) || (rgn.srcOffsets[0].y > static_cast<int32_t>(src_extent.height)) ||
                (rgn.srcOffsets[1].y < 0) || (rgn.srcOffsets[1].y > static_cast<int32_t>(src_extent.height))) {
                oob = true;
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001e8, "IMAGE",
                    "vkCmdBlitImage: region [%d] srcOffset[].y values (%1d, %1d) exceed srcSubresource height extent (%1d). %s", i,
                    rgn.srcOffsets[0].y, rgn.srcOffsets[1].y, src_extent.height, validation_error_map[VALIDATION_ERROR_09a001e8]);
            }
            if ((rgn.srcOffsets[0].z < 0) || (rgn.srcOffsets[0].z > static_cast<int32_t>(src_extent.depth)) ||
                (rgn.srcOffsets[1].z < 0) || (rgn.srcOffsets[1].z > static_cast<int32_t>(src_extent.depth))) {
                oob = true;
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001ec, "IMAGE",
                    "vkCmdBlitImage: region [%d] srcOffset[].z values (%1d, %1d) exceed srcSubresource depth extent (%1d). %s", i,
                    rgn.srcOffsets[0].z, rgn.srcOffsets[1].z, src_extent.depth, validation_error_map[VALIDATION_ERROR_09a001ec]);
            }
            if (rgn.srcSubresource.mipLevel >= src_image_state->createInfo.mipLevels) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001ae, "IMAGE",
                                "vkCmdBlitImage: region [%d] source image, attempt to access a non-existant mip level %1d. %s", i,
                                rgn.srcSubresource.mipLevel, validation_error_map[VALIDATION_ERROR_184001ae]);
            } else if (oob) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001ae, "IMAGE",
                                "vkCmdBlitImage: region [%d] source image blit region exceeds image dimensions. %s", i,
                                validation_error_map[VALIDATION_ERROR_184001ae]);
            }

            // Validate dest image offsets
            VkExtent3D dst_extent = GetImageSubresourceExtent(dst_image_state, &(rgn.dstSubresource));
            if (VK_IMAGE_TYPE_1D == dst_type) {
                if ((0 != rgn.dstOffsets[0].y) || (1 != rgn.dstOffsets[1].y)) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001f4, "IMAGE",
                                    "vkCmdBlitImage: region [%d], dest image of type VK_IMAGE_TYPE_1D with dstOffset[].y values of "
                                    "(%1d, %1d). These must be (0, 1). %s",
                                    i, rgn.dstOffsets[0].y, rgn.dstOffsets[1].y, validation_error_map[VALIDATION_ERROR_09a001f4]);
                }
            }

            if ((VK_IMAGE_TYPE_1D == dst_type) || (VK_IMAGE_TYPE_2D == dst_type)) {
                if ((0 != rgn.dstOffsets[0].z) || (1 != rgn.dstOffsets[1].z)) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001f8, "IMAGE",
                                    "vkCmdBlitImage: region [%d], dest image of type VK_IMAGE_TYPE_1D or VK_IMAGE_TYPE_2D with "
                                    "dstOffset[].z values of (%1d, %1d). These must be (0, 1). %s",
                                    i, rgn.dstOffsets[0].z, rgn.dstOffsets[1].z, validation_error_map[VALIDATION_ERROR_09a001f8]);
                }
            }

            oob = false;
            if ((rgn.dstOffsets[0].x < 0) || (rgn.dstOffsets[0].x > static_cast<int32_t>(dst_extent.width)) ||
                (rgn.dstOffsets[1].x < 0) || (rgn.dstOffsets[1].x > static_cast<int32_t>(dst_extent.width))) {
                oob = true;
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001f0, "IMAGE",
                    "vkCmdBlitImage: region [%d] dstOffset[].x values (%1d, %1d) exceed dstSubresource width extent (%1d). %s", i,
                    rgn.dstOffsets[0].x, rgn.dstOffsets[1].x, dst_extent.width, validation_error_map[VALIDATION_ERROR_09a001f0]);
            }
            if ((rgn.dstOffsets[0].y < 0) || (rgn.dstOffsets[0].y > static_cast<int32_t>(dst_extent.height)) ||
                (rgn.dstOffsets[1].y < 0) || (rgn.dstOffsets[1].y > static_cast<int32_t>(dst_extent.height))) {
                oob = true;
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001f2, "IMAGE",
                    "vkCmdBlitImage: region [%d] dstOffset[].y values (%1d, %1d) exceed dstSubresource height extent (%1d). %s", i,
                    rgn.dstOffsets[0].y, rgn.dstOffsets[1].y, dst_extent.height, validation_error_map[VALIDATION_ERROR_09a001f2]);
            }
            if ((rgn.dstOffsets[0].z < 0) || (rgn.dstOffsets[0].z > static_cast<int32_t>(dst_extent.depth)) ||
                (rgn.dstOffsets[1].z < 0) || (rgn.dstOffsets[1].z > static_cast<int32_t>(dst_extent.depth))) {
                oob = true;
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001f6, "IMAGE",
                    "vkCmdBlitImage: region [%d] dstOffset[].z values (%1d, %1d) exceed dstSubresource depth extent (%1d). %s", i,
                    rgn.dstOffsets[0].z, rgn.dstOffsets[1].z, dst_extent.depth, validation_error_map[VALIDATION_ERROR_09a001f6]);
            }
            if (rgn.dstSubresource.mipLevel >= dst_image_state->createInfo.mipLevels) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001b0, "IMAGE",
                                "vkCmdBlitImage: region [%d] destination image, attempt to access a non-existant mip level %1d. %s",
                                i, rgn.dstSubresource.mipLevel, validation_error_map[VALIDATION_ERROR_184001b0]);
            } else if (oob) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_184001b0, "IMAGE",
                                "vkCmdBlitImage: region [%d] destination image blit region exceeds image dimensions. %s", i,
                                validation_error_map[VALIDATION_ERROR_184001b0]);
            }

            if ((VK_IMAGE_TYPE_3D == src_type) || (VK_IMAGE_TYPE_3D == dst_type)) {
                if ((0 != rgn.srcSubresource.baseArrayLayer) || (1 != rgn.srcSubresource.layerCount) ||
                    (0 != rgn.dstSubresource.baseArrayLayer) || (1 != rgn.dstSubresource.layerCount)) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_09a001e0, "IMAGE",
                                    "vkCmdBlitImage: region [%d] blit to/from a 3D image type with a non-zero baseArrayLayer, or a "
                                    "layerCount other than 1. %s",
                                    i, validation_error_map[VALIDATION_ERROR_09a001e0]);
                }
            }
        }  // per-region checks
    } else {
        assert(0);
    }
    return skip;
}

void PreCallRecordCmdBlitImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                               IMAGE_STATE *dst_image_state) {
    // Update bindings between images and cmd buffer
    AddCommandBufferBindingImage(device_data, cb_node, src_image_state);
    AddCommandBufferBindingImage(device_data, cb_node, dst_image_state);

    std::function<bool()> function = [=]() { return ValidateImageMemoryIsValid(device_data, src_image_state, "vkCmdBlitImage()"); };
    cb_node->queue_submit_functions.push_back(function);
    function = [=]() {
        SetImageMemoryValid(device_data, dst_image_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
}

// This validates that the initial layout specified in the command buffer for
// the IMAGE is the same
// as the global IMAGE layout
bool ValidateCmdBufImageLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB,
                                std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> const & globalImageLayoutMap,
                                std::unordered_map<ImageSubresourcePair, IMAGE_LAYOUT_NODE> & overlayLayoutMap) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    for (auto cb_image_data : pCB->imageLayoutMap) {
        VkImageLayout imageLayout;

        if (FindLayout(overlayLayoutMap, cb_image_data.first, imageLayout) ||
            FindLayout(globalImageLayoutMap, cb_image_data.first, imageLayout)) {
            if (cb_image_data.second.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
                // TODO: Set memory invalid which is in mem_tracker currently
            } else if (imageLayout != cb_image_data.second.initialLayout) {
                if (cb_image_data.first.hasSubresource) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(pCB->commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Cannot submit cmd buffer using image (0x%" PRIx64
                                    ") [sub-resource: aspectMask 0x%X array layer %u, mip level %u], "
                                    "with layout %s when first use is %s.",
                                    HandleToUint64(cb_image_data.first.image), cb_image_data.first.subresource.aspectMask,
                                    cb_image_data.first.subresource.arrayLayer, cb_image_data.first.subresource.mipLevel,
                                    string_VkImageLayout(imageLayout), string_VkImageLayout(cb_image_data.second.initialLayout));
                } else {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                    HandleToUint64(pCB->commandBuffer), __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Cannot submit cmd buffer using image (0x%" PRIx64
                                    ") with layout %s when "
                                    "first use is %s.",
                                    HandleToUint64(cb_image_data.first.image), string_VkImageLayout(imageLayout),
                                    string_VkImageLayout(cb_image_data.second.initialLayout));
                }
            }
            SetLayout(overlayLayoutMap, cb_image_data.first, cb_image_data.second.layout);
        }
    }
    return skip;
}

void UpdateCmdBufImageLayouts(layer_data *device_data, GLOBAL_CB_NODE *pCB) {
    for (auto cb_image_data : pCB->imageLayoutMap) {
        VkImageLayout imageLayout;
        FindGlobalLayout(device_data, cb_image_data.first, imageLayout);
        SetGlobalLayout(device_data, cb_image_data.first, cb_image_data.second.layout);
    }
}

// Print readable FlagBits in FlagMask
static std::string string_VkAccessFlags(VkAccessFlags accessMask) {
    std::string result;
    std::string separator;

    if (accessMask == 0) {
        result = "[None]";
    } else {
        result = "[";
        for (auto i = 0; i < 32; i++) {
            if (accessMask & (1 << i)) {
                result = result + separator + string_VkAccessFlagBits((VkAccessFlagBits)(1 << i));
                separator = " | ";
            }
        }
        result = result + "]";
    }
    return result;
}

// AccessFlags MUST have 'required_bit' set, and may have one or more of 'optional_bits' set. If required_bit is zero, accessMask
// must have at least one of 'optional_bits' set
// TODO: Add tracking to ensure that at least one barrier has been set for these layout transitions
static bool ValidateMaskBits(core_validation::layer_data *device_data, VkCommandBuffer cmdBuffer, const VkAccessFlags &accessMask,
                             const VkImageLayout &layout, VkAccessFlags required_bit, VkAccessFlags optional_bits,
                             const char *type) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;

    if ((accessMask & required_bit) || (!required_bit && (accessMask & optional_bits))) {
        if (accessMask & ~(required_bit | optional_bits)) {
            // TODO: Verify against Valid Use
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cmdBuffer), __LINE__, DRAWSTATE_INVALID_BARRIER, "DS",
                            "Additional bits in %s accessMask 0x%X %s are specified when layout is %s.", type, accessMask,
                            string_VkAccessFlags(accessMask).c_str(), string_VkImageLayout(layout));
        }
    } else {
        if (!required_bit) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cmdBuffer), __LINE__, DRAWSTATE_INVALID_BARRIER, "DS",
                            "%s AccessMask %d %s must contain at least one of access bits %d "
                            "%s when layout is %s, unless the app has previously added a "
                            "barrier for this transition.",
                            type, accessMask, string_VkAccessFlags(accessMask).c_str(), optional_bits,
                            string_VkAccessFlags(optional_bits).c_str(), string_VkImageLayout(layout));
        } else {
            std::string opt_bits;
            if (optional_bits != 0) {
                std::stringstream ss;
                ss << optional_bits;
                opt_bits = "and may have optional bits " + ss.str() + ' ' + string_VkAccessFlags(optional_bits);
            }
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            HandleToUint64(cmdBuffer), __LINE__, DRAWSTATE_INVALID_BARRIER, "DS",
                            "%s AccessMask %d %s must have required access bit %d %s %s when "
                            "layout is %s, unless the app has previously added a barrier for "
                            "this transition.",
                            type, accessMask, string_VkAccessFlags(accessMask).c_str(), required_bit,
                            string_VkAccessFlags(required_bit).c_str(), opt_bits.c_str(), string_VkImageLayout(layout));
        }
    }
    return skip;
}

bool ValidateMaskBitsFromLayouts(core_validation::layer_data *device_data, VkCommandBuffer cmdBuffer,
                                 const VkAccessFlags &accessMask, const VkImageLayout &layout, const char *type) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    bool skip = false;
    switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: {
            skip |= ValidateMaskBits(device_data, cmdBuffer, accessMask, layout, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, type);
            break;
        }
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: {
            skip |= ValidateMaskBits(device_data, cmdBuffer, accessMask, layout, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, type);
            break;
        }
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: {
            skip |= ValidateMaskBits(device_data, cmdBuffer, accessMask, layout, VK_ACCESS_TRANSFER_WRITE_BIT, 0, type);
            break;
        }
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: {
            skip |= ValidateMaskBits(
                device_data, cmdBuffer, accessMask, layout, 0,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
                type);
            break;
        }
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: {
            skip |= ValidateMaskBits(device_data, cmdBuffer, accessMask, layout, 0,
                                     VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT, type);
            break;
        }
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: {
            skip |= ValidateMaskBits(device_data, cmdBuffer, accessMask, layout, VK_ACCESS_TRANSFER_READ_BIT, 0, type);
            break;
        }
        case VK_IMAGE_LAYOUT_UNDEFINED: {
            if (accessMask != 0) {
                // TODO: Verify against Valid Use section spec
                skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                                HandleToUint64(cmdBuffer), __LINE__, DRAWSTATE_INVALID_BARRIER, "DS",
                                "Additional bits in %s accessMask 0x%X %s are specified when layout is %s.", type, accessMask,
                                string_VkAccessFlags(accessMask).c_str(), string_VkImageLayout(layout));
            }
            break;
        }
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        // Notes: QueuePresentKHR performs automatic visibility operations,
        // so the app is /NOT/ required to include VK_ACCESS_MEMORY_READ_BIT
        // when transitioning to this layout.
        //
        // When transitioning /from/ this layout, the application needs to
        // avoid only a WAR hazard -- any writes need to be ordered after
        // the PE's reads. There is no need for a memory dependency for this
        // case.
        // Intentionally fall through

        case VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR:
        // Todo -- shouldn't be valid unless extension is enabled
        // Intentionally fall through

        case VK_IMAGE_LAYOUT_GENERAL:
        default: { break; }
    }
    return skip;
}

// ValidateLayoutVsAttachmentDescription is a general function where we can validate various state associated with the
// VkAttachmentDescription structs that are used by the sub-passes of a renderpass. Initial check is to make sure that READ_ONLY
// layout attachments don't have CLEAR as their loadOp.
bool ValidateLayoutVsAttachmentDescription(const debug_report_data *report_data, const VkImageLayout first_layout,
                                           const uint32_t attachment, const VkAttachmentDescription &attachment_description) {
    bool skip = false;
    // Verify that initial loadOp on READ_ONLY attachments is not CLEAR
    if (attachment_description.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        if ((first_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) ||
            (first_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_12200688, "DS", "Cannot clear attachment %d with invalid first layout %s. %s",
                            attachment, string_VkImageLayout(first_layout), validation_error_map[VALIDATION_ERROR_12200688]);
        }
    }
    return skip;
}

bool ValidateLayouts(core_validation::layer_data *device_data, VkDevice device, const VkRenderPassCreateInfo *pCreateInfo) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;

    // Track when we're observing the first use of an attachment
    std::vector<bool> attach_first_use(pCreateInfo->attachmentCount, true);
    for (uint32_t i = 0; i < pCreateInfo->subpassCount; ++i) {
        const VkSubpassDescription &subpass = pCreateInfo->pSubpasses[i];

        // Check input attachments first, so we can detect first-use-as-input for VU #00349
        for (uint32_t j = 0; j < subpass.inputAttachmentCount; ++j) {
            auto attach_index = subpass.pInputAttachments[j].attachment;
            if (attach_index == VK_ATTACHMENT_UNUSED) continue;

            switch (subpass.pInputAttachments[j].layout) {
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                    // These are ideal.
                    break;

                case VK_IMAGE_LAYOUT_GENERAL:
                    // May not be optimal. TODO: reconsider this warning based on other constraints.
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                                    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Layout for input attachment is GENERAL but should be READ_ONLY_OPTIMAL.");
                    break;

                default:
                    // No other layouts are acceptable
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Layout for input attachment is %s but can only be READ_ONLY_OPTIMAL or GENERAL.",
                                    string_VkImageLayout(subpass.pInputAttachments[j].layout));
            }

            VkImageLayout layout = subpass.pInputAttachments[j].layout;
            bool found_layout_mismatch = subpass.pDepthStencilAttachment &&
                                         subpass.pDepthStencilAttachment->attachment == attach_index &&
                                         subpass.pDepthStencilAttachment->layout != layout;
            for (uint32_t c = 0; !found_layout_mismatch && c < subpass.colorAttachmentCount; ++c) {
                found_layout_mismatch =
                    (subpass.pColorAttachments[c].attachment == attach_index && subpass.pColorAttachments[c].layout != layout);
            }
            if (found_layout_mismatch) {
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_140006ae, "DS",
                    "CreateRenderPass:  Subpass %u pInputAttachments[%u] (%u) has layout %u, but is also used as a depth/color "
                    "attachment with a different layout. %s",
                    i, j, attach_index, layout, validation_error_map[VALIDATION_ERROR_140006ae]);
            }

            if (attach_first_use[attach_index]) {
                skip |= ValidateLayoutVsAttachmentDescription(report_data, subpass.pInputAttachments[j].layout, attach_index,
                                                              pCreateInfo->pAttachments[attach_index]);

                bool used_as_depth =
                    (subpass.pDepthStencilAttachment != NULL && subpass.pDepthStencilAttachment->attachment == attach_index);
                bool used_as_color = false;
                for (uint32_t k = 0; !used_as_depth && !used_as_color && k < subpass.colorAttachmentCount; ++k) {
                    used_as_color = (subpass.pColorAttachments[k].attachment == attach_index);
                }
                if (!used_as_depth && !used_as_color &&
                    pCreateInfo->pAttachments[attach_index].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                    skip |= log_msg(
                        report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_1400069c, "DS",
                        "CreateRenderPass: attachment %u is first used as an input attachment in subpass %u with loadOp=CLEAR. %s",
                        attach_index, attach_index, validation_error_map[VALIDATION_ERROR_1400069c]);
                }
            }
            attach_first_use[attach_index] = false;
        }
        for (uint32_t j = 0; j < subpass.colorAttachmentCount; ++j) {
            auto attach_index = subpass.pColorAttachments[j].attachment;
            if (attach_index == VK_ATTACHMENT_UNUSED) continue;

            // TODO: Need a way to validate shared presentable images here, currently just allowing
            // VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR
            //  as an acceptable layout, but need to make sure shared presentable images ONLY use that layout
            switch (subpass.pColorAttachments[j].layout) {
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                // This is ideal.
                case VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR:
                    // TODO: See note above, just assuming that attachment is shared presentable and allowing this for now.
                    break;

                case VK_IMAGE_LAYOUT_GENERAL:
                    // May not be optimal; TODO: reconsider this warning based on other constraints?
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                                    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Layout for color attachment is GENERAL but should be COLOR_ATTACHMENT_OPTIMAL.");
                    break;

                default:
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Layout for color attachment is %s but can only be COLOR_ATTACHMENT_OPTIMAL or GENERAL.",
                                    string_VkImageLayout(subpass.pColorAttachments[j].layout));
            }

            if (attach_first_use[attach_index]) {
                skip |= ValidateLayoutVsAttachmentDescription(report_data, subpass.pColorAttachments[j].layout, attach_index,
                                                              pCreateInfo->pAttachments[attach_index]);
            }
            attach_first_use[attach_index] = false;
        }
        if (subpass.pDepthStencilAttachment && subpass.pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED) {
            switch (subpass.pDepthStencilAttachment->layout) {
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                    // These are ideal.
                    break;

                case VK_IMAGE_LAYOUT_GENERAL:
                    // May not be optimal; TODO: reconsider this warning based on other constraints? GENERAL can be better than
                    // doing a bunch of transitions.
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                                    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "GENERAL layout for depth attachment may not give optimal performance.");
                    break;

                default:
                    // No other layouts are acceptable
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                    "Layout for depth attachment is %s but can only be DEPTH_STENCIL_ATTACHMENT_OPTIMAL, "
                                    "DEPTH_STENCIL_READ_ONLY_OPTIMAL or GENERAL.",
                                    string_VkImageLayout(subpass.pDepthStencilAttachment->layout));
            }

            auto attach_index = subpass.pDepthStencilAttachment->attachment;
            if (attach_first_use[attach_index]) {
                skip |= ValidateLayoutVsAttachmentDescription(report_data, subpass.pDepthStencilAttachment->layout, attach_index,
                                                              pCreateInfo->pAttachments[attach_index]);
            }
            attach_first_use[attach_index] = false;
        }
    }
    return skip;
}

// For any image objects that overlap mapped memory, verify that their layouts are PREINIT or GENERAL
bool ValidateMapImageLayouts(core_validation::layer_data *device_data, VkDevice device, DEVICE_MEM_INFO const *mem_info,
                             VkDeviceSize offset, VkDeviceSize end_offset) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    // Iterate over all bound image ranges and verify that for any that overlap the map ranges, the layouts are
    // VK_IMAGE_LAYOUT_PREINITIALIZED or VK_IMAGE_LAYOUT_GENERAL
    // TODO : This can be optimized if we store ranges based on starting address and early exit when we pass our range
    for (auto image_handle : mem_info->bound_images) {
        auto img_it = mem_info->bound_ranges.find(image_handle);
        if (img_it != mem_info->bound_ranges.end()) {
            if (rangesIntersect(device_data, &img_it->second, offset, end_offset)) {
                std::vector<VkImageLayout> layouts;
                if (FindLayouts(device_data, VkImage(image_handle), layouts)) {
                    for (auto layout : layouts) {
                        if (layout != VK_IMAGE_LAYOUT_PREINITIALIZED && layout != VK_IMAGE_LAYOUT_GENERAL) {
                            skip |=
                                log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT,
                                        HandleToUint64(mem_info->mem), __LINE__, DRAWSTATE_INVALID_IMAGE_LAYOUT, "DS",
                                        "Mapping an image with layout %s can result in undefined behavior if this memory is "
                                        "used by the device. Only GENERAL or PREINITIALIZED should be used.",
                                        string_VkImageLayout(layout));
                        }
                    }
                }
            }
        }
    }
    return skip;
}

// Helper function to validate correct usage bits set for buffers or images. Verify that (actual & desired) flags != 0 or, if strict
// is true, verify that (actual & desired) flags == desired
static bool validate_usage_flags(layer_data *device_data, VkFlags actual, VkFlags desired, VkBool32 strict, uint64_t obj_handle,
                                 VulkanObjectType obj_type, int32_t const msgCode, char const *func_name, char const *usage_str) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    bool correct_usage = false;
    bool skip = false;
    const char *type_str = object_string[obj_type];
    if (strict) {
        correct_usage = ((actual & desired) == desired);
    } else {
        correct_usage = ((actual & desired) != 0);
    }
    if (!correct_usage) {
        if (msgCode == -1) {
            // TODO: Fix callers with msgCode == -1 to use correct validation checks.
            skip = log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, get_debug_report_enum[obj_type], obj_handle, __LINE__,
                           MEMTRACK_INVALID_USAGE_FLAG, "MEM",
                           "Invalid usage flag for %s 0x%" PRIxLEAST64
                           " used by %s. In this case, %s should have %s set during creation.",
                           type_str, obj_handle, func_name, type_str, usage_str);
        } else {
            const char *valid_usage = (msgCode == -1) ? "" : validation_error_map[msgCode];
            skip = log_msg(
                report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, get_debug_report_enum[obj_type], obj_handle, __LINE__, msgCode, "MEM",
                "Invalid usage flag for %s 0x%" PRIxLEAST64 " used by %s. In this case, %s should have %s set during creation. %s",
                type_str, obj_handle, func_name, type_str, usage_str, valid_usage);
        }
    }
    return skip;
}

// Helper function to validate usage flags for buffers. For given buffer_state send actual vs. desired usage off to helper above
// where an error will be flagged if usage is not correct
bool ValidateImageUsageFlags(layer_data *device_data, IMAGE_STATE const *image_state, VkFlags desired, bool strict,
                             int32_t const msgCode, char const *func_name, char const *usage_string) {
    return validate_usage_flags(device_data, image_state->createInfo.usage, desired, strict, HandleToUint64(image_state->image),
                                kVulkanObjectTypeImage, msgCode, func_name, usage_string);
}

// Helper function to validate usage flags for buffers. For given buffer_state send actual vs. desired usage off to helper above
// where an error will be flagged if usage is not correct
bool ValidateBufferUsageFlags(layer_data *device_data, BUFFER_STATE const *buffer_state, VkFlags desired, bool strict,
                              int32_t const msgCode, char const *func_name, char const *usage_string) {
    return validate_usage_flags(device_data, buffer_state->createInfo.usage, desired, strict, HandleToUint64(buffer_state->buffer),
                                kVulkanObjectTypeBuffer, msgCode, func_name, usage_string);
}

bool PreCallValidateCreateBuffer(layer_data *device_data, const VkBufferCreateInfo *pCreateInfo) {
    bool skip = false;
    const debug_report_data *report_data = core_validation::GetReportData(device_data);

    // TODO: Add check for VALIDATION_ERROR_1ec0071e        (sparse address space accounting)

    if ((pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) && (!GetEnabledFeatures(device_data)->sparseBinding)) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                        VALIDATION_ERROR_01400726, "DS",
                        "vkCreateBuffer(): the sparseBinding device feature is disabled: Buffers cannot be created with the "
                        "VK_BUFFER_CREATE_SPARSE_BINDING_BIT set. %s",
                        validation_error_map[VALIDATION_ERROR_01400726]);
    }

    if ((pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT) && (!GetEnabledFeatures(device_data)->sparseResidencyBuffer)) {
        skip |=
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_01400728, "DS",
                    "vkCreateBuffer(): the sparseResidencyBuffer device feature is disabled: Buffers cannot be created with the "
                    "VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT set. %s",
                    validation_error_map[VALIDATION_ERROR_01400728]);
    }

    if ((pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_ALIASED_BIT) && (!GetEnabledFeatures(device_data)->sparseResidencyAliased)) {
        skip |=
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_0140072a, "DS",
                    "vkCreateBuffer(): the sparseResidencyAliased device feature is disabled: Buffers cannot be created with the "
                    "VK_BUFFER_CREATE_SPARSE_ALIASED_BIT set. %s",
                    validation_error_map[VALIDATION_ERROR_0140072a]);
    }
    return skip;
}

void PostCallRecordCreateBuffer(layer_data *device_data, const VkBufferCreateInfo *pCreateInfo, VkBuffer *pBuffer) {
    // TODO : This doesn't create deep copy of pQueueFamilyIndices so need to fix that if/when we want that data to be valid
    GetBufferMap(device_data)
        ->insert(std::make_pair(*pBuffer, std::unique_ptr<BUFFER_STATE>(new BUFFER_STATE(*pBuffer, pCreateInfo))));
}

bool PreCallValidateCreateBufferView(layer_data *device_data, const VkBufferViewCreateInfo *pCreateInfo) {
    bool skip = false;
    BUFFER_STATE *buffer_state = GetBufferState(device_data, pCreateInfo->buffer);
    // If this isn't a sparse buffer, it needs to have memory backing it at CreateBufferView time
    if (buffer_state) {
        skip |= ValidateMemoryIsBoundToBuffer(device_data, buffer_state, "vkCreateBufferView()", VALIDATION_ERROR_01a0074e);
        // In order to create a valid buffer view, the buffer must have been created with at least one of the following flags:
        // UNIFORM_TEXEL_BUFFER_BIT or STORAGE_TEXEL_BUFFER_BIT
        skip |= ValidateBufferUsageFlags(
            device_data, buffer_state, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, false,
            VALIDATION_ERROR_01a00748, "vkCreateBufferView()", "VK_BUFFER_USAGE_[STORAGE|UNIFORM]_TEXEL_BUFFER_BIT");
    }
    return skip;
}

void PostCallRecordCreateBufferView(layer_data *device_data, const VkBufferViewCreateInfo *pCreateInfo, VkBufferView *pView) {
    (*GetBufferViewMap(device_data))[*pView] = std::unique_ptr<BUFFER_VIEW_STATE>(new BUFFER_VIEW_STATE(*pView, pCreateInfo));
}

// For the given format verify that the aspect masks make sense
bool ValidateImageAspectMask(layer_data *device_data, VkImage image, VkFormat format, VkImageAspectFlags aspect_mask,
                             const char *func_name) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    if (FormatIsColor(format)) {
        if ((aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != VK_IMAGE_ASPECT_COLOR_BIT) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Color image formats must have the VK_IMAGE_ASPECT_COLOR_BIT set. %s", func_name,
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        } else if ((aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != aspect_mask) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Color image formats must have ONLY the VK_IMAGE_ASPECT_COLOR_BIT set. %s", func_name,
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        }
    } else if (FormatIsDepthAndStencil(format)) {
        if ((aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) == 0) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Depth/stencil image formats must have "
                            "at least one of VK_IMAGE_ASPECT_DEPTH_BIT "
                            "and VK_IMAGE_ASPECT_STENCIL_BIT set. %s",
                            func_name, validation_error_map[VALIDATION_ERROR_0a400c01]);
        } else if ((aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != aspect_mask) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Combination depth/stencil image formats can have only the VK_IMAGE_ASPECT_DEPTH_BIT and "
                            "VK_IMAGE_ASPECT_STENCIL_BIT set. %s",
                            func_name, validation_error_map[VALIDATION_ERROR_0a400c01]);
        }
    } else if (FormatIsDepthOnly(format)) {
        if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != VK_IMAGE_ASPECT_DEPTH_BIT) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Depth-only image formats must have the VK_IMAGE_ASPECT_DEPTH_BIT set. %s", func_name,
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        } else if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != aspect_mask) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Depth-only image formats can have only the VK_IMAGE_ASPECT_DEPTH_BIT set. %s", func_name,
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        }
    } else if (FormatIsStencilOnly(format)) {
        if ((aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != VK_IMAGE_ASPECT_STENCIL_BIT) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Stencil-only image formats must have the VK_IMAGE_ASPECT_STENCIL_BIT set. %s", func_name,
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        } else if ((aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != aspect_mask) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "%s: Stencil-only image formats can have only the VK_IMAGE_ASPECT_STENCIL_BIT set. %s", func_name,
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        }
    }
    return skip;
}

bool ValidateImageSubresourceRange(const layer_data *device_data, const IMAGE_STATE *image_state, const bool is_imageview_2d_array,
                                   const VkImageSubresourceRange &subresourceRange, const char *cmd_name, const char *param_name) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;

    // Validate mip levels
    const auto image_mip_count = image_state->createInfo.mipLevels;

    if (subresourceRange.levelCount == 0) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_0a8007fc, "IMAGE",
                        "%s: %s.levelCount is 0. %s", cmd_name, param_name, validation_error_map[VALIDATION_ERROR_0a8007fc]);
    } else if (subresourceRange.levelCount == VK_REMAINING_MIP_LEVELS) {
        // TODO: Not in the spec VUs. Probably missing -- KhronosGroup/Vulkan-Docs#416
        if (subresourceRange.baseMipLevel >= image_mip_count) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, DRAWSTATE_INVALID_IMAGE_SUBRANGE, "IMAGE",
                            "%s: %s.baseMipLevel (= %" PRIu32
                            ") is greater or equal to the mip level count of the image (i.e. "
                            "greater or equal to %" PRIu32 ").",
                            cmd_name, param_name, subresourceRange.baseMipLevel, image_mip_count);
        }
    } else {
        const uint64_t necessary_mip_count = uint64_t{subresourceRange.baseMipLevel} + uint64_t{subresourceRange.levelCount};

        if (necessary_mip_count > image_mip_count) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_0a8007fc, "IMAGE",
                            "%s: %s.baseMipLevel + .levelCount (= %" PRIu32 " + %" PRIu32 " = %" PRIu64
                            ") is greater than the "
                            "mip level count of the image (i.e. greater than %" PRIu32 "). %s",
                            cmd_name, param_name, subresourceRange.baseMipLevel, subresourceRange.levelCount, necessary_mip_count,
                            image_mip_count, validation_error_map[VALIDATION_ERROR_0a8007fc]);
        }
    }

    // Validate array layers
    bool is_khr_maintenance1 = GetDeviceExtensions(device_data)->vk_khr_maintenance1;
    bool is_3D_to_2D_map = is_khr_maintenance1 && image_state->createInfo.imageType == VK_IMAGE_TYPE_3D && is_imageview_2d_array;

    const auto image_layer_count = is_3D_to_2D_map ? image_state->createInfo.extent.depth : image_state->createInfo.arrayLayers;
    const auto image_layer_count_var_name = is_3D_to_2D_map ? "extent.depth" : "arrayLayers";

    const auto invalid_layer_code =
        is_khr_maintenance1 ? (is_3D_to_2D_map ? VALIDATION_ERROR_0a800800 : VALIDATION_ERROR_0a800802) : VALIDATION_ERROR_0a8007fe;

    if (subresourceRange.layerCount == 0) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, invalid_layer_code, "IMAGE", "%s: %s.layerCount is 0. %s",
                        cmd_name, param_name, validation_error_map[invalid_layer_code]);
    } else if (subresourceRange.layerCount == VK_REMAINING_ARRAY_LAYERS) {
        // TODO: Not in the spec VUs. Probably missing -- KhronosGroup/Vulkan-Docs#416
        if (subresourceRange.baseArrayLayer >= image_layer_count) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, DRAWSTATE_INVALID_IMAGE_SUBRANGE, "IMAGE",
                            "%s: %s.baseArrayLayer (= %" PRIu32
                            ") is greater or equal to the %s of the image when it was created "
                            "(i.e. greater or equal to %" PRIu32 ").",
                            cmd_name, param_name, subresourceRange.baseArrayLayer, image_layer_count_var_name, image_layer_count);
        }
    } else {
        const uint64_t necessary_layer_count = uint64_t{subresourceRange.baseArrayLayer} + uint64_t{subresourceRange.layerCount};

        if (necessary_layer_count > image_layer_count) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                        HandleToUint64(image_state->image), __LINE__, invalid_layer_code, "IMAGE",
                        "%s: %s.baseArrayLayer + .layerCount (= %" PRIu32 " + %" PRIu32 " = %" PRIu64
                        ") is greater than the "
                        "%s of the image when it was created (i.e. greater than %" PRIu32 "). %s",
                        cmd_name, param_name, subresourceRange.baseArrayLayer, subresourceRange.layerCount, necessary_layer_count,
                        image_layer_count_var_name, image_layer_count, validation_error_map[invalid_layer_code]);
        }
    }

    return skip;
}

bool PreCallValidateCreateImageView(layer_data *device_data, const VkImageViewCreateInfo *create_info) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    IMAGE_STATE *image_state = GetImageState(device_data, create_info->image);
    if (image_state) {
        skip |= ValidateImageUsageFlags(
            device_data, image_state,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            false, -1, "vkCreateImageView()",
            "VK_IMAGE_USAGE_[SAMPLED|STORAGE|COLOR_ATTACHMENT|DEPTH_STENCIL_ATTACHMENT|INPUT_ATTACHMENT]_BIT");
        // If this isn't a sparse image, it needs to have memory backing it at CreateImageView time
        skip |= ValidateMemoryIsBoundToImage(device_data, image_state, "vkCreateImageView()", VALIDATION_ERROR_0ac007f8);
        // Checks imported from image layer
        skip |= ValidateImageSubresourceRange(device_data, image_state, create_info->viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            create_info->subresourceRange, "vkCreateImageView", "pCreateInfo->subresourceRange");

        VkImageCreateFlags image_flags = image_state->createInfo.flags;
        VkFormat image_format = image_state->createInfo.format;
        VkFormat view_format = create_info->format;
        VkImageAspectFlags aspect_mask = create_info->subresourceRange.aspectMask;
        VkImageType image_type = image_state->createInfo.imageType;
        VkImageViewType view_type = create_info->viewType;

        // Validate VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT state
        if (image_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
            // Format MUST be compatible (in the same format compatibility class) as the format the image was created with
            if (FormatCompatibilityClass(image_format) != FormatCompatibilityClass(view_format)) {
                std::stringstream ss;
                ss << "vkCreateImageView(): ImageView format " << string_VkFormat(view_format)
                   << " is not in the same format compatibility class as image (" << HandleToUint64(create_info->image)
                   << ")  format " << string_VkFormat(image_format)
                   << ".  Images created with the VK_IMAGE_CREATE_MUTABLE_FORMAT BIT "
                   << "can support ImageViews with differing formats but they must be in the same compatibility class.";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_0ac007f4, "IMAGE", "%s %s", ss.str().c_str(),
                                validation_error_map[VALIDATION_ERROR_0ac007f4]);
            }
        } else {
            // Format MUST be IDENTICAL to the format the image was created with
            if (image_format != view_format) {
                std::stringstream ss;
                ss << "vkCreateImageView() format " << string_VkFormat(view_format) << " differs from image "
                   << HandleToUint64(create_info->image) << " format " << string_VkFormat(image_format)
                   << ".  Formats MUST be IDENTICAL unless VK_IMAGE_CREATE_MUTABLE_FORMAT BIT was set on image creation.";
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                VALIDATION_ERROR_0ac007f6, "IMAGE", "%s %s", ss.str().c_str(),
                                validation_error_map[VALIDATION_ERROR_0ac007f6]);
            }
        }

        // Validate correct image aspect bits for desired formats and format consistency
        skip |= ValidateImageAspectMask(device_data, image_state->image, image_format, aspect_mask, "vkCreateImageView()");

        switch (image_type) {
            case VK_IMAGE_TYPE_1D:
                if (view_type != VK_IMAGE_VIEW_TYPE_1D && view_type != VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_0ac007fa, "IMAGE",
                                    "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s. %s",
                                    string_VkImageViewType(view_type), string_VkImageType(image_type),
                                    validation_error_map[VALIDATION_ERROR_0ac007fa]);
                }
                break;
            case VK_IMAGE_TYPE_2D:
                if (view_type != VK_IMAGE_VIEW_TYPE_2D && view_type != VK_IMAGE_VIEW_TYPE_2D_ARRAY) {
                    if ((view_type == VK_IMAGE_VIEW_TYPE_CUBE || view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) &&
                        !(image_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_0ac007d6, "IMAGE",
                                        "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s. %s",
                                        string_VkImageViewType(view_type), string_VkImageType(image_type),
                                        validation_error_map[VALIDATION_ERROR_0ac007d6]);
                    } else if (view_type != VK_IMAGE_VIEW_TYPE_CUBE && view_type != VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_0ac007fa, "IMAGE",
                                        "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s. %s",
                                        string_VkImageViewType(view_type), string_VkImageType(image_type),
                                        validation_error_map[VALIDATION_ERROR_0ac007fa]);
                    }
                }
                break;
            case VK_IMAGE_TYPE_3D:
                if (GetDeviceExtensions(device_data)->vk_khr_maintenance1) {
                    if (view_type != VK_IMAGE_VIEW_TYPE_3D) {
                        if ((view_type == VK_IMAGE_VIEW_TYPE_2D || view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY)) {
                            if (!(image_flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR)) {
                                skip |= log_msg(
                                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_0ac007da, "IMAGE",
                                    "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s. %s",
                                    string_VkImageViewType(view_type), string_VkImageType(image_type),
                                    validation_error_map[VALIDATION_ERROR_0ac007da]);
                            } else if ((image_flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                                                       VK_IMAGE_CREATE_SPARSE_ALIASED_BIT))) {
                                skip |= log_msg(
                                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, VALIDATION_ERROR_0ac007fa, "IMAGE",
                                    "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s when the "
                                    "VK_IMAGE_CREATE_SPARSE_BINDING_BIT, VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT, or "
                                    "VK_IMAGE_CREATE_SPARSE_ALIASED_BIT flags are enabled. %s",
                                    string_VkImageViewType(view_type), string_VkImageType(image_type),
                                    validation_error_map[VALIDATION_ERROR_0ac007fa]);
                            }
                        } else {
                            skip |=
                                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_0ac007fa, "IMAGE",
                                        "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s. %s",
                                        string_VkImageViewType(view_type), string_VkImageType(image_type),
                                        validation_error_map[VALIDATION_ERROR_0ac007fa]);
                        }
                    }
                } else {
                    if (view_type != VK_IMAGE_VIEW_TYPE_3D) {
                        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                        __LINE__, VALIDATION_ERROR_0ac007fa, "IMAGE",
                                        "vkCreateImageView(): pCreateInfo->viewType %s is not compatible with image type %s. %s",
                                        string_VkImageViewType(view_type), string_VkImageType(image_type),
                                        validation_error_map[VALIDATION_ERROR_0ac007fa]);
                    }
                }
                break;
            default:
                break;
        }
    }
    return skip;
}

void PostCallRecordCreateImageView(layer_data *device_data, const VkImageViewCreateInfo *create_info, VkImageView view) {
    auto image_view_map = GetImageViewMap(device_data);
    (*image_view_map)[view] = std::unique_ptr<IMAGE_VIEW_STATE>(new IMAGE_VIEW_STATE(view, create_info));

    auto image_state = GetImageState(device_data, create_info->image);
    auto &sub_res_range = (*image_view_map)[view].get()->create_info.subresourceRange;
    sub_res_range.levelCount = ResolveRemainingLevels(&sub_res_range, image_state->createInfo.mipLevels);
    sub_res_range.layerCount = ResolveRemainingLayers(&sub_res_range, image_state->createInfo.arrayLayers);
}

bool PreCallValidateCmdCopyBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *src_buffer_state,
                                  BUFFER_STATE *dst_buffer_state) {
    bool skip = false;
    skip |= ValidateMemoryIsBoundToBuffer(device_data, src_buffer_state, "vkCmdCopyBuffer()", VALIDATION_ERROR_18c000ee);
    skip |= ValidateMemoryIsBoundToBuffer(device_data, dst_buffer_state, "vkCmdCopyBuffer()", VALIDATION_ERROR_18c000f2);
    // Validate that SRC & DST buffers have correct usage flags set
    skip |= ValidateBufferUsageFlags(device_data, src_buffer_state, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true,
                                     VALIDATION_ERROR_18c000ec, "vkCmdCopyBuffer()", "VK_BUFFER_USAGE_TRANSFER_SRC_BIT");
    skip |= ValidateBufferUsageFlags(device_data, dst_buffer_state, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                                     VALIDATION_ERROR_18c000f0, "vkCmdCopyBuffer()", "VK_BUFFER_USAGE_TRANSFER_DST_BIT");
    skip |= ValidateCmdQueueFlags(device_data, cb_node, "vkCmdCopyBuffer()",
                                  VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, VALIDATION_ERROR_18c02415);
    skip |= ValidateCmd(device_data, cb_node, CMD_COPYBUFFER, "vkCmdCopyBuffer()");
    skip |= insideRenderPass(device_data, cb_node, "vkCmdCopyBuffer()", VALIDATION_ERROR_18c00017);
    return skip;
}

void PreCallRecordCmdCopyBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *src_buffer_state,
                                BUFFER_STATE *dst_buffer_state) {
    // Update bindings between buffers and cmd buffer
    AddCommandBufferBindingBuffer(device_data, cb_node, src_buffer_state);
    AddCommandBufferBindingBuffer(device_data, cb_node, dst_buffer_state);

    std::function<bool()> function = [=]() {
        return ValidateBufferMemoryIsValid(device_data, src_buffer_state, "vkCmdCopyBuffer()");
    };
    cb_node->queue_submit_functions.push_back(function);
    function = [=]() {
        SetBufferMemoryValid(device_data, dst_buffer_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
}

static bool validateIdleBuffer(layer_data *device_data, VkBuffer buffer) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    auto buffer_state = GetBufferState(device_data, buffer);
    if (!buffer_state) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, HandleToUint64(buffer),
                        __LINE__, DRAWSTATE_DOUBLE_DESTROY, "DS",
                        "Cannot free buffer 0x%" PRIxLEAST64 " that has not been allocated.", HandleToUint64(buffer));
    } else {
        if (buffer_state->in_use.load()) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,
                            HandleToUint64(buffer), __LINE__, VALIDATION_ERROR_23c00734, "DS",
                            "Cannot free buffer 0x%" PRIxLEAST64 " that is in use by a command buffer. %s", HandleToUint64(buffer),
                            validation_error_map[VALIDATION_ERROR_23c00734]);
        }
    }
    return skip;
}

bool PreCallValidateDestroyImageView(layer_data *device_data, VkImageView image_view, IMAGE_VIEW_STATE **image_view_state,
                                     VK_OBJECT *obj_struct) {
    *image_view_state = GetImageViewState(device_data, image_view);
    *obj_struct = {HandleToUint64(image_view), kVulkanObjectTypeImageView};
    if (GetDisables(device_data)->destroy_image_view) return false;
    bool skip = false;
    if (*image_view_state) {
        skip |= ValidateObjectNotInUse(device_data, *image_view_state, *obj_struct, VALIDATION_ERROR_25400804);
    }
    return skip;
}

void PostCallRecordDestroyImageView(layer_data *device_data, VkImageView image_view, IMAGE_VIEW_STATE *image_view_state,
                                    VK_OBJECT obj_struct) {
    // Any bound cmd buffers are now invalid
    invalidateCommandBuffers(device_data, image_view_state->cb_bindings, obj_struct);
    (*GetImageViewMap(device_data)).erase(image_view);
}

bool PreCallValidateDestroyBuffer(layer_data *device_data, VkBuffer buffer, BUFFER_STATE **buffer_state, VK_OBJECT *obj_struct) {
    *buffer_state = GetBufferState(device_data, buffer);
    *obj_struct = {HandleToUint64(buffer), kVulkanObjectTypeBuffer};
    if (GetDisables(device_data)->destroy_buffer) return false;
    bool skip = false;
    if (*buffer_state) {
        skip |= validateIdleBuffer(device_data, buffer);
    }
    return skip;
}

void PostCallRecordDestroyBuffer(layer_data *device_data, VkBuffer buffer, BUFFER_STATE *buffer_state, VK_OBJECT obj_struct) {
    invalidateCommandBuffers(device_data, buffer_state->cb_bindings, obj_struct);
    for (auto mem_binding : buffer_state->GetBoundMemory()) {
        auto mem_info = GetMemObjInfo(device_data, mem_binding);
        if (mem_info) {
            core_validation::RemoveBufferMemoryRange(HandleToUint64(buffer), mem_info);
        }
    }
    ClearMemoryObjectBindings(device_data, HandleToUint64(buffer), kVulkanObjectTypeBuffer);
    GetBufferMap(device_data)->erase(buffer_state->buffer);
}

bool PreCallValidateDestroyBufferView(layer_data *device_data, VkBufferView buffer_view, BUFFER_VIEW_STATE **buffer_view_state,
                                      VK_OBJECT *obj_struct) {
    *buffer_view_state = GetBufferViewState(device_data, buffer_view);
    *obj_struct = {HandleToUint64(buffer_view), kVulkanObjectTypeBufferView};
    if (GetDisables(device_data)->destroy_buffer_view) return false;
    bool skip = false;
    if (*buffer_view_state) {
        skip |= ValidateObjectNotInUse(device_data, *buffer_view_state, *obj_struct, VALIDATION_ERROR_23e00750);
    }
    return skip;
}

void PostCallRecordDestroyBufferView(layer_data *device_data, VkBufferView buffer_view, BUFFER_VIEW_STATE *buffer_view_state,
                                     VK_OBJECT obj_struct) {
    // Any bound cmd buffers are now invalid
    invalidateCommandBuffers(device_data, buffer_view_state->cb_bindings, obj_struct);
    GetBufferViewMap(device_data)->erase(buffer_view);
}

bool PreCallValidateCmdFillBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *buffer_state) {
    bool skip = false;
    skip |= ValidateMemoryIsBoundToBuffer(device_data, buffer_state, "vkCmdFillBuffer()", VALIDATION_ERROR_1b40003e);
    skip |= ValidateCmdQueueFlags(device_data, cb_node, "vkCmdFillBuffer()",
                                  VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, VALIDATION_ERROR_1b402415);
    skip |= ValidateCmd(device_data, cb_node, CMD_FILLBUFFER, "vkCmdFillBuffer()");
    // Validate that DST buffer has correct usage flags set
    skip |= ValidateBufferUsageFlags(device_data, buffer_state, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, VALIDATION_ERROR_1b40003a,
                                     "vkCmdFillBuffer()", "VK_BUFFER_USAGE_TRANSFER_DST_BIT");
    skip |= insideRenderPass(device_data, cb_node, "vkCmdFillBuffer()", VALIDATION_ERROR_1b400017);
    return skip;
}

void PreCallRecordCmdFillBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *buffer_state) {
    std::function<bool()> function = [=]() {
        SetBufferMemoryValid(device_data, buffer_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
    // Update bindings between buffer and cmd buffer
    AddCommandBufferBindingBuffer(device_data, cb_node, buffer_state);
}

bool ValidateBufferImageCopyData(const debug_report_data *report_data, uint32_t regionCount, const VkBufferImageCopy *pRegions,
                                 IMAGE_STATE *image_state, const char *function) {
    bool skip = false;

    for (uint32_t i = 0; i < regionCount; i++) {
        if (image_state->createInfo.imageType == VK_IMAGE_TYPE_1D) {
            if ((pRegions[i].imageOffset.y != 0) || (pRegions[i].imageExtent.height != 1)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_0160018e, "IMAGE",
                                "%s(): pRegion[%d] imageOffset.y is %d and imageExtent.height is %d. For 1D images these "
                                "must be 0 and 1, respectively. %s",
                                function, i, pRegions[i].imageOffset.y, pRegions[i].imageExtent.height,
                                validation_error_map[VALIDATION_ERROR_0160018e]);
            }
        }

        if ((image_state->createInfo.imageType == VK_IMAGE_TYPE_1D) || (image_state->createInfo.imageType == VK_IMAGE_TYPE_2D)) {
            if ((pRegions[i].imageOffset.z != 0) || (pRegions[i].imageExtent.depth != 1)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600192, "IMAGE",
                                "%s(): pRegion[%d] imageOffset.z is %d and imageExtent.depth is %d. For 1D and 2D images these "
                                "must be 0 and 1, respectively. %s",
                                function, i, pRegions[i].imageOffset.z, pRegions[i].imageExtent.depth,
                                validation_error_map[VALIDATION_ERROR_01600192]);
            }
        }

        if (image_state->createInfo.imageType == VK_IMAGE_TYPE_3D) {
            if ((0 != pRegions[i].imageSubresource.baseArrayLayer) || (1 != pRegions[i].imageSubresource.layerCount)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_016001aa, "IMAGE",
                                "%s(): pRegion[%d] imageSubresource.baseArrayLayer is %d and imageSubresource.layerCount is "
                                "%d. For 3D images these must be 0 and 1, respectively. %s",
                                function, i, pRegions[i].imageSubresource.baseArrayLayer, pRegions[i].imageSubresource.layerCount,
                                validation_error_map[VALIDATION_ERROR_016001aa]);
            }
        }

        // If the the calling command's VkImage parameter's format is not a depth/stencil format,
        // then bufferOffset must be a multiple of the calling command's VkImage parameter's texel size
        auto texel_size = FormatSize(image_state->createInfo.format);
        if (!FormatIsDepthAndStencil(image_state->createInfo.format) && SafeModulo(pRegions[i].bufferOffset, texel_size) != 0) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600182, "IMAGE",
                            "%s(): pRegion[%d] bufferOffset 0x%" PRIxLEAST64
                            " must be a multiple of this format's texel size (" PRINTF_SIZE_T_SPECIFIER "). %s",
                            function, i, pRegions[i].bufferOffset, texel_size, validation_error_map[VALIDATION_ERROR_01600182]);
        }

        //  BufferOffset must be a multiple of 4
        if (SafeModulo(pRegions[i].bufferOffset, 4) != 0) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600184, "IMAGE",
                            "%s(): pRegion[%d] bufferOffset 0x%" PRIxLEAST64 " must be a multiple of 4. %s", function, i,
                            pRegions[i].bufferOffset, validation_error_map[VALIDATION_ERROR_01600184]);
        }

        //  BufferRowLength must be 0, or greater than or equal to the width member of imageExtent
        if ((pRegions[i].bufferRowLength != 0) && (pRegions[i].bufferRowLength < pRegions[i].imageExtent.width)) {
            skip |= log_msg(
                report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600186, "IMAGE",
                "%s(): pRegion[%d] bufferRowLength (%d) must be zero or greater-than-or-equal-to imageExtent.width (%d). %s",
                function, i, pRegions[i].bufferRowLength, pRegions[i].imageExtent.width,
                validation_error_map[VALIDATION_ERROR_01600186]);
        }

        //  BufferImageHeight must be 0, or greater than or equal to the height member of imageExtent
        if ((pRegions[i].bufferImageHeight != 0) && (pRegions[i].bufferImageHeight < pRegions[i].imageExtent.height)) {
            skip |= log_msg(
                report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600188, "IMAGE",
                "%s(): pRegion[%d] bufferImageHeight (%d) must be zero or greater-than-or-equal-to imageExtent.height (%d). %s",
                function, i, pRegions[i].bufferImageHeight, pRegions[i].imageExtent.height,
                validation_error_map[VALIDATION_ERROR_01600188]);
        }

        // subresource aspectMask must have exactly 1 bit set
        const int num_bits = sizeof(VkFlags) * CHAR_BIT;
        std::bitset<num_bits> aspect_mask_bits(pRegions[i].imageSubresource.aspectMask);
        if (aspect_mask_bits.count() != 1) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_016001a8, "IMAGE",
                            "%s: aspectMasks for imageSubresource in each region must have only a single bit set. %s", function,
                            validation_error_map[VALIDATION_ERROR_016001a8]);
        }

        // image subresource aspect bit must match format
        if (!VerifyAspectsPresent(pRegions[i].imageSubresource.aspectMask, image_state->createInfo.format)) {
            skip |= log_msg(
                report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_016001a6, "IMAGE",
                "%s(): pRegion[%d] subresource aspectMask 0x%x specifies aspects that are not present in image format 0x%x. %s",
                function, i, pRegions[i].imageSubresource.aspectMask, image_state->createInfo.format,
                validation_error_map[VALIDATION_ERROR_016001a6]);
        }

        // Checks that apply only to compressed images
        // TODO: there is a comment in ValidateCopyBufferImageTransferGranularityRequirements() in core_validation.cpp that
        //       reserves a place for these compressed image checks.  This block of code could move there once the image
        //       stuff is moved into core validation.
        if (FormatIsCompressed(image_state->createInfo.format)) {
            auto block_size = FormatCompressedTexelBlockExtent(image_state->createInfo.format);

            //  BufferRowLength must be a multiple of block width
            if (SafeModulo(pRegions[i].bufferRowLength, block_size.width) != 0) {
                skip |= log_msg(
                    report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                    HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600196, "IMAGE",
                    "%s(): pRegion[%d] bufferRowLength (%d) must be a multiple of the compressed image's texel width (%d). %s.",
                    function, i, pRegions[i].bufferRowLength, block_size.width, validation_error_map[VALIDATION_ERROR_01600196]);
            }

            //  BufferRowHeight must be a multiple of block height
            if (SafeModulo(pRegions[i].bufferImageHeight, block_size.height) != 0) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_01600198, "IMAGE",
                                "%s(): pRegion[%d] bufferImageHeight (%d) must be a multiple of the compressed image's texel "
                                "height (%d). %s.",
                                function, i, pRegions[i].bufferImageHeight, block_size.height,
                                validation_error_map[VALIDATION_ERROR_01600198]);
            }

            //  image offsets must be multiples of block dimensions
            if ((SafeModulo(pRegions[i].imageOffset.x, block_size.width) != 0) ||
                (SafeModulo(pRegions[i].imageOffset.y, block_size.height) != 0) ||
                (SafeModulo(pRegions[i].imageOffset.z, block_size.depth) != 0)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_0160019a, "IMAGE",
                                "%s(): pRegion[%d] imageOffset(x,y) (%d, %d) must be multiples of the compressed image's texel "
                                "width & height (%d, %d). %s.",
                                function, i, pRegions[i].imageOffset.x, pRegions[i].imageOffset.y, block_size.width,
                                block_size.height, validation_error_map[VALIDATION_ERROR_0160019a]);
            }

            // bufferOffset must be a multiple of block size (linear bytes)
            size_t block_size_in_bytes = FormatSize(image_state->createInfo.format);
            if (SafeModulo(pRegions[i].bufferOffset, block_size_in_bytes) != 0) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_0160019c, "IMAGE",
                                "%s(): pRegion[%d] bufferOffset (0x%" PRIxLEAST64
                                ") must be a multiple of the compressed image's texel block "
                                "size (" PRINTF_SIZE_T_SPECIFIER "). %s.",
                                function, i, pRegions[i].bufferOffset, block_size_in_bytes,
                                validation_error_map[VALIDATION_ERROR_0160019c]);
            }

            // imageExtent width must be a multiple of block width, or extent+offset width must equal subresource width
            VkExtent3D mip_extent = GetImageSubresourceExtent(image_state, &(pRegions[i].imageSubresource));
            if ((SafeModulo(pRegions[i].imageExtent.width, block_size.width) != 0) &&
                (pRegions[i].imageExtent.width + pRegions[i].imageOffset.x != mip_extent.width)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_0160019e, "IMAGE",
                                "%s(): pRegion[%d] extent width (%d) must be a multiple of the compressed texture block width "
                                "(%d), or when added to offset.x (%d) must equal the image subresource width (%d). %s.",
                                function, i, pRegions[i].imageExtent.width, block_size.width, pRegions[i].imageOffset.x,
                                mip_extent.width, validation_error_map[VALIDATION_ERROR_0160019e]);
            }

            // imageExtent height must be a multiple of block height, or extent+offset height must equal subresource height
            if ((SafeModulo(pRegions[i].imageExtent.height, block_size.height) != 0) &&
                (pRegions[i].imageExtent.height + pRegions[i].imageOffset.y != mip_extent.height)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_016001a0, "IMAGE",
                                "%s(): pRegion[%d] extent height (%d) must be a multiple of the compressed texture block height "
                                "(%d), or when added to offset.y (%d) must equal the image subresource height (%d). %s.",
                                function, i, pRegions[i].imageExtent.height, block_size.height, pRegions[i].imageOffset.y,
                                mip_extent.height, validation_error_map[VALIDATION_ERROR_016001a0]);
            }

            // imageExtent depth must be a multiple of block depth, or extent+offset depth must equal subresource depth
            if ((SafeModulo(pRegions[i].imageExtent.depth, block_size.depth) != 0) &&
                (pRegions[i].imageExtent.depth + pRegions[i].imageOffset.z != mip_extent.depth)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                                HandleToUint64(image_state->image), __LINE__, VALIDATION_ERROR_016001a2, "IMAGE",
                                "%s(): pRegion[%d] extent width (%d) must be a multiple of the compressed texture block depth "
                                "(%d), or when added to offset.z (%d) must equal the image subresource depth (%d). %s.",
                                function, i, pRegions[i].imageExtent.depth, block_size.depth, pRegions[i].imageOffset.z,
                                mip_extent.depth, validation_error_map[VALIDATION_ERROR_016001a2]);
            }
        }
    }

    return skip;
}

static bool ValidateImageBounds(const debug_report_data *report_data, const IMAGE_STATE *image_state, const uint32_t regionCount,
                                const VkBufferImageCopy *pRegions, const char *func_name, UNIQUE_VALIDATION_ERROR_CODE msg_code) {
    bool skip = false;
    const VkImageCreateInfo *image_info = &(image_state->createInfo);

    for (uint32_t i = 0; i < regionCount; i++) {
        VkExtent3D extent = pRegions[i].imageExtent;
        VkOffset3D offset = pRegions[i].imageOffset;

        if (IsExtentSizeZero(&extent))  // Warn on zero area subresource
        {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                            (uint64_t)0, __LINE__, IMAGE_ZERO_AREA_SUBREGION, "IMAGE",
                            "%s: pRegion[%d] imageExtent of {%1d, %1d, %1d} has zero area", func_name, i, extent.width,
                            extent.height, extent.depth);
        }

        VkExtent3D image_extent = GetImageSubresourceExtent(image_state, &(pRegions[i].imageSubresource));

        // If we're using a compressed format, valid extent is rounded up to multiple of block size (per 18.1)
        if (FormatIsCompressed(image_info->format)) {
            auto block_extent = FormatCompressedTexelBlockExtent(image_info->format);
            if (image_extent.width % block_extent.width) {
                image_extent.width += (block_extent.width - (image_extent.width % block_extent.width));
            }
            if (image_extent.height % block_extent.height) {
                image_extent.height += (block_extent.height - (image_extent.height % block_extent.height));
            }
            if (image_extent.depth % block_extent.depth) {
                image_extent.depth += (block_extent.depth - (image_extent.depth % block_extent.depth));
            }
        }

        if (0 != ExceedsBounds(&offset, &extent, &image_extent)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, (uint64_t)0,
                            __LINE__, msg_code, "IMAGE", "%s: pRegion[%d] exceeds image bounds. %s.", func_name, i,
                            validation_error_map[msg_code]);
        }
    }

    return skip;
}

static inline bool ValidateBufferBounds(const debug_report_data *report_data, IMAGE_STATE *image_state, BUFFER_STATE *buff_state,
                                        uint32_t regionCount, const VkBufferImageCopy *pRegions, const char *func_name,
                                        UNIQUE_VALIDATION_ERROR_CODE msg_code) {
    bool skip = false;

    VkDeviceSize buffer_size = buff_state->createInfo.size;

    for (uint32_t i = 0; i < regionCount; i++) {
        VkExtent3D copy_extent = pRegions[i].imageExtent;

        VkDeviceSize buffer_width = (0 == pRegions[i].bufferRowLength ? copy_extent.width : pRegions[i].bufferRowLength);
        VkDeviceSize buffer_height = (0 == pRegions[i].bufferImageHeight ? copy_extent.height : pRegions[i].bufferImageHeight);
        VkDeviceSize unit_size = FormatSize(image_state->createInfo.format);  // size (bytes) of texel or block

        // Handle special buffer packing rules for specific depth/stencil formats
        if (pRegions[i].imageSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) {
            unit_size = FormatSize(VK_FORMAT_S8_UINT);
        } else if (pRegions[i].imageSubresource.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
            switch (image_state->createInfo.format) {
                case VK_FORMAT_D16_UNORM_S8_UINT:
                    unit_size = FormatSize(VK_FORMAT_D16_UNORM);
                    break;
                case VK_FORMAT_D32_SFLOAT_S8_UINT:
                    unit_size = FormatSize(VK_FORMAT_D32_SFLOAT);
                    break;
                case VK_FORMAT_X8_D24_UNORM_PACK32:  // Fall through
                case VK_FORMAT_D24_UNORM_S8_UINT:
                    unit_size = 4;
                    break;
                default:
                    break;
            }
        }

        if (FormatIsCompressed(image_state->createInfo.format)) {
            // Switch to texel block units, rounding up for any partially-used blocks
            auto block_dim = FormatCompressedTexelBlockExtent(image_state->createInfo.format);
            buffer_width = (buffer_width + block_dim.width - 1) / block_dim.width;
            buffer_height = (buffer_height + block_dim.height - 1) / block_dim.height;

            copy_extent.width = (copy_extent.width + block_dim.width - 1) / block_dim.width;
            copy_extent.height = (copy_extent.height + block_dim.height - 1) / block_dim.height;
            copy_extent.depth = (copy_extent.depth + block_dim.depth - 1) / block_dim.depth;
        }

        // Either depth or layerCount may be greater than 1 (not both). This is the number of 'slices' to copy
        uint32_t z_copies = std::max(copy_extent.depth, pRegions[i].imageSubresource.layerCount);
        if (IsExtentSizeZero(&copy_extent) || (0 == z_copies)) {
            // TODO: Issue warning here? Already warned in ValidateImageBounds()...
        } else {
            // Calculate buffer offset of final copied byte, + 1.
            VkDeviceSize max_buffer_offset = (z_copies - 1) * buffer_height * buffer_width;      // offset to slice
            max_buffer_offset += ((copy_extent.height - 1) * buffer_width) + copy_extent.width;  // add row,col
            max_buffer_offset *= unit_size;                                                      // convert to bytes
            max_buffer_offset += pRegions[i].bufferOffset;                                       // add initial offset (bytes)

            if (buffer_size < max_buffer_offset) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, (uint64_t)0,
                            __LINE__, msg_code, "IMAGE", "%s: pRegion[%d] exceeds buffer size of %" PRIu64 " bytes. %s.", func_name,
                            i, buffer_size, validation_error_map[msg_code]);
            }
        }
    }

    return skip;
}

bool PreCallValidateCmdCopyImageToBuffer(layer_data *device_data, VkImageLayout srcImageLayout, GLOBAL_CB_NODE *cb_node,
                                         IMAGE_STATE *src_image_state, BUFFER_STATE *dst_buffer_state, uint32_t regionCount,
                                         const VkBufferImageCopy *pRegions, const char *func_name) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = ValidateBufferImageCopyData(report_data, regionCount, pRegions, src_image_state, "vkCmdCopyImageToBuffer");

    // Validate command buffer state
    if (CB_RECORDING != cb_node->state) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_19202413, "DS",
                        "Cannot call vkCmdCopyImageToBuffer() on command buffer which is not in recording state. %s.",
                        validation_error_map[VALIDATION_ERROR_19202413]);
    } else {
        skip |= ValidateCmdSubpassState(device_data, cb_node, CMD_COPYIMAGETOBUFFER);
    }

    // Command pool must support graphics, compute, or transfer operations
    auto pPool = GetCommandPoolNode(device_data, cb_node->createInfo.commandPool);

    VkQueueFlags queue_flags = GetPhysDevProperties(device_data)->queue_family_properties[pPool->queueFamilyIndex].queueFlags;
    if (0 == (queue_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->createInfo.commandPool), __LINE__, VALIDATION_ERROR_19202415, "DS",
                        "Cannot call vkCmdCopyImageToBuffer() on a command buffer allocated from a pool without graphics, compute, "
                        "or transfer capabilities. %s.",
                        validation_error_map[VALIDATION_ERROR_19202415]);
    }
    skip |= ValidateImageBounds(report_data, src_image_state, regionCount, pRegions, "vkCmdCopyBufferToImage()",
                                VALIDATION_ERROR_1920016c);
    skip |= ValidateBufferBounds(report_data, src_image_state, dst_buffer_state, regionCount, pRegions, "vkCmdCopyImageToBuffer()",
                                 VALIDATION_ERROR_1920016e);

    skip |= ValidateImageSampleCount(device_data, src_image_state, VK_SAMPLE_COUNT_1_BIT, "vkCmdCopyImageToBuffer(): srcImage",
                                     VALIDATION_ERROR_19200178);
    skip |= ValidateMemoryIsBoundToImage(device_data, src_image_state, "vkCmdCopyImageToBuffer()", VALIDATION_ERROR_19200176);
    skip |= ValidateMemoryIsBoundToBuffer(device_data, dst_buffer_state, "vkCmdCopyImageToBuffer()", VALIDATION_ERROR_19200180);

    // Validate that SRC image & DST buffer have correct usage flags set
    skip |= ValidateImageUsageFlags(device_data, src_image_state, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true, VALIDATION_ERROR_19200174,
                                    "vkCmdCopyImageToBuffer()", "VK_IMAGE_USAGE_TRANSFER_SRC_BIT");
    skip |= ValidateBufferUsageFlags(device_data, dst_buffer_state, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                                     VALIDATION_ERROR_1920017e, "vkCmdCopyImageToBuffer()", "VK_BUFFER_USAGE_TRANSFER_DST_BIT");
    skip |= insideRenderPass(device_data, cb_node, "vkCmdCopyImageToBuffer()", VALIDATION_ERROR_19200017);
    bool hit_error = false;
    for (uint32_t i = 0; i < regionCount; ++i) {
        skip |= VerifyImageLayout(device_data, cb_node, src_image_state, pRegions[i].imageSubresource, srcImageLayout,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, "vkCmdCopyImageToBuffer()", VALIDATION_ERROR_1920017c,
                                  &hit_error);
        skip |= ValidateCopyBufferImageTransferGranularityRequirements(device_data, cb_node, src_image_state, &pRegions[i], i,
                                                                       "vkCmdCopyImageToBuffer()");
    }
    return skip;
}

void PreCallRecordCmdCopyImageToBuffer(layer_data *device_data, GLOBAL_CB_NODE *cb_node, IMAGE_STATE *src_image_state,
                                       BUFFER_STATE *dst_buffer_state, uint32_t region_count, const VkBufferImageCopy *regions,
                                       VkImageLayout src_image_layout) {
    // Make sure that all image slices are updated to correct layout
    for (uint32_t i = 0; i < region_count; ++i) {
        SetImageLayout(device_data, cb_node, src_image_state, regions[i].imageSubresource, src_image_layout);
    }
    // Update bindings between buffer/image and cmd buffer
    AddCommandBufferBindingImage(device_data, cb_node, src_image_state);
    AddCommandBufferBindingBuffer(device_data, cb_node, dst_buffer_state);

    std::function<bool()> function = [=]() {
        return ValidateImageMemoryIsValid(device_data, src_image_state, "vkCmdCopyImageToBuffer()");
    };
    cb_node->queue_submit_functions.push_back(function);
    function = [=]() {
        SetBufferMemoryValid(device_data, dst_buffer_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
}

bool PreCallValidateCmdCopyBufferToImage(layer_data *device_data, VkImageLayout dstImageLayout, GLOBAL_CB_NODE *cb_node,
                                         BUFFER_STATE *src_buffer_state, IMAGE_STATE *dst_image_state, uint32_t regionCount,
                                         const VkBufferImageCopy *pRegions, const char *func_name) {
    const debug_report_data *report_data = core_validation::GetReportData(device_data);
    bool skip = ValidateBufferImageCopyData(report_data, regionCount, pRegions, dst_image_state, "vkCmdCopyBufferToImage");

    // Validate command buffer state
    if (CB_RECORDING != cb_node->state) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->commandBuffer), __LINE__, VALIDATION_ERROR_18e02413, "DS",
                        "Cannot call vkCmdCopyBufferToImage() on command buffer which is not in recording state. %s.",
                        validation_error_map[VALIDATION_ERROR_18e02413]);
    } else {
        skip |= ValidateCmdSubpassState(device_data, cb_node, CMD_COPYBUFFERTOIMAGE);
    }

    // Command pool must support graphics, compute, or transfer operations
    auto pPool = GetCommandPoolNode(device_data, cb_node->createInfo.commandPool);
    VkQueueFlags queue_flags = GetPhysDevProperties(device_data)->queue_family_properties[pPool->queueFamilyIndex].queueFlags;
    if (0 == (queue_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                        HandleToUint64(cb_node->createInfo.commandPool), __LINE__, VALIDATION_ERROR_18e02415, "DS",
                        "Cannot call vkCmdCopyBufferToImage() on a command buffer allocated from a pool without graphics, compute, "
                        "or transfer capabilities. %s.",
                        validation_error_map[VALIDATION_ERROR_18e02415]);
    }
    skip |= ValidateImageBounds(report_data, dst_image_state, regionCount, pRegions, "vkCmdCopyBufferToImage()",
                                VALIDATION_ERROR_18e00158);
    skip |= ValidateBufferBounds(report_data, dst_image_state, src_buffer_state, regionCount, pRegions, "vkCmdCopyBufferToImage()",
                                 VALIDATION_ERROR_18e00156);
    skip |= ValidateImageSampleCount(device_data, dst_image_state, VK_SAMPLE_COUNT_1_BIT, "vkCmdCopyBufferToImage(): dstImage",
                                     VALIDATION_ERROR_18e00166);
    skip |= ValidateMemoryIsBoundToBuffer(device_data, src_buffer_state, "vkCmdCopyBufferToImage()", VALIDATION_ERROR_18e00160);
    skip |= ValidateMemoryIsBoundToImage(device_data, dst_image_state, "vkCmdCopyBufferToImage()", VALIDATION_ERROR_18e00164);
    skip |= ValidateBufferUsageFlags(device_data, src_buffer_state, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true,
                                     VALIDATION_ERROR_18e0015c, "vkCmdCopyBufferToImage()", "VK_BUFFER_USAGE_TRANSFER_SRC_BIT");
    skip |= ValidateImageUsageFlags(device_data, dst_image_state, VK_IMAGE_USAGE_TRANSFER_DST_BIT, true, VALIDATION_ERROR_18e00162,
                                    "vkCmdCopyBufferToImage()", "VK_IMAGE_USAGE_TRANSFER_DST_BIT");
    skip |= insideRenderPass(device_data, cb_node, "vkCmdCopyBufferToImage()", VALIDATION_ERROR_18e00017);
    bool hit_error = false;
    for (uint32_t i = 0; i < regionCount; ++i) {
        skip |= VerifyImageLayout(device_data, cb_node, dst_image_state, pRegions[i].imageSubresource, dstImageLayout,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, "vkCmdCopyBufferToImage()", VALIDATION_ERROR_18e0016a,
                                  &hit_error);
        skip |= ValidateCopyBufferImageTransferGranularityRequirements(device_data, cb_node, dst_image_state, &pRegions[i], i,
                                                                       "vkCmdCopyBufferToImage()");
    }
    return skip;
}

void PreCallRecordCmdCopyBufferToImage(layer_data *device_data, GLOBAL_CB_NODE *cb_node, BUFFER_STATE *src_buffer_state,
                                       IMAGE_STATE *dst_image_state, uint32_t region_count, const VkBufferImageCopy *regions,
                                       VkImageLayout dst_image_layout) {
    // Make sure that all image slices are updated to correct layout
    for (uint32_t i = 0; i < region_count; ++i) {
        SetImageLayout(device_data, cb_node, dst_image_state, regions[i].imageSubresource, dst_image_layout);
    }
    AddCommandBufferBindingBuffer(device_data, cb_node, src_buffer_state);
    AddCommandBufferBindingImage(device_data, cb_node, dst_image_state);
    std::function<bool()> function = [=]() {
        SetImageMemoryValid(device_data, dst_image_state, true);
        return false;
    };
    cb_node->queue_submit_functions.push_back(function);
    function = [=]() { return ValidateBufferMemoryIsValid(device_data, src_buffer_state, "vkCmdCopyBufferToImage()"); };
    cb_node->queue_submit_functions.push_back(function);
}

bool PreCallValidateGetImageSubresourceLayout(layer_data *device_data, VkImage image, const VkImageSubresource *pSubresource) {
    const auto report_data = core_validation::GetReportData(device_data);
    bool skip = false;
    const VkImageAspectFlags sub_aspect = pSubresource->aspectMask;

    // VU 00733: The aspectMask member of pSubresource must only have a single bit set
    const int num_bits = sizeof(sub_aspect) * CHAR_BIT;
    std::bitset<num_bits> aspect_mask_bits(sub_aspect);
    if (aspect_mask_bits.count() != 1) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(image),
                        __LINE__, VALIDATION_ERROR_2a6007ca, "IMAGE",
                        "vkGetImageSubresourceLayout(): VkImageSubresource.aspectMask must have exactly 1 bit set. %s",
                        validation_error_map[VALIDATION_ERROR_2a6007ca]);
    }

    IMAGE_STATE *image_entry = GetImageState(device_data, image);
    if (!image_entry) {
        return skip;
    }

    // VU 00732: image must have been created with tiling equal to VK_IMAGE_TILING_LINEAR
    if (image_entry->createInfo.tiling != VK_IMAGE_TILING_LINEAR) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(image),
                        __LINE__, VALIDATION_ERROR_2a6007c8, "IMAGE",
                        "vkGetImageSubresourceLayout(): Image must have tiling of VK_IMAGE_TILING_LINEAR. %s",
                        validation_error_map[VALIDATION_ERROR_2a6007c8]);
    }

    // VU 00739: mipLevel must be less than the mipLevels specified in VkImageCreateInfo when the image was created
    if (pSubresource->mipLevel >= image_entry->createInfo.mipLevels) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(image),
                        __LINE__, VALIDATION_ERROR_0a4007cc, "IMAGE",
                        "vkGetImageSubresourceLayout(): pSubresource.mipLevel (%d) must be less than %d. %s",
                        pSubresource->mipLevel, image_entry->createInfo.mipLevels, validation_error_map[VALIDATION_ERROR_0a4007cc]);
    }

    // VU 00740: arrayLayer must be less than the arrayLayers specified in VkImageCreateInfo when the image was created
    if (pSubresource->arrayLayer >= image_entry->createInfo.arrayLayers) {
        skip |=
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(image),
                    __LINE__, VALIDATION_ERROR_0a4007ce, "IMAGE",
                    "vkGetImageSubresourceLayout(): pSubresource.arrayLayer (%d) must be less than %d. %s",
                    pSubresource->arrayLayer, image_entry->createInfo.arrayLayers, validation_error_map[VALIDATION_ERROR_0a4007ce]);
    }

    // VU 00741: subresource's aspect must be compatible with image's format.
    const VkFormat img_format = image_entry->createInfo.format;
    if (FormatIsColor(img_format)) {
        if (sub_aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
            skip |= log_msg(
                report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, HandleToUint64(image), __LINE__,
                VALIDATION_ERROR_0a400c01, "IMAGE",
                "vkGetImageSubresourceLayout(): For color formats, VkImageSubresource.aspectMask must be VK_IMAGE_ASPECT_COLOR. %s",
                validation_error_map[VALIDATION_ERROR_0a400c01]);
        }
    } else if (FormatIsDepthOrStencil(img_format)) {
        if ((sub_aspect != VK_IMAGE_ASPECT_DEPTH_BIT) && (sub_aspect != VK_IMAGE_ASPECT_STENCIL_BIT)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
                            HandleToUint64(image), __LINE__, VALIDATION_ERROR_0a400c01, "IMAGE",
                            "vkGetImageSubresourceLayout(): For depth/stencil formats, VkImageSubresource.aspectMask must be "
                            "either VK_IMAGE_ASPECT_DEPTH_BIT or VK_IMAGE_ASPECT_STENCIL_BIT. %s",
                            validation_error_map[VALIDATION_ERROR_0a400c01]);
        }
    }
    return skip;
}

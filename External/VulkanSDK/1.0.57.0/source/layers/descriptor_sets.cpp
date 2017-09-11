/* Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015-2016 Google Inc.
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
 * Author: Tobin Ehlis <tobine@google.com>
 */

// Allow use of STL min and max functions in Windows
#define NOMINMAX

#include "descriptor_sets.h"
#include "vk_enum_string_helper.h"
#include "vk_safe_struct.h"
#include "buffer_validation.h"
#include <sstream>
#include <algorithm>

// Construct DescriptorSetLayout instance from given create info
cvdescriptorset::DescriptorSetLayout::DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo *p_create_info,
                                                          const VkDescriptorSetLayout layout)
    : layout_(layout), binding_count_(p_create_info->bindingCount), descriptor_count_(0), dynamic_descriptor_count_(0) {
    // Dyn array indicies are ordered by binding # and array index of any array within the binding
    //  so we store up bindings w/ count in ordered map in order to create dyn array mappings below
    std::map<uint32_t, uint32_t> binding_to_dyn_count;
    for (uint32_t i = 0; i < binding_count_; ++i) {
        auto binding_num = p_create_info->pBindings[i].binding;
        descriptor_count_ += p_create_info->pBindings[i].descriptorCount;
        uint32_t insert_index = 0;  // Track vector index where we insert element
        if (bindings_.empty() || binding_num > bindings_.back().binding) {
            bindings_.push_back(safe_VkDescriptorSetLayoutBinding(&p_create_info->pBindings[i]));
            insert_index = static_cast<uint32_t>(bindings_.size()) - 1;
        } else {  // out-of-order binding number, need to insert into vector in-order
            auto it = bindings_.begin();
            // Find currently binding's spot in vector
            while (binding_num > it->binding) {
                assert(it != bindings_.end());
                ++insert_index;
                ++it;
            }
            bindings_.insert(it, safe_VkDescriptorSetLayoutBinding(&p_create_info->pBindings[i]));
        }
        // In cases where we should ignore pImmutableSamplers make sure it's NULL
        if ((p_create_info->pBindings[i].pImmutableSamplers) &&
            ((p_create_info->pBindings[i].descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) &&
             (p_create_info->pBindings[i].descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))) {
            bindings_[insert_index].pImmutableSamplers = nullptr;
        }
        if (p_create_info->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
            p_create_info->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
            binding_to_dyn_count[p_create_info->pBindings[i].binding] = p_create_info->pBindings[i].descriptorCount;
            dynamic_descriptor_count_ += p_create_info->pBindings[i].descriptorCount;
        }
    }
    assert(bindings_.size() == binding_count_);
    uint32_t global_index = 0;
    // Vector order is finalized so create maps of bindings to indices
    for (uint32_t i = 0; i < binding_count_; ++i) {
        auto binding_num = bindings_[i].binding;
        binding_to_index_map_[binding_num] = i;
        binding_to_global_start_index_map_[binding_num] = global_index;
        global_index += bindings_[i].descriptorCount ? bindings_[i].descriptorCount - 1 : 0;
        binding_to_global_end_index_map_[binding_num] = global_index;
        global_index += bindings_[i].descriptorCount ? 1 : 0;
    }
    // Now create dyn offset array mapping for any dynamic descriptors
    uint32_t dyn_array_idx = 0;
    for (const auto &bc_pair : binding_to_dyn_count) {
        binding_to_dynamic_array_idx_map_[bc_pair.first] = dyn_array_idx;
        dyn_array_idx += bc_pair.second;
    }
}

// Validate descriptor set layout create info
bool cvdescriptorset::DescriptorSetLayout::ValidateCreateInfo(debug_report_data *report_data,
                                                              const VkDescriptorSetLayoutCreateInfo *create_info) {
    bool skip = false;
    std::unordered_set<uint32_t> bindings;
    for (uint32_t i = 0; i < create_info->bindingCount; ++i) {
        if (!bindings.insert(create_info->pBindings[i].binding).second) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            VALIDATION_ERROR_0500022e, "DS", "duplicated binding number in VkDescriptorSetLayoutBinding. %s",
                            validation_error_map[VALIDATION_ERROR_0500022e]);
        }
    }
    return skip;
}

// put all bindings into the given set
void cvdescriptorset::DescriptorSetLayout::FillBindingSet(std::unordered_set<uint32_t> *binding_set) const {
    for (auto binding_index_pair : binding_to_index_map_) binding_set->insert(binding_index_pair.first);
}

VkDescriptorSetLayoutBinding const *cvdescriptorset::DescriptorSetLayout::GetDescriptorSetLayoutBindingPtrFromBinding(
    const uint32_t binding) const {
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        return bindings_[bi_itr->second].ptr();
    }
    return nullptr;
}
VkDescriptorSetLayoutBinding const *cvdescriptorset::DescriptorSetLayout::GetDescriptorSetLayoutBindingPtrFromIndex(
    const uint32_t index) const {
    if (index >= bindings_.size()) return nullptr;
    return bindings_[index].ptr();
}
// Return descriptorCount for given binding, 0 if index is unavailable
uint32_t cvdescriptorset::DescriptorSetLayout::GetDescriptorCountFromBinding(const uint32_t binding) const {
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        return bindings_[bi_itr->second].descriptorCount;
    }
    return 0;
}
// Return descriptorCount for given index, 0 if index is unavailable
uint32_t cvdescriptorset::DescriptorSetLayout::GetDescriptorCountFromIndex(const uint32_t index) const {
    if (index >= bindings_.size()) return 0;
    return bindings_[index].descriptorCount;
}
// For the given binding, return descriptorType
VkDescriptorType cvdescriptorset::DescriptorSetLayout::GetTypeFromBinding(const uint32_t binding) const {
    assert(binding_to_index_map_.count(binding));
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        return bindings_[bi_itr->second].descriptorType;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}
// For the given index, return descriptorType
VkDescriptorType cvdescriptorset::DescriptorSetLayout::GetTypeFromIndex(const uint32_t index) const {
    assert(index < bindings_.size());
    return bindings_[index].descriptorType;
}
// For the given global index, return descriptorType
//  Currently just counting up through bindings_, may improve this in future
VkDescriptorType cvdescriptorset::DescriptorSetLayout::GetTypeFromGlobalIndex(const uint32_t index) const {
    uint32_t global_offset = 0;
    for (auto binding : bindings_) {
        global_offset += binding.descriptorCount;
        if (index < global_offset) return binding.descriptorType;
    }
    assert(0);  // requested global index is out of bounds
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}
// For the given binding, return stageFlags
VkShaderStageFlags cvdescriptorset::DescriptorSetLayout::GetStageFlagsFromBinding(const uint32_t binding) const {
    assert(binding_to_index_map_.count(binding));
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        return bindings_[bi_itr->second].stageFlags;
    }
    return VkShaderStageFlags(0);
}
// For the given binding, return start index
uint32_t cvdescriptorset::DescriptorSetLayout::GetGlobalStartIndexFromBinding(const uint32_t binding) const {
    assert(binding_to_global_start_index_map_.count(binding));
    const auto &btgsi_itr = binding_to_global_start_index_map_.find(binding);
    if (btgsi_itr != binding_to_global_start_index_map_.end()) {
        return btgsi_itr->second;
    }
    // In error case max uint32_t so index is out of bounds to break ASAP
    assert(0);
    return 0xFFFFFFFF;
}
// For the given binding, return end index
uint32_t cvdescriptorset::DescriptorSetLayout::GetGlobalEndIndexFromBinding(const uint32_t binding) const {
    assert(binding_to_global_end_index_map_.count(binding));
    const auto &btgei_itr = binding_to_global_end_index_map_.find(binding);
    if (btgei_itr != binding_to_global_end_index_map_.end()) {
        return btgei_itr->second;
    }
    // In error case max uint32_t so index is out of bounds to break ASAP
    assert(0);
    return 0xFFFFFFFF;
}
// For given binding, return ptr to ImmutableSampler array
VkSampler const *cvdescriptorset::DescriptorSetLayout::GetImmutableSamplerPtrFromBinding(const uint32_t binding) const {
    assert(binding_to_index_map_.count(binding));
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        return bindings_[bi_itr->second].pImmutableSamplers;
    }
    return nullptr;
}
// Move to next valid binding having a non-zero binding count
uint32_t cvdescriptorset::DescriptorSetLayout::GetNextValidBinding(const uint32_t binding) const {
    uint32_t new_binding = binding;
    do {
        new_binding++;
    } while (GetDescriptorCountFromBinding(new_binding) == 0);
    return new_binding;
}
// For given index, return ptr to ImmutableSampler array
VkSampler const *cvdescriptorset::DescriptorSetLayout::GetImmutableSamplerPtrFromIndex(const uint32_t index) const {
    assert(index < bindings_.size());
    return bindings_[index].pImmutableSamplers;
}
// If our layout is compatible with rh_ds_layout, return true,
//  else return false and fill in error_msg will description of what causes incompatibility
bool cvdescriptorset::DescriptorSetLayout::IsCompatible(DescriptorSetLayout const *const rh_ds_layout,
                                                        std::string *error_msg) const {
    // Trivial case
    if (layout_ == rh_ds_layout->GetDescriptorSetLayout()) return true;
    if (descriptor_count_ != rh_ds_layout->descriptor_count_) {
        std::stringstream error_str;
        error_str << "DescriptorSetLayout " << layout_ << " has " << descriptor_count_ << " descriptors, but DescriptorSetLayout "
                  << rh_ds_layout->GetDescriptorSetLayout() << ", which comes from pipelineLayout, has "
                  << rh_ds_layout->descriptor_count_ << " descriptors.";
        *error_msg = error_str.str();
        return false;  // trivial fail case
    }
    // Descriptor counts match so need to go through bindings one-by-one
    //  and verify that type and stageFlags match
    for (auto binding : bindings_) {
        // TODO : Do we also need to check immutable samplers?
        // VkDescriptorSetLayoutBinding *rh_binding;
        if (binding.descriptorCount != rh_ds_layout->GetDescriptorCountFromBinding(binding.binding)) {
            std::stringstream error_str;
            error_str << "Binding " << binding.binding << " for DescriptorSetLayout " << layout_ << " has a descriptorCount of "
                      << binding.descriptorCount << " but binding " << binding.binding << " for DescriptorSetLayout "
                      << rh_ds_layout->GetDescriptorSetLayout() << ", which comes from pipelineLayout, has a descriptorCount of "
                      << rh_ds_layout->GetDescriptorCountFromBinding(binding.binding);
            *error_msg = error_str.str();
            return false;
        } else if (binding.descriptorType != rh_ds_layout->GetTypeFromBinding(binding.binding)) {
            std::stringstream error_str;
            error_str << "Binding " << binding.binding << " for DescriptorSetLayout " << layout_ << " is type '"
                      << string_VkDescriptorType(binding.descriptorType) << "' but binding " << binding.binding
                      << " for DescriptorSetLayout " << rh_ds_layout->GetDescriptorSetLayout()
                      << ", which comes from pipelineLayout, is type '"
                      << string_VkDescriptorType(rh_ds_layout->GetTypeFromBinding(binding.binding)) << "'";
            *error_msg = error_str.str();
            return false;
        } else if (binding.stageFlags != rh_ds_layout->GetStageFlagsFromBinding(binding.binding)) {
            std::stringstream error_str;
            error_str << "Binding " << binding.binding << " for DescriptorSetLayout " << layout_ << " has stageFlags "
                      << binding.stageFlags << " but binding " << binding.binding << " for DescriptorSetLayout "
                      << rh_ds_layout->GetDescriptorSetLayout() << ", which comes from pipelineLayout, has stageFlags "
                      << rh_ds_layout->GetStageFlagsFromBinding(binding.binding);
            *error_msg = error_str.str();
            return false;
        }
    }
    return true;
}

bool cvdescriptorset::DescriptorSetLayout::IsNextBindingConsistent(const uint32_t binding) const {
    if (!binding_to_index_map_.count(binding + 1)) return false;
    auto const &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.end()) {
        const auto &next_bi_itr = binding_to_index_map_.find(binding + 1);
        if (next_bi_itr != binding_to_index_map_.end()) {
            auto type = bindings_[bi_itr->second].descriptorType;
            auto stage_flags = bindings_[bi_itr->second].stageFlags;
            auto immut_samp = bindings_[bi_itr->second].pImmutableSamplers ? true : false;
            if ((type != bindings_[next_bi_itr->second].descriptorType) ||
                (stage_flags != bindings_[next_bi_itr->second].stageFlags) ||
                (immut_samp != (bindings_[next_bi_itr->second].pImmutableSamplers ? true : false))) {
                return false;
            }
            return true;
        }
    }
    return false;
}
// Starting at offset descriptor of given binding, parse over update_count
//  descriptor updates and verify that for any binding boundaries that are crossed, the next binding(s) are all consistent
//  Consistency means that their type, stage flags, and whether or not they use immutable samplers matches
//  If so, return true. If not, fill in error_msg and return false
bool cvdescriptorset::DescriptorSetLayout::VerifyUpdateConsistency(uint32_t current_binding, uint32_t offset, uint32_t update_count,
                                                                   const char *type, const VkDescriptorSet set,
                                                                   std::string *error_msg) const {
    // Verify consecutive bindings match (if needed)
    auto orig_binding = current_binding;
    // Track count of descriptors in the current_bindings that are remaining to be updated
    auto binding_remaining = GetDescriptorCountFromBinding(current_binding);
    // First, it's legal to offset beyond your own binding so handle that case
    //  Really this is just searching for the binding in which the update begins and adjusting offset accordingly
    while (offset >= binding_remaining) {
        // Advance to next binding, decrement offset by binding size
        offset -= binding_remaining;
        binding_remaining = GetDescriptorCountFromBinding(++current_binding);
    }
    binding_remaining -= offset;
    while (update_count > binding_remaining) {  // While our updates overstep current binding
        // Verify next consecutive binding matches type, stage flags & immutable sampler use
        if (!IsNextBindingConsistent(current_binding++)) {
            std::stringstream error_str;
            error_str << "Attempting " << type << " descriptor set " << set << " binding #" << orig_binding << " with #"
                      << update_count << " descriptors being updated but this update oversteps the bounds of this binding and the "
                                         "next binding is not consistent with current binding so this update is invalid.";
            *error_msg = error_str.str();
            return false;
        }
        // For sake of this check consider the bindings updated and grab count for next binding
        update_count -= binding_remaining;
        binding_remaining = GetDescriptorCountFromBinding(current_binding);
    }
    return true;
}

cvdescriptorset::AllocateDescriptorSetsData::AllocateDescriptorSetsData(uint32_t count)
    : required_descriptors_by_type{}, layout_nodes(count, nullptr) {}

cvdescriptorset::DescriptorSet::DescriptorSet(const VkDescriptorSet set, const VkDescriptorPool pool,
                                              const std::shared_ptr<DescriptorSetLayout const> &layout, const layer_data *dev_data)
    : some_update_(false),
      set_(set),
      pool_state_(nullptr),
      p_layout_(layout),
      device_data_(dev_data),
      limits_(GetPhysDevProperties(dev_data)->properties.limits) {
    pool_state_ = GetDescriptorPoolState(dev_data, pool);
    // Foreach binding, create default descriptors of given type
    for (uint32_t i = 0; i < p_layout_->GetBindingCount(); ++i) {
        auto type = p_layout_->GetTypeFromIndex(i);
        switch (type) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                auto immut_sampler = p_layout_->GetImmutableSamplerPtrFromIndex(i);
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di) {
                    if (immut_sampler) {
                        descriptors_.emplace_back(new SamplerDescriptor(immut_sampler + di));
                        some_update_ = true;  // Immutable samplers are updated at creation
                    } else
                        descriptors_.emplace_back(new SamplerDescriptor(nullptr));
                }
                break;
            }
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                auto immut = p_layout_->GetImmutableSamplerPtrFromIndex(i);
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di) {
                    if (immut) {
                        descriptors_.emplace_back(new ImageSamplerDescriptor(immut + di));
                        some_update_ = true;  // Immutable samplers are updated at creation
                    } else
                        descriptors_.emplace_back(new ImageSamplerDescriptor(nullptr));
                }
                break;
            }
            // ImageDescriptors
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new ImageDescriptor(type));
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new TexelDescriptor(type));
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di)
                    descriptors_.emplace_back(new BufferDescriptor(type));
                break;
            default:
                assert(0);  // Bad descriptor type specified
                break;
        }
    }
}

cvdescriptorset::DescriptorSet::~DescriptorSet() { InvalidateBoundCmdBuffers(); }

static std::string string_descriptor_req_view_type(descriptor_req req) {
    std::string result("");
    for (unsigned i = 0; i <= VK_IMAGE_VIEW_TYPE_END_RANGE; i++) {
        if (req & (1 << i)) {
            if (result.size()) result += ", ";
            result += string_VkImageViewType(VkImageViewType(i));
        }
    }

    if (!result.size()) result = "(none)";

    return result;
}

// Is this sets underlying layout compatible with passed in layout according to "Pipeline Layout Compatibility" in spec?
bool cvdescriptorset::DescriptorSet::IsCompatible(DescriptorSetLayout const *const layout, std::string *error) const {
    return layout->IsCompatible(p_layout_.get(), error);
}

// Validate that the state of this set is appropriate for the given bindings and dynamic_offsets at Draw time
//  This includes validating that all descriptors in the given bindings are updated,
//  that any update buffers are valid, and that any dynamic offsets are within the bounds of their buffers.
// Return true if state is acceptable, or false and write an error message into error string
bool cvdescriptorset::DescriptorSet::ValidateDrawState(const std::map<uint32_t, descriptor_req> &bindings,
                                                       const std::vector<uint32_t> &dynamic_offsets, const GLOBAL_CB_NODE *cb_node,
                                                       const char *caller, std::string *error) const {
    for (auto binding_pair : bindings) {
        auto binding = binding_pair.first;
        if (!p_layout_->HasBinding(binding)) {
            std::stringstream error_str;
            error_str << "Attempting to validate DrawState for binding #" << binding
                      << " which is an invalid binding for this descriptor set.";
            *error = error_str.str();
            return false;
        }
        auto start_idx = p_layout_->GetGlobalStartIndexFromBinding(binding);
        auto end_idx = p_layout_->GetGlobalEndIndexFromBinding(binding);
        auto array_idx = 0;  // Track array idx if we're dealing with array descriptors
        for (uint32_t i = start_idx; i <= end_idx; ++i, ++array_idx) {
            if (!descriptors_[i]->updated) {
                std::stringstream error_str;
                error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                          << " is being used in draw but has not been updated.";
                *error = error_str.str();
                return false;
            } else {
                auto descriptor_class = descriptors_[i]->GetClass();
                if (descriptor_class == GeneralBuffer) {
                    // Verify that buffers are valid
                    auto buffer = static_cast<BufferDescriptor *>(descriptors_[i].get())->GetBuffer();
                    auto buffer_node = GetBufferState(device_data_, buffer);
                    if (!buffer_node) {
                        std::stringstream error_str;
                        error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                                  << " references invalid buffer " << buffer << ".";
                        *error = error_str.str();
                        return false;
                    } else {
                        for (auto mem_binding : buffer_node->GetBoundMemory()) {
                            if (!GetMemObjInfo(device_data_, mem_binding)) {
                                std::stringstream error_str;
                                error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                                          << " uses buffer " << buffer << " that references invalid memory " << mem_binding << ".";
                                *error = error_str.str();
                                return false;
                            }
                        }
                    }
                    if (descriptors_[i]->IsDynamic()) {
                        // Validate that dynamic offsets are within the buffer
                        auto buffer_size = buffer_node->createInfo.size;
                        auto range = static_cast<BufferDescriptor *>(descriptors_[i].get())->GetRange();
                        auto desc_offset = static_cast<BufferDescriptor *>(descriptors_[i].get())->GetOffset();
                        auto dyn_offset = dynamic_offsets[GetDynamicOffsetIndexFromBinding(binding) + array_idx];
                        if (VK_WHOLE_SIZE == range) {
                            if ((dyn_offset + desc_offset) > buffer_size) {
                                std::stringstream error_str;
                                error_str << "Dynamic descriptor in binding #" << binding << " at global descriptor index " << i
                                          << " uses buffer " << buffer << " with update range of VK_WHOLE_SIZE has dynamic offset "
                                          << dyn_offset << " combined with offset " << desc_offset
                                          << " that oversteps the buffer size of " << buffer_size << ".";
                                *error = error_str.str();
                                return false;
                            }
                        } else {
                            if ((dyn_offset + desc_offset + range) > buffer_size) {
                                std::stringstream error_str;
                                error_str << "Dynamic descriptor in binding #" << binding << " at global descriptor index " << i
                                          << " uses buffer " << buffer << " with dynamic offset " << dyn_offset
                                          << " combined with offset " << desc_offset << " and range " << range
                                          << " that oversteps the buffer size of " << buffer_size << ".";
                                *error = error_str.str();
                                return false;
                            }
                        }
                    }
                } else if (descriptor_class == ImageSampler || descriptor_class == Image) {
                    VkImageView image_view;
                    VkImageLayout image_layout;
                    if (descriptor_class == ImageSampler) {
                        image_view = static_cast<ImageSamplerDescriptor *>(descriptors_[i].get())->GetImageView();
                        image_layout = static_cast<ImageSamplerDescriptor *>(descriptors_[i].get())->GetImageLayout();
                    } else {
                        image_view = static_cast<ImageDescriptor *>(descriptors_[i].get())->GetImageView();
                        image_layout = static_cast<ImageDescriptor *>(descriptors_[i].get())->GetImageLayout();
                    }
                    auto reqs = binding_pair.second;

                    auto image_view_state = GetImageViewState(device_data_, image_view);
                    if (nullptr == image_view_state) {
                        // Image view must have been destroyed since initial update. Could potentially flag the descriptor
                        //  as "invalid" (updated = false) at DestroyImageView() time and detect this error at bind time
                        std::stringstream error_str;
                        error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                                  << " is using imageView " << image_view << " that has been destroyed.";
                        *error = error_str.str();
                        return false;
                    }
                    auto image_view_ci = image_view_state->create_info;

                    if ((reqs & DESCRIPTOR_REQ_ALL_VIEW_TYPE_BITS) && (~reqs & (1 << image_view_ci.viewType))) {
                        // bad view type
                        std::stringstream error_str;
                        error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                                  << " requires an image view of type " << string_descriptor_req_view_type(reqs) << " but got "
                                  << string_VkImageViewType(image_view_ci.viewType) << ".";
                        *error = error_str.str();
                        return false;
                    }

                    auto image_node = GetImageState(device_data_, image_view_ci.image);
                    assert(image_node);
                    // Verify Image Layout
                    // Copy first mip level into sub_layers and loop over each mip level to verify layout
                    VkImageSubresourceLayers sub_layers;
                    sub_layers.aspectMask = image_view_ci.subresourceRange.aspectMask;
                    sub_layers.baseArrayLayer = image_view_ci.subresourceRange.baseArrayLayer;
                    sub_layers.layerCount = image_view_ci.subresourceRange.layerCount;
                    bool hit_error = false;
                    for (auto cur_level = image_view_ci.subresourceRange.baseMipLevel;
                         cur_level < image_view_ci.subresourceRange.levelCount; ++cur_level) {
                        sub_layers.mipLevel = cur_level;
                        VerifyImageLayout(device_data_, cb_node, image_node, sub_layers, image_layout, VK_IMAGE_LAYOUT_UNDEFINED,
                                          caller, VALIDATION_ERROR_046002b0, &hit_error);
                        if (hit_error) {
                            *error =
                                "Image layout specified at vkUpdateDescriptorSets() time doesn't match actual image layout at "
                                "time descriptor is used. See previous error callback for specific details.";
                            return false;
                        }
                    }
                    // Verify Sample counts
                    if ((reqs & DESCRIPTOR_REQ_SINGLE_SAMPLE) && image_node->createInfo.samples != VK_SAMPLE_COUNT_1_BIT) {
                        std::stringstream error_str;
                        error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                                  << " requires bound image to have VK_SAMPLE_COUNT_1_BIT but got "
                                  << string_VkSampleCountFlagBits(image_node->createInfo.samples) << ".";
                        *error = error_str.str();
                        return false;
                    }
                    if ((reqs & DESCRIPTOR_REQ_MULTI_SAMPLE) && image_node->createInfo.samples == VK_SAMPLE_COUNT_1_BIT) {
                        std::stringstream error_str;
                        error_str << "Descriptor in binding #" << binding << " at global descriptor index " << i
                                  << " requires bound image to have multiple samples, but got VK_SAMPLE_COUNT_1_BIT.";
                        *error = error_str.str();
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// For given bindings, place any update buffers or images into the passed-in unordered_sets
uint32_t cvdescriptorset::DescriptorSet::GetStorageUpdates(const std::map<uint32_t, descriptor_req> &bindings,
                                                           std::unordered_set<VkBuffer> *buffer_set,
                                                           std::unordered_set<VkImageView> *image_set) const {
    auto num_updates = 0;
    for (auto binding_pair : bindings) {
        auto binding = binding_pair.first;
        // If a binding doesn't exist, skip it
        if (!p_layout_->HasBinding(binding)) {
            continue;
        }
        auto start_idx = p_layout_->GetGlobalStartIndexFromBinding(binding);
        if (descriptors_[start_idx]->IsStorage()) {
            if (Image == descriptors_[start_idx]->descriptor_class) {
                for (uint32_t i = 0; i < p_layout_->GetDescriptorCountFromBinding(binding); ++i) {
                    if (descriptors_[start_idx + i]->updated) {
                        image_set->insert(static_cast<ImageDescriptor *>(descriptors_[start_idx + i].get())->GetImageView());
                        num_updates++;
                    }
                }
            } else if (TexelBuffer == descriptors_[start_idx]->descriptor_class) {
                for (uint32_t i = 0; i < p_layout_->GetDescriptorCountFromBinding(binding); ++i) {
                    if (descriptors_[start_idx + i]->updated) {
                        auto bufferview = static_cast<TexelDescriptor *>(descriptors_[start_idx + i].get())->GetBufferView();
                        auto bv_state = GetBufferViewState(device_data_, bufferview);
                        if (bv_state) {
                            buffer_set->insert(bv_state->create_info.buffer);
                            num_updates++;
                        }
                    }
                }
            } else if (GeneralBuffer == descriptors_[start_idx]->descriptor_class) {
                for (uint32_t i = 0; i < p_layout_->GetDescriptorCountFromBinding(binding); ++i) {
                    if (descriptors_[start_idx + i]->updated) {
                        buffer_set->insert(static_cast<BufferDescriptor *>(descriptors_[start_idx + i].get())->GetBuffer());
                        num_updates++;
                    }
                }
            }
        }
    }
    return num_updates;
}
// Set is being deleted or updates so invalidate all bound cmd buffers
void cvdescriptorset::DescriptorSet::InvalidateBoundCmdBuffers() {
    core_validation::invalidateCommandBuffers(device_data_, cb_bindings, {HandleToUint64(set_), kVulkanObjectTypeDescriptorSet});
}
// Perform write update in given update struct
void cvdescriptorset::DescriptorSet::PerformWriteUpdate(const VkWriteDescriptorSet *update) {
    // Perform update on a per-binding basis as consecutive updates roll over to next binding
    auto descriptors_remaining = update->descriptorCount;
    auto binding_being_updated = update->dstBinding;
    auto offset = update->dstArrayElement;
    while (descriptors_remaining) {
        uint32_t update_count = std::min(descriptors_remaining, GetDescriptorCountFromBinding(binding_being_updated));
        auto global_idx = p_layout_->GetGlobalStartIndexFromBinding(binding_being_updated) + offset;
        // Loop over the updates for a single binding at a time
        for (uint32_t di = 0; di < update_count; ++di) {
            descriptors_[global_idx + di]->WriteUpdate(update, di);
        }
        // Roll over to next binding in case of consecutive update
        descriptors_remaining -= update_count;
        offset = 0;
        binding_being_updated++;
    }
    if (update->descriptorCount) some_update_ = true;

    InvalidateBoundCmdBuffers();
}
// Validate Copy update
bool cvdescriptorset::DescriptorSet::ValidateCopyUpdate(const debug_report_data *report_data, const VkCopyDescriptorSet *update,
                                                        const DescriptorSet *src_set, UNIQUE_VALIDATION_ERROR_CODE *error_code,
                                                        std::string *error_msg) {
    // Verify idle ds
    if (in_use.load()) {
        // TODO : Re-using Free Idle error code, need copy update idle error code
        *error_code = VALIDATION_ERROR_2860026a;
        std::stringstream error_str;
        error_str << "Cannot call vkUpdateDescriptorSets() to perform copy update on descriptor set " << set_
                  << " that is in use by a command buffer";
        *error_msg = error_str.str();
        return false;
    }
    if (!p_layout_->HasBinding(update->dstBinding)) {
        *error_code = VALIDATION_ERROR_032002b6;
        std::stringstream error_str;
        error_str << "DescriptorSet " << set_ << " does not have copy update dest binding of " << update->dstBinding;
        *error_msg = error_str.str();
        return false;
    }
    if (!src_set->HasBinding(update->srcBinding)) {
        *error_code = VALIDATION_ERROR_032002b2;
        std::stringstream error_str;
        error_str << "DescriptorSet " << set_ << " does not have copy update src binding of " << update->srcBinding;
        *error_msg = error_str.str();
        return false;
    }
    // src & dst set bindings are valid
    // Check bounds of src & dst
    auto src_start_idx = src_set->GetGlobalStartIndexFromBinding(update->srcBinding) + update->srcArrayElement;
    if ((src_start_idx + update->descriptorCount) > src_set->GetTotalDescriptorCount()) {
        // SRC update out of bounds
        *error_code = VALIDATION_ERROR_032002b4;
        std::stringstream error_str;
        error_str << "Attempting copy update from descriptorSet " << update->srcSet << " binding#" << update->srcBinding
                  << " with offset index of " << src_set->GetGlobalStartIndexFromBinding(update->srcBinding)
                  << " plus update array offset of " << update->srcArrayElement << " and update of " << update->descriptorCount
                  << " descriptors oversteps total number of descriptors in set: " << src_set->GetTotalDescriptorCount();
        *error_msg = error_str.str();
        return false;
    }
    auto dst_start_idx = p_layout_->GetGlobalStartIndexFromBinding(update->dstBinding) + update->dstArrayElement;
    if ((dst_start_idx + update->descriptorCount) > p_layout_->GetTotalDescriptorCount()) {
        // DST update out of bounds
        *error_code = VALIDATION_ERROR_032002b8;
        std::stringstream error_str;
        error_str << "Attempting copy update to descriptorSet " << set_ << " binding#" << update->dstBinding
                  << " with offset index of " << p_layout_->GetGlobalStartIndexFromBinding(update->dstBinding)
                  << " plus update array offset of " << update->dstArrayElement << " and update of " << update->descriptorCount
                  << " descriptors oversteps total number of descriptors in set: " << p_layout_->GetTotalDescriptorCount();
        *error_msg = error_str.str();
        return false;
    }
    // Check that types match
    // TODO : Base default error case going from here is VALIDATION_ERROR_0002b8012ba which covers all consistency issues, need more
    // fine-grained error codes
    *error_code = VALIDATION_ERROR_032002ba;
    auto src_type = src_set->GetTypeFromBinding(update->srcBinding);
    auto dst_type = p_layout_->GetTypeFromBinding(update->dstBinding);
    if (src_type != dst_type) {
        std::stringstream error_str;
        error_str << "Attempting copy update to descriptorSet " << set_ << " binding #" << update->dstBinding << " with type "
                  << string_VkDescriptorType(dst_type) << " from descriptorSet " << src_set->GetSet() << " binding #"
                  << update->srcBinding << " with type " << string_VkDescriptorType(src_type) << ". Types do not match";
        *error_msg = error_str.str();
        return false;
    }
    // Verify consistency of src & dst bindings if update crosses binding boundaries
    if ((!src_set->GetLayout()->VerifyUpdateConsistency(update->srcBinding, update->srcArrayElement, update->descriptorCount,
                                                        "copy update from", src_set->GetSet(), error_msg)) ||
        (!p_layout_->VerifyUpdateConsistency(update->dstBinding, update->dstArrayElement, update->descriptorCount, "copy update to",
                                             set_, error_msg))) {
        return false;
    }
    // First make sure source descriptors are updated
    for (uint32_t i = 0; i < update->descriptorCount; ++i) {
        if (!src_set->descriptors_[src_start_idx + i]) {
            std::stringstream error_str;
            error_str << "Attempting copy update from descriptorSet " << src_set << " binding #" << update->srcBinding
                      << " but descriptor at array offset " << update->srcArrayElement + i << " has not been updated";
            *error_msg = error_str.str();
            return false;
        }
    }
    // Update parameters all look good and descriptor updated so verify update contents
    if (!VerifyCopyUpdateContents(update, src_set, src_type, src_start_idx, error_code, error_msg)) return false;

    // All checks passed so update is good
    return true;
}
// Perform Copy update
void cvdescriptorset::DescriptorSet::PerformCopyUpdate(const VkCopyDescriptorSet *update, const DescriptorSet *src_set) {
    auto src_start_idx = src_set->GetGlobalStartIndexFromBinding(update->srcBinding) + update->srcArrayElement;
    auto dst_start_idx = p_layout_->GetGlobalStartIndexFromBinding(update->dstBinding) + update->dstArrayElement;
    // Update parameters all look good so perform update
    for (uint32_t di = 0; di < update->descriptorCount; ++di) {
        descriptors_[dst_start_idx + di]->CopyUpdate(src_set->descriptors_[src_start_idx + di].get());
    }
    if (update->descriptorCount) some_update_ = true;

    InvalidateBoundCmdBuffers();
}

// Bind cb_node to this set and this set to cb_node.
// Prereq: This should be called for a set that has been confirmed to be active for the given cb_node, meaning it's going
//   to be used in a draw by the given cb_node
void cvdescriptorset::DescriptorSet::BindCommandBuffer(GLOBAL_CB_NODE *cb_node,
                                                       const std::map<uint32_t, descriptor_req> &binding_req_map) {
    // bind cb to this descriptor set
    cb_bindings.insert(cb_node);
    // Add bindings for descriptor set, the set's pool, and individual objects in the set
    cb_node->object_bindings.insert({HandleToUint64(set_), kVulkanObjectTypeDescriptorSet});
    pool_state_->cb_bindings.insert(cb_node);
    cb_node->object_bindings.insert({HandleToUint64(pool_state_->pool), kVulkanObjectTypeDescriptorPool});
    // For the active slots, use set# to look up descriptorSet from boundDescriptorSets, and bind all of that descriptor set's
    // resources
    for (auto binding_req_pair : binding_req_map) {
        auto binding = binding_req_pair.first;
        auto start_idx = p_layout_->GetGlobalStartIndexFromBinding(binding);
        auto end_idx = p_layout_->GetGlobalEndIndexFromBinding(binding);
        for (uint32_t i = start_idx; i <= end_idx; ++i) {
            descriptors_[i]->BindCommandBuffer(device_data_, cb_node);
        }
    }
}

cvdescriptorset::SamplerDescriptor::SamplerDescriptor(const VkSampler *immut) : sampler_(VK_NULL_HANDLE), immutable_(false) {
    updated = false;
    descriptor_class = PlainSampler;
    if (immut) {
        sampler_ = *immut;
        immutable_ = true;
        updated = true;
    }
}
// Validate given sampler. Currently this only checks to make sure it exists in the samplerMap
bool cvdescriptorset::ValidateSampler(const VkSampler sampler, const layer_data *dev_data) {
    return (GetSamplerState(dev_data, sampler) != nullptr);
}

bool cvdescriptorset::ValidateImageUpdate(VkImageView image_view, VkImageLayout image_layout, VkDescriptorType type,
                                          const layer_data *dev_data, UNIQUE_VALIDATION_ERROR_CODE *error_code,
                                          std::string *error_msg) {
    // TODO : Defaulting to 00943 for all cases here. Need to create new error codes for various cases.
    *error_code = VALIDATION_ERROR_15c0028c;
    auto iv_state = GetImageViewState(dev_data, image_view);
    if (!iv_state) {
        std::stringstream error_str;
        error_str << "Invalid VkImageView: " << image_view;
        *error_msg = error_str.str();
        return false;
    }
    // Note that when an imageview is created, we validated that memory is bound so no need to re-check here
    // Validate that imageLayout is compatible with aspect_mask and image format
    //  and validate that image usage bits are correct for given usage
    VkImageAspectFlags aspect_mask = iv_state->create_info.subresourceRange.aspectMask;
    VkImage image = iv_state->create_info.image;
    VkFormat format = VK_FORMAT_MAX_ENUM;
    VkImageUsageFlags usage = 0;
    auto image_node = GetImageState(dev_data, image);
    if (image_node) {
        format = image_node->createInfo.format;
        usage = image_node->createInfo.usage;
        // Validate that memory is bound to image
        // TODO: This should have its own valid usage id apart from 2524 which is from CreateImageView case. The only
        //  the error here occurs is if memory bound to a created imageView has been freed.
        if (ValidateMemoryIsBoundToImage(dev_data, image_node, "vkUpdateDescriptorSets()", VALIDATION_ERROR_0ac007f8)) {
            *error_code = VALIDATION_ERROR_0ac007f8;
            *error_msg = "No memory bound to image.";
            return false;
        }

        // KHR_maintenance1 allows rendering into 2D or 2DArray views which slice a 3D image,
        // but not binding them to descriptor sets.
        if (image_node->createInfo.imageType == VK_IMAGE_TYPE_3D &&
            (iv_state->create_info.viewType == VK_IMAGE_VIEW_TYPE_2D ||
             iv_state->create_info.viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY)) {
            *error_code = VALIDATION_ERROR_046002ae;
            *error_msg = "ImageView must not be a 2D or 2DArray view of a 3D image";
            return false;
        }
    }
    // First validate that format and layout are compatible
    if (format == VK_FORMAT_MAX_ENUM) {
        std::stringstream error_str;
        error_str << "Invalid image (" << image << ") in imageView (" << image_view << ").";
        *error_msg = error_str.str();
        return false;
    }
    // TODO : The various image aspect and format checks here are based on general spec language in 11.5 Image Views section under
    // vkCreateImageView(). What's the best way to create unique id for these cases?
    bool ds = FormatIsDepthOrStencil(format);
    switch (image_layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            // Only Color bit must be set
            if ((aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != VK_IMAGE_ASPECT_COLOR_BIT) {
                std::stringstream error_str;
                error_str << "ImageView (" << image_view << ") uses layout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL but does "
                                                            "not have VK_IMAGE_ASPECT_COLOR_BIT set.";
                *error_msg = error_str.str();
                return false;
            }
            // format must NOT be DS
            if (ds) {
                std::stringstream error_str;
                error_str << "ImageView (" << image_view
                          << ") uses layout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL but the image format is "
                          << string_VkFormat(format) << " which is not a color format.";
                *error_msg = error_str.str();
                return false;
            }
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            // Depth or stencil bit must be set, but both must NOT be set
            if (aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) {
                if (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) {
                    // both  must NOT be set
                    std::stringstream error_str;
                    error_str << "ImageView (" << image_view << ") has both STENCIL and DEPTH aspects set";
                    *error_msg = error_str.str();
                    return false;
                }
            } else if (!(aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT)) {
                // Neither were set
                std::stringstream error_str;
                error_str << "ImageView (" << image_view << ") has layout " << string_VkImageLayout(image_layout)
                          << " but does not have STENCIL or DEPTH aspects set";
                *error_msg = error_str.str();
                return false;
            }
            // format must be DS
            if (!ds) {
                std::stringstream error_str;
                error_str << "ImageView (" << image_view << ") has layout " << string_VkImageLayout(image_layout)
                          << " but the image format is " << string_VkFormat(format) << " which is not a depth/stencil format.";
                *error_msg = error_str.str();
                return false;
            }
            break;
        default:
            // For other layouts if the source is depth/stencil image, both aspect bits must not be set
            if (ds) {
                if (aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) {
                    if (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) {
                        // both  must NOT be set
                        std::stringstream error_str;
                        error_str << "ImageView (" << image_view << ") has layout " << string_VkImageLayout(image_layout)
                                  << " and is using depth/stencil image of format " << string_VkFormat(format)
                                  << " but it has both STENCIL and DEPTH aspects set, which is illegal. When using a depth/stencil "
                                     "image in a descriptor set, please only set either VK_IMAGE_ASPECT_DEPTH_BIT or "
                                     "VK_IMAGE_ASPECT_STENCIL_BIT depending on whether it will be used for depth reads or stencil "
                                     "reads respectively.";
                        *error_msg = error_str.str();
                        return false;
                    }
                }
            }
            break;
    }
    // Now validate that usage flags are correctly set for given type of update
    //  As we're switching per-type, if any type has specific layout requirements, check those here as well
    // TODO : The various image usage bit requirements are in general spec language for VkImageUsageFlags bit block in 11.3 Images
    // under vkCreateImage()
    // TODO : Need to also validate case VALIDATION_ERROR_15c002a0 where STORAGE_IMAGE & INPUT_ATTACH types must have been created
    // with identify swizzle
    std::string error_usage_bit;
    switch (type) {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
            if (!(usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
                error_usage_bit = "VK_IMAGE_USAGE_SAMPLED_BIT";
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            if (!(usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
                error_usage_bit = "VK_IMAGE_USAGE_STORAGE_BIT";
            } else if (VK_IMAGE_LAYOUT_GENERAL != image_layout) {
                std::stringstream error_str;
                // TODO : Need to create custom enum error codes for these cases
                if (image_node->shared_presentable) {
                    if (VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR != image_layout) {
                        error_str
                            << "ImageView (" << image_view
                            << ") of VK_DESCRIPTOR_TYPE_STORAGE_IMAGE type with a front-buffered image is being updated with "
                               "layout "
                            << string_VkImageLayout(image_layout)
                            << " but according to spec section 13.1 Descriptor Types, 'Front-buffered images that report support "
                               "for "
                               "VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT must be in the VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR layout.'";
                        *error_msg = error_str.str();
                        return false;
                    }
                } else if (VK_IMAGE_LAYOUT_GENERAL != image_layout) {
                    error_str
                        << "ImageView (" << image_view << ") of VK_DESCRIPTOR_TYPE_STORAGE_IMAGE type is being updated with layout "
                        << string_VkImageLayout(image_layout)
                        << " but according to spec section 13.1 Descriptor Types, 'Load and store operations on storage images can "
                           "only be done on images in VK_IMAGE_LAYOUT_GENERAL layout.'";
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
            if (!(usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) {
                error_usage_bit = "VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT";
            }
            break;
        }
        default:
            break;
    }
    if (!error_usage_bit.empty()) {
        std::stringstream error_str;
        error_str << "ImageView (" << image_view << ") with usage mask 0x" << usage
                  << " being used for a descriptor update of type " << string_VkDescriptorType(type) << " does not have "
                  << error_usage_bit << " set.";
        *error_msg = error_str.str();
        return false;
    }
    return true;
}

void cvdescriptorset::SamplerDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    sampler_ = update->pImageInfo[index].sampler;
    updated = true;
}

void cvdescriptorset::SamplerDescriptor::CopyUpdate(const Descriptor *src) {
    if (!immutable_) {
        auto update_sampler = static_cast<const SamplerDescriptor *>(src)->sampler_;
        sampler_ = update_sampler;
    }
    updated = true;
}

void cvdescriptorset::SamplerDescriptor::BindCommandBuffer(const layer_data *dev_data, GLOBAL_CB_NODE *cb_node) {
    if (!immutable_) {
        auto sampler_state = GetSamplerState(dev_data, sampler_);
        if (sampler_state) core_validation::AddCommandBufferBindingSampler(cb_node, sampler_state);
    }
}

cvdescriptorset::ImageSamplerDescriptor::ImageSamplerDescriptor(const VkSampler *immut)
    : sampler_(VK_NULL_HANDLE), immutable_(false), image_view_(VK_NULL_HANDLE), image_layout_(VK_IMAGE_LAYOUT_UNDEFINED) {
    updated = false;
    descriptor_class = ImageSampler;
    if (immut) {
        sampler_ = *immut;
        immutable_ = true;
    }
}

void cvdescriptorset::ImageSamplerDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    const auto &image_info = update->pImageInfo[index];
    sampler_ = image_info.sampler;
    image_view_ = image_info.imageView;
    image_layout_ = image_info.imageLayout;
}

void cvdescriptorset::ImageSamplerDescriptor::CopyUpdate(const Descriptor *src) {
    if (!immutable_) {
        auto update_sampler = static_cast<const ImageSamplerDescriptor *>(src)->sampler_;
        sampler_ = update_sampler;
    }
    auto image_view = static_cast<const ImageSamplerDescriptor *>(src)->image_view_;
    auto image_layout = static_cast<const ImageSamplerDescriptor *>(src)->image_layout_;
    updated = true;
    image_view_ = image_view;
    image_layout_ = image_layout;
}

void cvdescriptorset::ImageSamplerDescriptor::BindCommandBuffer(const layer_data *dev_data, GLOBAL_CB_NODE *cb_node) {
    // First add binding for any non-immutable sampler
    if (!immutable_) {
        auto sampler_state = GetSamplerState(dev_data, sampler_);
        if (sampler_state) core_validation::AddCommandBufferBindingSampler(cb_node, sampler_state);
    }
    // Add binding for image
    auto iv_state = GetImageViewState(dev_data, image_view_);
    if (iv_state) {
        core_validation::AddCommandBufferBindingImageView(dev_data, cb_node, iv_state);
    }
}

cvdescriptorset::ImageDescriptor::ImageDescriptor(const VkDescriptorType type)
    : storage_(false), image_view_(VK_NULL_HANDLE), image_layout_(VK_IMAGE_LAYOUT_UNDEFINED) {
    updated = false;
    descriptor_class = Image;
    if (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE == type) storage_ = true;
};

void cvdescriptorset::ImageDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    const auto &image_info = update->pImageInfo[index];
    image_view_ = image_info.imageView;
    image_layout_ = image_info.imageLayout;
}

void cvdescriptorset::ImageDescriptor::CopyUpdate(const Descriptor *src) {
    auto image_view = static_cast<const ImageDescriptor *>(src)->image_view_;
    auto image_layout = static_cast<const ImageDescriptor *>(src)->image_layout_;
    updated = true;
    image_view_ = image_view;
    image_layout_ = image_layout;
}

void cvdescriptorset::ImageDescriptor::BindCommandBuffer(const layer_data *dev_data, GLOBAL_CB_NODE *cb_node) {
    // Add binding for image
    auto iv_state = GetImageViewState(dev_data, image_view_);
    if (iv_state) {
        core_validation::AddCommandBufferBindingImageView(dev_data, cb_node, iv_state);
    }
}

cvdescriptorset::BufferDescriptor::BufferDescriptor(const VkDescriptorType type)
    : storage_(false), dynamic_(false), buffer_(VK_NULL_HANDLE), offset_(0), range_(0) {
    updated = false;
    descriptor_class = GeneralBuffer;
    if (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC == type) {
        dynamic_ = true;
    } else if (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER == type) {
        storage_ = true;
    } else if (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC == type) {
        dynamic_ = true;
        storage_ = true;
    }
}
void cvdescriptorset::BufferDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    const auto &buffer_info = update->pBufferInfo[index];
    buffer_ = buffer_info.buffer;
    offset_ = buffer_info.offset;
    range_ = buffer_info.range;
}

void cvdescriptorset::BufferDescriptor::CopyUpdate(const Descriptor *src) {
    auto buff_desc = static_cast<const BufferDescriptor *>(src);
    updated = true;
    buffer_ = buff_desc->buffer_;
    offset_ = buff_desc->offset_;
    range_ = buff_desc->range_;
}

void cvdescriptorset::BufferDescriptor::BindCommandBuffer(const layer_data *dev_data, GLOBAL_CB_NODE *cb_node) {
    auto buffer_node = GetBufferState(dev_data, buffer_);
    if (buffer_node) core_validation::AddCommandBufferBindingBuffer(dev_data, cb_node, buffer_node);
}

cvdescriptorset::TexelDescriptor::TexelDescriptor(const VkDescriptorType type) : buffer_view_(VK_NULL_HANDLE), storage_(false) {
    updated = false;
    descriptor_class = TexelBuffer;
    if (VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER == type) storage_ = true;
};

void cvdescriptorset::TexelDescriptor::WriteUpdate(const VkWriteDescriptorSet *update, const uint32_t index) {
    updated = true;
    buffer_view_ = update->pTexelBufferView[index];
}

void cvdescriptorset::TexelDescriptor::CopyUpdate(const Descriptor *src) {
    updated = true;
    buffer_view_ = static_cast<const TexelDescriptor *>(src)->buffer_view_;
}

void cvdescriptorset::TexelDescriptor::BindCommandBuffer(const layer_data *dev_data, GLOBAL_CB_NODE *cb_node) {
    auto bv_state = GetBufferViewState(dev_data, buffer_view_);
    if (bv_state) {
        core_validation::AddCommandBufferBindingBufferView(dev_data, cb_node, bv_state);
    }
}

// This is a helper function that iterates over a set of Write and Copy updates, pulls the DescriptorSet* for updated
//  sets, and then calls their respective Validate[Write|Copy]Update functions.
// If the update hits an issue for which the callback returns "true", meaning that the call down the chain should
//  be skipped, then true is returned.
// If there is no issue with the update, then false is returned.
bool cvdescriptorset::ValidateUpdateDescriptorSets(const debug_report_data *report_data, const layer_data *dev_data,
                                                   uint32_t write_count, const VkWriteDescriptorSet *p_wds, uint32_t copy_count,
                                                   const VkCopyDescriptorSet *p_cds) {
    bool skip = false;
    // Validate Write updates
    for (uint32_t i = 0; i < write_count; i++) {
        auto dest_set = p_wds[i].dstSet;
        auto set_node = core_validation::GetSetNode(dev_data, dest_set);
        if (!set_node) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
                        HandleToUint64(dest_set), __LINE__, DRAWSTATE_INVALID_DESCRIPTOR_SET, "DS",
                        "Cannot call vkUpdateDescriptorSets() on descriptor set 0x%" PRIxLEAST64 " that has not been allocated.",
                        HandleToUint64(dest_set));
        } else {
            UNIQUE_VALIDATION_ERROR_CODE error_code;
            std::string error_str;
            if (!set_node->ValidateWriteUpdate(report_data, &p_wds[i], &error_code, &error_str)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
                                HandleToUint64(dest_set), __LINE__, error_code, "DS",
                                "vkUpdateDescriptorsSets() failed write update validation for Descriptor Set 0x%" PRIx64
                                " with error: %s. %s",
                                HandleToUint64(dest_set), error_str.c_str(), validation_error_map[error_code]);
            }
        }
    }
    // Now validate copy updates
    for (uint32_t i = 0; i < copy_count; ++i) {
        auto dst_set = p_cds[i].dstSet;
        auto src_set = p_cds[i].srcSet;
        auto src_node = core_validation::GetSetNode(dev_data, src_set);
        auto dst_node = core_validation::GetSetNode(dev_data, dst_set);
        // Object_tracker verifies that src & dest descriptor set are valid
        assert(src_node);
        assert(dst_node);
        UNIQUE_VALIDATION_ERROR_CODE error_code;
        std::string error_str;
        if (!dst_node->ValidateCopyUpdate(report_data, &p_cds[i], src_node, &error_code, &error_str)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
                            HandleToUint64(dst_set), __LINE__, error_code, "DS",
                            "vkUpdateDescriptorsSets() failed copy update from Descriptor Set 0x%" PRIx64
                            " to Descriptor Set 0x%" PRIx64 " with error: %s. %s",
                            HandleToUint64(src_set), HandleToUint64(dst_set), error_str.c_str(), validation_error_map[error_code]);
        }
    }
    return skip;
}
// This is a helper function that iterates over a set of Write and Copy updates, pulls the DescriptorSet* for updated
//  sets, and then calls their respective Perform[Write|Copy]Update functions.
// Prerequisite : ValidateUpdateDescriptorSets() should be called and return "false" prior to calling PerformUpdateDescriptorSets()
//  with the same set of updates.
// This is split from the validate code to allow validation prior to calling down the chain, and then update after
//  calling down the chain.
void cvdescriptorset::PerformUpdateDescriptorSets(const layer_data *dev_data, uint32_t write_count,
                                                  const VkWriteDescriptorSet *p_wds, uint32_t copy_count,
                                                  const VkCopyDescriptorSet *p_cds) {
    // Write updates first
    uint32_t i = 0;
    for (i = 0; i < write_count; ++i) {
        auto dest_set = p_wds[i].dstSet;
        auto set_node = core_validation::GetSetNode(dev_data, dest_set);
        if (set_node) {
            set_node->PerformWriteUpdate(&p_wds[i]);
        }
    }
    // Now copy updates
    for (i = 0; i < copy_count; ++i) {
        auto dst_set = p_cds[i].dstSet;
        auto src_set = p_cds[i].srcSet;
        auto src_node = core_validation::GetSetNode(dev_data, src_set);
        auto dst_node = core_validation::GetSetNode(dev_data, dst_set);
        if (src_node && dst_node) {
            dst_node->PerformCopyUpdate(&p_cds[i], src_node);
        }
    }
}
// This helper function carries out the state updates for descriptor updates peformed via update templates. It basically collects
// data and leverages the PerformUpdateDescriptor helper functions to do this.
void cvdescriptorset::PerformUpdateDescriptorSetsWithTemplateKHR(layer_data *device_data, VkDescriptorSet descriptorSet,
                                                                 std::unique_ptr<TEMPLATE_STATE> const &template_state,
                                                                 const void *pData) {
    auto const &create_info = template_state->create_info;

    // Create a vector of write structs
    std::vector<VkWriteDescriptorSet> desc_writes;
    auto layout_obj = GetDescriptorSetLayout(device_data, create_info.descriptorSetLayout);

    // Create a WriteDescriptorSet struct for each template update entry
    for (uint32_t i = 0; i < create_info.descriptorUpdateEntryCount; i++) {
        auto binding_count = layout_obj->GetDescriptorCountFromBinding(create_info.pDescriptorUpdateEntries[i].dstBinding);
        auto binding_being_updated = create_info.pDescriptorUpdateEntries[i].dstBinding;
        auto dst_array_element = create_info.pDescriptorUpdateEntries[i].dstArrayElement;

        for (uint32_t j = 0; j < create_info.pDescriptorUpdateEntries[i].descriptorCount; j++) {
            desc_writes.emplace_back();
            auto &write_entry = desc_writes.back();

            size_t offset = create_info.pDescriptorUpdateEntries[i].offset + j * create_info.pDescriptorUpdateEntries[i].stride;
            char *update_entry = (char *)(pData) + offset;

            if (dst_array_element >= binding_count) {
                dst_array_element = 0;
                binding_being_updated = layout_obj->GetNextValidBinding(binding_being_updated);
            }

            write_entry.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_entry.pNext = NULL;
            write_entry.dstSet = descriptorSet;
            write_entry.dstBinding = binding_being_updated;
            write_entry.dstArrayElement = dst_array_element;
            write_entry.descriptorCount = 1;
            write_entry.descriptorType = create_info.pDescriptorUpdateEntries[i].descriptorType;

            switch (create_info.pDescriptorUpdateEntries[i].descriptorType) {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    write_entry.pImageInfo = reinterpret_cast<VkDescriptorImageInfo *>(update_entry);
                    break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    write_entry.pBufferInfo = reinterpret_cast<VkDescriptorBufferInfo *>(update_entry);
                    break;

                case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                    write_entry.pTexelBufferView = reinterpret_cast<VkBufferView *>(update_entry);
                    break;
                default:
                    assert(0);
                    break;
            }
            dst_array_element++;
        }
    }
    PerformUpdateDescriptorSets(device_data, static_cast<uint32_t>(desc_writes.size()), desc_writes.data(), 0, NULL);
}
// Validate the state for a given write update but don't actually perform the update
//  If an error would occur for this update, return false and fill in details in error_msg string
bool cvdescriptorset::DescriptorSet::ValidateWriteUpdate(const debug_report_data *report_data, const VkWriteDescriptorSet *update,
                                                         UNIQUE_VALIDATION_ERROR_CODE *error_code, std::string *error_msg) {
    // Verify idle ds
    if (in_use.load()) {
        // TODO : Re-using Free Idle error code, need write update idle error code
        *error_code = VALIDATION_ERROR_2860026a;
        std::stringstream error_str;
        error_str << "Cannot call vkUpdateDescriptorSets() to perform write update on descriptor set " << set_
                  << " that is in use by a command buffer";
        *error_msg = error_str.str();
        return false;
    }
    // Verify dst binding exists
    if (!p_layout_->HasBinding(update->dstBinding)) {
        *error_code = VALIDATION_ERROR_15c00276;
        std::stringstream error_str;
        error_str << "DescriptorSet " << set_ << " does not have binding " << update->dstBinding;
        *error_msg = error_str.str();
        return false;
    } else {
        // Make sure binding isn't empty
        if (0 == p_layout_->GetDescriptorCountFromBinding(update->dstBinding)) {
            *error_code = VALIDATION_ERROR_15c00278;
            std::stringstream error_str;
            error_str << "DescriptorSet " << set_ << " cannot updated binding " << update->dstBinding << " that has 0 descriptors";
            *error_msg = error_str.str();
            return false;
        }
    }
    // We know that binding is valid, verify update and do update on each descriptor
    auto start_idx = p_layout_->GetGlobalStartIndexFromBinding(update->dstBinding) + update->dstArrayElement;
    auto type = p_layout_->GetTypeFromBinding(update->dstBinding);
    if (type != update->descriptorType) {
        *error_code = VALIDATION_ERROR_15c0027e;
        std::stringstream error_str;
        error_str << "Attempting write update to descriptor set " << set_ << " binding #" << update->dstBinding << " with type "
                  << string_VkDescriptorType(type) << " but update type is " << string_VkDescriptorType(update->descriptorType);
        *error_msg = error_str.str();
        return false;
    }
    if (update->descriptorCount > (descriptors_.size() - start_idx)) {
        *error_code = VALIDATION_ERROR_15c00282;
        std::stringstream error_str;
        error_str << "Attempting write update to descriptor set " << set_ << " binding #" << update->dstBinding << " with "
                  << descriptors_.size() - start_idx
                  << " descriptors in that binding and all successive bindings of the set, but update of "
                  << update->descriptorCount << " descriptors combined with update array element offset of "
                  << update->dstArrayElement << " oversteps the available number of consecutive descriptors";
        *error_msg = error_str.str();
        return false;
    }
    // Verify consecutive bindings match (if needed)
    if (!p_layout_->VerifyUpdateConsistency(update->dstBinding, update->dstArrayElement, update->descriptorCount, "write update to",
                                            set_, error_msg)) {
        // TODO : Should break out "consecutive binding updates" language into valid usage statements
        *error_code = VALIDATION_ERROR_15c00282;
        return false;
    }
    // Update is within bounds and consistent so last step is to validate update contents
    if (!VerifyWriteUpdateContents(update, start_idx, error_code, error_msg)) {
        std::stringstream error_str;
        error_str << "Write update to descriptor in set " << set_ << " binding #" << update->dstBinding
                  << " failed with error message: " << error_msg->c_str();
        *error_msg = error_str.str();
        return false;
    }
    // All checks passed, update is clean
    return true;
}
// For the given buffer, verify that its creation parameters are appropriate for the given type
//  If there's an error, update the error_msg string with details and return false, else return true
bool cvdescriptorset::DescriptorSet::ValidateBufferUsage(BUFFER_STATE const *buffer_node, VkDescriptorType type,
                                                         UNIQUE_VALIDATION_ERROR_CODE *error_code, std::string *error_msg) const {
    // Verify that usage bits set correctly for given type
    auto usage = buffer_node->createInfo.usage;
    std::string error_usage_bit;
    switch (type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            if (!(usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT)) {
                *error_code = VALIDATION_ERROR_15c0029c;
                error_usage_bit = "VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT";
            }
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            if (!(usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) {
                *error_code = VALIDATION_ERROR_15c0029e;
                error_usage_bit = "VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT";
            }
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            if (!(usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
                *error_code = VALIDATION_ERROR_15c00292;
                error_usage_bit = "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT";
            }
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            if (!(usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
                *error_code = VALIDATION_ERROR_15c00296;
                error_usage_bit = "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT";
            }
            break;
        default:
            break;
    }
    if (!error_usage_bit.empty()) {
        std::stringstream error_str;
        error_str << "Buffer (" << buffer_node->buffer << ") with usage mask 0x" << usage
                  << " being used for a descriptor update of type " << string_VkDescriptorType(type) << " does not have "
                  << error_usage_bit << " set.";
        *error_msg = error_str.str();
        return false;
    }
    return true;
}
// For buffer descriptor updates, verify the buffer usage and VkDescriptorBufferInfo struct which includes:
//  1. buffer is valid
//  2. buffer was created with correct usage flags
//  3. offset is less than buffer size
//  4. range is either VK_WHOLE_SIZE or falls in (0, (buffer size - offset)]
//  5. range and offset are within the device's limits
// If there's an error, update the error_msg string with details and return false, else return true
bool cvdescriptorset::DescriptorSet::ValidateBufferUpdate(VkDescriptorBufferInfo const *buffer_info, VkDescriptorType type,
                                                          UNIQUE_VALIDATION_ERROR_CODE *error_code, std::string *error_msg) const {
    // First make sure that buffer is valid
    auto buffer_node = GetBufferState(device_data_, buffer_info->buffer);
    // Any invalid buffer should already be caught by object_tracker
    assert(buffer_node);
    if (ValidateMemoryIsBoundToBuffer(device_data_, buffer_node, "vkUpdateDescriptorSets()", VALIDATION_ERROR_15c00294)) {
        *error_code = VALIDATION_ERROR_15c00294;
        *error_msg = "No memory bound to buffer.";
        return false;
    }
    // Verify usage bits
    if (!ValidateBufferUsage(buffer_node, type, error_code, error_msg)) {
        // error_msg will have been updated by ValidateBufferUsage()
        return false;
    }
    // offset must be less than buffer size
    if (buffer_info->offset >= buffer_node->createInfo.size) {
        *error_code = VALIDATION_ERROR_044002a8;
        std::stringstream error_str;
        error_str << "VkDescriptorBufferInfo offset of " << buffer_info->offset << " is greater than or equal to buffer "
                  << buffer_node->buffer << " size of " << buffer_node->createInfo.size;
        *error_msg = error_str.str();
        return false;
    }
    if (buffer_info->range != VK_WHOLE_SIZE) {
        // Range must be VK_WHOLE_SIZE or > 0
        if (!buffer_info->range) {
            *error_code = VALIDATION_ERROR_044002aa;
            std::stringstream error_str;
            error_str << "VkDescriptorBufferInfo range is not VK_WHOLE_SIZE and is zero, which is not allowed.";
            *error_msg = error_str.str();
            return false;
        }
        // Range must be VK_WHOLE_SIZE or <= (buffer size - offset)
        if (buffer_info->range > (buffer_node->createInfo.size - buffer_info->offset)) {
            *error_code = VALIDATION_ERROR_044002ac;
            std::stringstream error_str;
            error_str << "VkDescriptorBufferInfo range is " << buffer_info->range << " which is greater than buffer size ("
                      << buffer_node->createInfo.size << ") minus requested offset of " << buffer_info->offset;
            *error_msg = error_str.str();
            return false;
        }
    }
    // Check buffer update sizes against device limits
    if (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER == type || VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC == type) {
        auto max_ub_range = limits_.maxUniformBufferRange;
        // TODO : If range is WHOLE_SIZE, need to make sure underlying buffer size doesn't exceed device max
        if (buffer_info->range != VK_WHOLE_SIZE && buffer_info->range > max_ub_range) {
            *error_code = VALIDATION_ERROR_15c00298;
            std::stringstream error_str;
            error_str << "VkDescriptorBufferInfo range is " << buffer_info->range
                      << " which is greater than this device's maxUniformBufferRange (" << max_ub_range << ")";
            *error_msg = error_str.str();
            return false;
        }
    } else if (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER == type || VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC == type) {
        auto max_sb_range = limits_.maxStorageBufferRange;
        // TODO : If range is WHOLE_SIZE, need to make sure underlying buffer size doesn't exceed device max
        if (buffer_info->range != VK_WHOLE_SIZE && buffer_info->range > max_sb_range) {
            *error_code = VALIDATION_ERROR_15c0029a;
            std::stringstream error_str;
            error_str << "VkDescriptorBufferInfo range is " << buffer_info->range
                      << " which is greater than this device's maxStorageBufferRange (" << max_sb_range << ")";
            *error_msg = error_str.str();
            return false;
        }
    }
    return true;
}

// Verify that the contents of the update are ok, but don't perform actual update
bool cvdescriptorset::DescriptorSet::VerifyWriteUpdateContents(const VkWriteDescriptorSet *update, const uint32_t index,
                                                               UNIQUE_VALIDATION_ERROR_CODE *error_code,
                                                               std::string *error_msg) const {
    switch (update->descriptorType) {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                // Validate image
                auto image_view = update->pImageInfo[di].imageView;
                auto image_layout = update->pImageInfo[di].imageLayout;
                if (!ValidateImageUpdate(image_view, image_layout, update->descriptorType, device_data_, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted write update to combined image sampler descriptor failed due to: "
                              << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            // Intentional fall-through to validate sampler
        }
        case VK_DESCRIPTOR_TYPE_SAMPLER: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                if (!descriptors_[index + di].get()->IsImmutableSampler()) {
                    if (!ValidateSampler(update->pImageInfo[di].sampler, device_data_)) {
                        *error_code = VALIDATION_ERROR_15c0028a;
                        std::stringstream error_str;
                        error_str << "Attempted write update to sampler descriptor with invalid sampler: "
                                  << update->pImageInfo[di].sampler << ".";
                        *error_msg = error_str.str();
                        return false;
                    }
                } else {
                    // TODO : Warn here
                }
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                auto image_view = update->pImageInfo[di].imageView;
                auto image_layout = update->pImageInfo[di].imageLayout;
                if (!ValidateImageUpdate(image_view, image_layout, update->descriptorType, device_data_, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted write update to image descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                auto buffer_view = update->pTexelBufferView[di];
                auto bv_state = GetBufferViewState(device_data_, buffer_view);
                if (!bv_state) {
                    *error_code = VALIDATION_ERROR_15c00286;
                    std::stringstream error_str;
                    error_str << "Attempted write update to texel buffer descriptor with invalid buffer view: " << buffer_view;
                    *error_msg = error_str.str();
                    return false;
                }
                auto buffer = bv_state->create_info.buffer;
                if (!ValidateBufferUsage(GetBufferState(device_data_, buffer), update->descriptorType, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted write update to texel buffer descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                if (!ValidateBufferUpdate(update->pBufferInfo + di, update->descriptorType, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted write update to buffer descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        default:
            assert(0);  // We've already verified update type so should never get here
            break;
    }
    // All checks passed so update contents are good
    return true;
}
// Verify that the contents of the update are ok, but don't perform actual update
bool cvdescriptorset::DescriptorSet::VerifyCopyUpdateContents(const VkCopyDescriptorSet *update, const DescriptorSet *src_set,
                                                              VkDescriptorType type, uint32_t index,
                                                              UNIQUE_VALIDATION_ERROR_CODE *error_code,
                                                              std::string *error_msg) const {
    // Note : Repurposing some Write update error codes here as specific details aren't called out for copy updates like they are
    // for write updates
    switch (src_set->descriptors_[index]->descriptor_class) {
        case PlainSampler: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                if (!src_set->descriptors_[index + di]->IsImmutableSampler()) {
                    auto update_sampler = static_cast<SamplerDescriptor *>(src_set->descriptors_[index + di].get())->GetSampler();
                    if (!ValidateSampler(update_sampler, device_data_)) {
                        *error_code = VALIDATION_ERROR_15c0028a;
                        std::stringstream error_str;
                        error_str << "Attempted copy update to sampler descriptor with invalid sampler: " << update_sampler << ".";
                        *error_msg = error_str.str();
                        return false;
                    }
                } else {
                    // TODO : Warn here
                }
            }
            break;
        }
        case ImageSampler: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                auto img_samp_desc = static_cast<const ImageSamplerDescriptor *>(src_set->descriptors_[index + di].get());
                // First validate sampler
                if (!img_samp_desc->IsImmutableSampler()) {
                    auto update_sampler = img_samp_desc->GetSampler();
                    if (!ValidateSampler(update_sampler, device_data_)) {
                        *error_code = VALIDATION_ERROR_15c0028a;
                        std::stringstream error_str;
                        error_str << "Attempted copy update to sampler descriptor with invalid sampler: " << update_sampler << ".";
                        *error_msg = error_str.str();
                        return false;
                    }
                } else {
                    // TODO : Warn here
                }
                // Validate image
                auto image_view = img_samp_desc->GetImageView();
                auto image_layout = img_samp_desc->GetImageLayout();
                if (!ValidateImageUpdate(image_view, image_layout, type, device_data_, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted copy update to combined image sampler descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        case Image: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                auto img_desc = static_cast<const ImageDescriptor *>(src_set->descriptors_[index + di].get());
                auto image_view = img_desc->GetImageView();
                auto image_layout = img_desc->GetImageLayout();
                if (!ValidateImageUpdate(image_view, image_layout, type, device_data_, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted copy update to image descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        case TexelBuffer: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                auto buffer_view = static_cast<TexelDescriptor *>(src_set->descriptors_[index + di].get())->GetBufferView();
                auto bv_state = GetBufferViewState(device_data_, buffer_view);
                if (!bv_state) {
                    *error_code = VALIDATION_ERROR_15c00286;
                    std::stringstream error_str;
                    error_str << "Attempted copy update to texel buffer descriptor with invalid buffer view: " << buffer_view;
                    *error_msg = error_str.str();
                    return false;
                }
                auto buffer = bv_state->create_info.buffer;
                if (!ValidateBufferUsage(GetBufferState(device_data_, buffer), type, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted copy update to texel buffer descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        case GeneralBuffer: {
            for (uint32_t di = 0; di < update->descriptorCount; ++di) {
                auto buffer = static_cast<BufferDescriptor *>(src_set->descriptors_[index + di].get())->GetBuffer();
                if (!ValidateBufferUsage(GetBufferState(device_data_, buffer), type, error_code, error_msg)) {
                    std::stringstream error_str;
                    error_str << "Attempted copy update to buffer descriptor failed due to: " << error_msg->c_str();
                    *error_msg = error_str.str();
                    return false;
                }
            }
            break;
        }
        default:
            assert(0);  // We've already verified update type so should never get here
            break;
    }
    // All checks passed so update contents are good
    return true;
}
// Update the common AllocateDescriptorSetsData
void cvdescriptorset::UpdateAllocateDescriptorSetsData(const layer_data *dev_data, const VkDescriptorSetAllocateInfo *p_alloc_info,
                                                       AllocateDescriptorSetsData *ds_data) {
    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        auto layout = GetDescriptorSetLayout(dev_data, p_alloc_info->pSetLayouts[i]);
        if (layout) {
            ds_data->layout_nodes[i] = layout;
            // Count total descriptors required per type
            for (uint32_t j = 0; j < layout->GetBindingCount(); ++j) {
                const auto &binding_layout = layout->GetDescriptorSetLayoutBindingPtrFromIndex(j);
                uint32_t typeIndex = static_cast<uint32_t>(binding_layout->descriptorType);
                ds_data->required_descriptors_by_type[typeIndex] += binding_layout->descriptorCount;
            }
        }
        // Any unknown layouts will be flagged as errors during ValidateAllocateDescriptorSets() call
    }
};
// Verify that the state at allocate time is correct, but don't actually allocate the sets yet
bool cvdescriptorset::ValidateAllocateDescriptorSets(const core_validation::layer_data *dev_data,
                                                     const VkDescriptorSetAllocateInfo *p_alloc_info,
                                                     const AllocateDescriptorSetsData *ds_data) {
    bool skip = false;
    auto report_data = core_validation::GetReportData(dev_data);

    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        auto layout = GetDescriptorSetLayout(dev_data, p_alloc_info->pSetLayouts[i]);
        if (!layout) {
            skip |=
                log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT,
                        HandleToUint64(p_alloc_info->pSetLayouts[i]), __LINE__, DRAWSTATE_INVALID_LAYOUT, "DS",
                        "Unable to find set layout node for layout 0x%" PRIxLEAST64 " specified in vkAllocateDescriptorSets() call",
                        HandleToUint64(p_alloc_info->pSetLayouts[i]));
        }
    }
    if (!GetDeviceExtensions(dev_data)->vk_khr_maintenance1) {
        auto pool_state = GetDescriptorPoolState(dev_data, p_alloc_info->descriptorPool);
        // Track number of descriptorSets allowable in this pool
        if (pool_state->availableSets < p_alloc_info->descriptorSetCount) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT,
                            HandleToUint64(pool_state->pool), __LINE__, VALIDATION_ERROR_04c00264, "DS",
                            "Unable to allocate %u descriptorSets from pool 0x%" PRIxLEAST64
                            ". This pool only has %d descriptorSets remaining. %s",
                            p_alloc_info->descriptorSetCount, HandleToUint64(pool_state->pool), pool_state->availableSets,
                            validation_error_map[VALIDATION_ERROR_04c00264]);
        }
        // Determine whether descriptor counts are satisfiable
        for (uint32_t i = 0; i < VK_DESCRIPTOR_TYPE_RANGE_SIZE; i++) {
            if (ds_data->required_descriptors_by_type[i] > pool_state->availableDescriptorTypeCount[i]) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT,
                                HandleToUint64(pool_state->pool), __LINE__, VALIDATION_ERROR_04c00266, "DS",
                                "Unable to allocate %u descriptors of type %s from pool 0x%" PRIxLEAST64
                                ". This pool only has %d descriptors of this type remaining. %s",
                                ds_data->required_descriptors_by_type[i], string_VkDescriptorType(VkDescriptorType(i)),
                                HandleToUint64(pool_state->pool), pool_state->availableDescriptorTypeCount[i],
                                validation_error_map[VALIDATION_ERROR_04c00266]);
            }
        }
    }

    return skip;
}
// Decrement allocated sets from the pool and insert new sets into set_map
void cvdescriptorset::PerformAllocateDescriptorSets(const VkDescriptorSetAllocateInfo *p_alloc_info,
                                                    const VkDescriptorSet *descriptor_sets,
                                                    const AllocateDescriptorSetsData *ds_data,
                                                    std::unordered_map<VkDescriptorPool, DESCRIPTOR_POOL_STATE *> *pool_map,
                                                    std::unordered_map<VkDescriptorSet, cvdescriptorset::DescriptorSet *> *set_map,
                                                    const layer_data *dev_data) {
    auto pool_state = (*pool_map)[p_alloc_info->descriptorPool];
    // Account for sets and individual descriptors allocated from pool
    pool_state->availableSets -= p_alloc_info->descriptorSetCount;
    for (uint32_t i = 0; i < VK_DESCRIPTOR_TYPE_RANGE_SIZE; i++) {
        pool_state->availableDescriptorTypeCount[i] -= ds_data->required_descriptors_by_type[i];
    }
    // Create tracking object for each descriptor set; insert into global map and the pool's set.
    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        auto new_ds = new cvdescriptorset::DescriptorSet(descriptor_sets[i], p_alloc_info->descriptorPool, ds_data->layout_nodes[i],
                                                         dev_data);

        pool_state->sets.insert(new_ds);
        new_ds->in_use.store(0);
        (*set_map)[descriptor_sets[i]] = new_ds;
    }
}

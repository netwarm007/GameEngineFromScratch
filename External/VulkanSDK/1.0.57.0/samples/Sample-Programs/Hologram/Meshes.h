/*
 * Copyright (C) 2016 Google, Inc.
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

#ifndef MESHES_H
#define MESHES_H

#include <vulkan/vulkan.h>
#include <vector>

class Meshes {
   public:
    Meshes(VkDevice dev, const std::vector<VkMemoryPropertyFlags> &mem_flags);
    ~Meshes();

    const VkPipelineVertexInputStateCreateInfo &vertex_input_state() const { return vertex_input_state_; }
    const VkPipelineInputAssemblyStateCreateInfo &input_assembly_state() const { return input_assembly_state_; }

    enum Type {
        MESH_PYRAMID,
        MESH_ICOSPHERE,
        MESH_TEAPOT,

        MESH_COUNT,
    };

    void cmd_bind_buffers(VkCommandBuffer cmd) const;
    void cmd_draw(VkCommandBuffer cmd, Type type) const;

   private:
    void allocate_resources(VkDeviceSize vb_size, VkDeviceSize ib_size, const std::vector<VkMemoryPropertyFlags> &mem_flags);

    VkDevice dev_;

    VkVertexInputBindingDescription vertex_input_binding_;
    std::vector<VkVertexInputAttributeDescription> vertex_input_attrs_;
    VkPipelineVertexInputStateCreateInfo vertex_input_state_;
    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_;
    VkIndexType index_type_;

    std::vector<VkDrawIndexedIndirectCommand> draw_commands_;

    VkBuffer vb_;
    VkBuffer ib_;
    VkDeviceMemory mem_;
    VkDeviceSize ib_mem_offset_;
};

#endif  // MESHES_H

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

#include <cassert>
#include <cmath>
#include <cstring>
#include <array>
#include <unordered_map>

#include "Helpers.h"
#include "Meshes.h"

namespace {

class Mesh {
   public:
    struct Position {
        float x;
        float y;
        float z;
    };

    struct Normal {
        float x;
        float y;
        float z;
    };

    struct Face {
        int v0;
        int v1;
        int v2;
    };

    static uint32_t vertex_stride() {
        // Position + Normal
        const int comp_count = 6;

        return sizeof(float) * comp_count;
    }

    static VkVertexInputBindingDescription vertex_input_binding() {
        VkVertexInputBindingDescription vi_binding = {};
        vi_binding.binding = 0;
        vi_binding.stride = vertex_stride();
        vi_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return vi_binding;
    }

    static std::vector<VkVertexInputAttributeDescription> vertex_input_attributes() {
        std::vector<VkVertexInputAttributeDescription> vi_attrs(2);
        // Position
        vi_attrs[0].location = 0;
        vi_attrs[0].binding = 0;
        vi_attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vi_attrs[0].offset = 0;
        // Normal
        vi_attrs[1].location = 1;
        vi_attrs[1].binding = 0;
        vi_attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vi_attrs[1].offset = sizeof(float) * 3;

        return vi_attrs;
    }

    static VkIndexType index_type() { return VK_INDEX_TYPE_UINT32; }

    static VkPipelineInputAssemblyStateCreateInfo input_assembly_state() {
        VkPipelineInputAssemblyStateCreateInfo ia_info = {};
        ia_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        ia_info.primitiveRestartEnable = false;
        return ia_info;
    }

    void build(const std::vector<std::array<float, 6>> &vertices, const std::vector<std::array<int, 3>> &faces) {
        positions_.reserve(vertices.size());
        normals_.reserve(vertices.size());
        for (const auto &v : vertices) {
            positions_.emplace_back(Position{v[0], v[1], v[2]});
            normals_.emplace_back(Normal{v[3], v[4], v[5]});
        }

        faces_.reserve(faces.size());
        for (const auto &f : faces) faces_.emplace_back(Face{f[0], f[1], f[2]});
    }

    uint32_t vertex_count() const { return static_cast<uint32_t>(positions_.size()); }

    VkDeviceSize vertex_buffer_size() const { return vertex_stride() * vertex_count(); }

    void vertex_buffer_write(void *data) const {
        float *dst = reinterpret_cast<float *>(data);
        for (size_t i = 0; i < positions_.size(); i++) {
            const Position &pos = positions_[i];
            const Normal &normal = normals_[i];
            dst[0] = pos.x;
            dst[1] = pos.y;
            dst[2] = pos.z;
            dst[3] = normal.x;
            dst[4] = normal.y;
            dst[5] = normal.z;
            dst += 6;
        }
    }

    uint32_t index_count() const { return static_cast<uint32_t>(faces_.size() * 3); }

    VkDeviceSize index_buffer_size() const { return sizeof(uint32_t) * index_count(); }

    void index_buffer_write(void *data) const {
        uint32_t *dst = reinterpret_cast<uint32_t *>(data);
        for (const auto &face : faces_) {
            dst[0] = face.v0;
            dst[1] = face.v1;
            dst[2] = face.v2;
            dst += 3;
        }
    }

    std::vector<Position> positions_;
    std::vector<Normal> normals_;
    std::vector<Face> faces_;
};

class BuildPyramid {
   public:
    BuildPyramid(Mesh &mesh) {
        const std::vector<std::array<float, 6>> vertices = {
            //      position                normal
            {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f},     {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f},
            {1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f},
            {-1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f},
        };

        const std::vector<std::array<int, 3>> faces = {
            {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}, {1, 4, 3}, {1, 3, 2},
        };

        mesh.build(vertices, faces);
    }
};

class BuildIcosphere {
   public:
    BuildIcosphere(Mesh &mesh) : mesh_(mesh), radius_(1.0f) {
        const int tessellate_level = 2;

        build_icosahedron();
        for (int i = 0; i < tessellate_level; i++) tessellate();
    }

   private:
    void build_icosahedron() {
        // https://en.wikipedia.org/wiki/Regular_icosahedron
        const float l1 = std::sqrt(2.0f / (5.0f + std::sqrt(5.0f))) * radius_;
        const float l2 = std::sqrt(2.0f / (5.0f - std::sqrt(5.0f))) * radius_;
        // vertices are from three golden rectangles
        const std::vector<std::array<float, 6>> icosahedron_vertices = {
            //   position           normal
            {
                -l1, -l2, 0.0f, -l1, -l2, 0.0f,
            },
            {
                l1, -l2, 0.0f, l1, -l2, 0.0f,
            },
            {
                l1, l2, 0.0f, l1, l2, 0.0f,
            },
            {
                -l1, l2, 0.0f, -l1, l2, 0.0f,
            },

            {
                -l2, 0.0f, -l1, -l2, 0.0f, -l1,
            },
            {
                l2, 0.0f, -l1, l2, 0.0f, -l1,
            },
            {
                l2, 0.0f, l1, l2, 0.0f, l1,
            },
            {
                -l2, 0.0f, l1, -l2, 0.0f, l1,
            },

            {
                0.0f, -l1, -l2, 0.0f, -l1, -l2,
            },
            {
                0.0f, l1, -l2, 0.0f, l1, -l2,
            },
            {
                0.0f, l1, l2, 0.0f, l1, l2,
            },
            {
                0.0f, -l1, l2, 0.0f, -l1, l2,
            },
        };
        const std::vector<std::array<int, 3>> icosahedron_faces = {
            // triangles sharing vertex 0
            {0, 1, 11},
            {0, 11, 7},
            {0, 7, 4},
            {0, 4, 8},
            {0, 8, 1},
            // adjacent triangles
            {11, 1, 6},
            {7, 11, 10},
            {4, 7, 3},
            {8, 4, 9},
            {1, 8, 5},
            // triangles sharing vertex 2
            {2, 3, 10},
            {2, 10, 6},
            {2, 6, 5},
            {2, 5, 9},
            {2, 9, 3},
            // adjacent triangles
            {10, 3, 7},
            {6, 10, 11},
            {5, 6, 1},
            {9, 5, 8},
            {3, 9, 4},
        };

        mesh_.build(icosahedron_vertices, icosahedron_faces);
    }

    void tessellate() {
        size_t middle_point_count = mesh_.faces_.size() * 3 / 2;
        size_t final_face_count = mesh_.faces_.size() * 4;

        std::vector<Mesh::Face> faces;
        faces.reserve(final_face_count);

        middle_points_.clear();
        middle_points_.reserve(middle_point_count);

        mesh_.positions_.reserve(mesh_.vertex_count() + middle_point_count);
        mesh_.normals_.reserve(mesh_.vertex_count() + middle_point_count);

        for (const auto &f : mesh_.faces_) {
            int v0 = f.v0;
            int v1 = f.v1;
            int v2 = f.v2;

            int v01 = add_middle_point(v0, v1);
            int v12 = add_middle_point(v1, v2);
            int v20 = add_middle_point(v2, v0);

            faces.emplace_back(Mesh::Face{v0, v01, v20});
            faces.emplace_back(Mesh::Face{v1, v12, v01});
            faces.emplace_back(Mesh::Face{v2, v20, v12});
            faces.emplace_back(Mesh::Face{v01, v12, v20});
        }

        mesh_.faces_.swap(faces);
    }

    int add_middle_point(int a, int b) {
        uint64_t key = (a < b) ? ((uint64_t)a << 32 | b) : ((uint64_t)b << 32 | a);
        auto it = middle_points_.find(key);
        if (it != middle_points_.end()) return it->second;

        const Mesh::Position &pos_a = mesh_.positions_[a];
        const Mesh::Position &pos_b = mesh_.positions_[b];
        Mesh::Position pos_mid = {
            (pos_a.x + pos_b.x) / 2.0f, (pos_a.y + pos_b.y) / 2.0f, (pos_a.z + pos_b.z) / 2.0f,
        };
        float scale = radius_ / std::sqrt(pos_mid.x * pos_mid.x + pos_mid.y * pos_mid.y + pos_mid.z * pos_mid.z);
        pos_mid.x *= scale;
        pos_mid.y *= scale;
        pos_mid.z *= scale;

        Mesh::Normal normal_mid = {pos_mid.x, pos_mid.y, pos_mid.z};
        normal_mid.x /= radius_;
        normal_mid.y /= radius_;
        normal_mid.z /= radius_;

        mesh_.positions_.emplace_back(pos_mid);
        mesh_.normals_.emplace_back(normal_mid);

        int mid = mesh_.vertex_count() - 1;
        middle_points_.emplace(std::make_pair(key, mid));

        return mid;
    }

    Mesh &mesh_;
    const float radius_;
    std::unordered_map<uint64_t, uint32_t> middle_points_;
};

class BuildTeapot {
   public:
    BuildTeapot(Mesh &mesh) {
#include "Meshes.teapot.h"
        const int position_count = sizeof(teapot_positions) / sizeof(teapot_positions[0]);
        const int index_count = sizeof(teapot_indices) / sizeof(teapot_indices[0]);
        assert(position_count % 3 == 0 && index_count % 3 == 0);

        Mesh::Position translate;
        float scale;
        get_transform(teapot_positions, position_count, translate, scale);

        for (int i = 0; i < position_count; i += 3) {
            mesh.positions_.emplace_back(Mesh::Position{
                (teapot_positions[i + 0] + translate.x) * scale, (teapot_positions[i + 1] + translate.y) * scale,
                (teapot_positions[i + 2] + translate.z) * scale,
            });

            mesh.normals_.emplace_back(Mesh::Normal{
                teapot_normals[i + 0], teapot_normals[i + 1], teapot_normals[i + 2],
            });
        }

        for (int i = 0; i < index_count; i += 3) {
            mesh.faces_.emplace_back(Mesh::Face{teapot_indices[i + 0], teapot_indices[i + 1], teapot_indices[i + 2]});
        }
    }

    void get_transform(const float *positions, int position_count, Mesh::Position &translate, float &scale) {
        float min[3] = {
            positions[0], positions[1], positions[2],
        };
        float max[3] = {
            positions[0], positions[1], positions[2],
        };
        for (int i = 3; i < position_count; i += 3) {
            for (int j = 0; j < 3; j++) {
                if (min[j] > positions[i + j]) min[j] = positions[i + j];
                if (max[j] < positions[i + j]) max[j] = positions[i + j];
            }
        }

        translate.x = -(min[0] + max[0]) / 2.0f;
        translate.y = -(min[1] + max[1]) / 2.0f;
        translate.z = -(min[2] + max[2]) / 2.0f;

        float extents[3] = {
            max[0] + translate.x, max[1] + translate.y, max[2] + translate.z,
        };

        float max_extent = extents[0];
        if (max_extent < extents[1]) max_extent = extents[1];
        if (max_extent < extents[2]) max_extent = extents[2];

        scale = 1.0f / max_extent;
    }
};

void build_meshes(std::array<Mesh, Meshes::MESH_COUNT> &meshes) {
    BuildPyramid build_pyramid(meshes[Meshes::MESH_PYRAMID]);
    BuildIcosphere build_icosphere(meshes[Meshes::MESH_ICOSPHERE]);
    BuildTeapot build_teapot(meshes[Meshes::MESH_TEAPOT]);
}

}  // namespace

Meshes::Meshes(VkDevice dev, const std::vector<VkMemoryPropertyFlags> &mem_flags)
    : dev_(dev),
      vertex_input_binding_(Mesh::vertex_input_binding()),
      vertex_input_attrs_(Mesh::vertex_input_attributes()),
      vertex_input_state_(),
      input_assembly_state_(Mesh::input_assembly_state()),
      index_type_(Mesh::index_type()) {
    vertex_input_state_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_state_.vertexBindingDescriptionCount = 1;
    vertex_input_state_.pVertexBindingDescriptions = &vertex_input_binding_;
    vertex_input_state_.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_input_attrs_.size());
    vertex_input_state_.pVertexAttributeDescriptions = vertex_input_attrs_.data();

    std::array<Mesh, MESH_COUNT> meshes;
    build_meshes(meshes);

    draw_commands_.reserve(meshes.size());
    uint32_t first_index = 0;
    int32_t vertex_offset = 0;
    VkDeviceSize vb_size = 0;
    VkDeviceSize ib_size = 0;
    for (const auto &mesh : meshes) {
        VkDrawIndexedIndirectCommand draw = {};
        draw.indexCount = mesh.index_count();
        draw.instanceCount = 1;
        draw.firstIndex = first_index;
        draw.vertexOffset = vertex_offset;
        draw.firstInstance = 0;

        draw_commands_.push_back(draw);

        first_index += mesh.index_count();
        vertex_offset += mesh.vertex_count();
        vb_size += mesh.vertex_buffer_size();
        ib_size += mesh.index_buffer_size();
    }

    allocate_resources(vb_size, ib_size, mem_flags);

    uint8_t *vb_data, *ib_data;
    vk::assert_success(vk::MapMemory(dev_, mem_, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&vb_data)));
    ib_data = vb_data + ib_mem_offset_;

    for (const auto &mesh : meshes) {
        mesh.vertex_buffer_write(vb_data);
        mesh.index_buffer_write(ib_data);
        vb_data += mesh.vertex_buffer_size();
        ib_data += mesh.index_buffer_size();
    }

    vk::UnmapMemory(dev_, mem_);
}

Meshes::~Meshes() {
    vk::FreeMemory(dev_, mem_, nullptr);
    vk::DestroyBuffer(dev_, vb_, nullptr);
    vk::DestroyBuffer(dev_, ib_, nullptr);
}

void Meshes::cmd_bind_buffers(VkCommandBuffer cmd) const {
    const VkDeviceSize vb_offset = 0;
    vk::CmdBindVertexBuffers(cmd, 0, 1, &vb_, &vb_offset);

    vk::CmdBindIndexBuffer(cmd, ib_, 0, index_type_);
}

void Meshes::cmd_draw(VkCommandBuffer cmd, Type type) const {
    const auto &draw = draw_commands_[type];
    vk::CmdDrawIndexed(cmd, draw.indexCount, draw.instanceCount, draw.firstIndex, draw.vertexOffset, draw.firstInstance);
}

void Meshes::allocate_resources(VkDeviceSize vb_size, VkDeviceSize ib_size, const std::vector<VkMemoryPropertyFlags> &mem_flags) {
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = vb_size;
    buf_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk::CreateBuffer(dev_, &buf_info, nullptr, &vb_);

    buf_info.size = ib_size;
    buf_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    vk::CreateBuffer(dev_, &buf_info, nullptr, &ib_);

    VkMemoryRequirements vb_mem_reqs, ib_mem_reqs;
    vk::GetBufferMemoryRequirements(dev_, vb_, &vb_mem_reqs);
    vk::GetBufferMemoryRequirements(dev_, ib_, &ib_mem_reqs);

    // indices follow vertices
    ib_mem_offset_ = vb_mem_reqs.size + (ib_mem_reqs.alignment - (vb_mem_reqs.size % ib_mem_reqs.alignment));

    VkMemoryAllocateInfo mem_info = {};
    mem_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_info.allocationSize = ib_mem_offset_ + ib_mem_reqs.size;

    // find any supported and mappable memory type
    uint32_t mem_types = (vb_mem_reqs.memoryTypeBits & ib_mem_reqs.memoryTypeBits);
    for (uint32_t idx = 0; idx < mem_flags.size(); idx++) {
        if ((mem_types & (1 << idx)) && (mem_flags[idx] & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mem_flags[idx] & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            // TODO this may not be reachable
            mem_info.memoryTypeIndex = idx;
            break;
        }
    }

    vk::AllocateMemory(dev_, &mem_info, nullptr, &mem_);

    vk::BindBufferMemory(dev_, vb_, mem_, 0);
    vk::BindBufferMemory(dev_, ib_, mem_, ib_mem_offset_);
}

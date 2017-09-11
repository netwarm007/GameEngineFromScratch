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
 * Author: Chris Forbes <chrisf@ijw.co.nz>
 */

#include <cinttypes>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <SPIRV/spirv.hpp>
#include "vk_loader_platform.h"
#include "vk_enum_string_helper.h"
#include "vk_layer_table.h"
#include "vk_layer_data.h"
#include "vk_layer_extension_utils.h"
#include "vk_layer_utils.h"
#include "core_validation.h"
#include "core_validation_types.h"
#include "shader_validation.h"
#include "spirv-tools/libspirv.h"

enum FORMAT_TYPE {
    FORMAT_TYPE_FLOAT = 1,  // UNORM, SNORM, FLOAT, USCALED, SSCALED, SRGB -- anything we consider float in the shader
    FORMAT_TYPE_SINT = 2,
    FORMAT_TYPE_UINT = 4,
};

typedef std::pair<unsigned, unsigned> location_t;

struct interface_var {
    uint32_t id;
    uint32_t type_id;
    uint32_t offset;
    bool is_patch;
    bool is_block_member;
    bool is_relaxed_precision;
    // TODO: collect the name, too? Isn't required to be present.
};

struct shader_stage_attributes {
    char const *const name;
    bool arrayed_input;
    bool arrayed_output;
};

static shader_stage_attributes shader_stage_attribs[] = {
    {"vertex shader", false, false},  {"tessellation control shader", true, true}, {"tessellation evaluation shader", true, false},
    {"geometry shader", true, false}, {"fragment shader", false, false},
};

// SPIRV utility functions
void shader_module::build_def_index() {
    for (auto insn : *this) {
        switch (insn.opcode()) {
            // Types
            case spv::OpTypeVoid:
            case spv::OpTypeBool:
            case spv::OpTypeInt:
            case spv::OpTypeFloat:
            case spv::OpTypeVector:
            case spv::OpTypeMatrix:
            case spv::OpTypeImage:
            case spv::OpTypeSampler:
            case spv::OpTypeSampledImage:
            case spv::OpTypeArray:
            case spv::OpTypeRuntimeArray:
            case spv::OpTypeStruct:
            case spv::OpTypeOpaque:
            case spv::OpTypePointer:
            case spv::OpTypeFunction:
            case spv::OpTypeEvent:
            case spv::OpTypeDeviceEvent:
            case spv::OpTypeReserveId:
            case spv::OpTypeQueue:
            case spv::OpTypePipe:
                def_index[insn.word(1)] = insn.offset();
                break;

                // Fixed constants
            case spv::OpConstantTrue:
            case spv::OpConstantFalse:
            case spv::OpConstant:
            case spv::OpConstantComposite:
            case spv::OpConstantSampler:
            case spv::OpConstantNull:
                def_index[insn.word(2)] = insn.offset();
                break;

                // Specialization constants
            case spv::OpSpecConstantTrue:
            case spv::OpSpecConstantFalse:
            case spv::OpSpecConstant:
            case spv::OpSpecConstantComposite:
            case spv::OpSpecConstantOp:
                def_index[insn.word(2)] = insn.offset();
                break;

                // Variables
            case spv::OpVariable:
                def_index[insn.word(2)] = insn.offset();
                break;

                // Functions
            case spv::OpFunction:
                def_index[insn.word(2)] = insn.offset();
                break;

            default:
                // We don't care about any other defs for now.
                break;
        }
    }
}

static spirv_inst_iter find_entrypoint(shader_module const *src, char const *name, VkShaderStageFlagBits stageBits) {
    for (auto insn : *src) {
        if (insn.opcode() == spv::OpEntryPoint) {
            auto entrypointName = (char const *)&insn.word(3);
            auto entrypointStageBits = 1u << insn.word(1);

            if (!strcmp(entrypointName, name) && (entrypointStageBits & stageBits)) {
                return insn;
            }
        }
    }

    return src->end();
}

static char const *storage_class_name(unsigned sc) {
    switch (sc) {
        case spv::StorageClassInput:
            return "input";
        case spv::StorageClassOutput:
            return "output";
        case spv::StorageClassUniformConstant:
            return "const uniform";
        case spv::StorageClassUniform:
            return "uniform";
        case spv::StorageClassWorkgroup:
            return "workgroup local";
        case spv::StorageClassCrossWorkgroup:
            return "workgroup global";
        case spv::StorageClassPrivate:
            return "private global";
        case spv::StorageClassFunction:
            return "function";
        case spv::StorageClassGeneric:
            return "generic";
        case spv::StorageClassAtomicCounter:
            return "atomic counter";
        case spv::StorageClassImage:
            return "image";
        case spv::StorageClassPushConstant:
            return "push constant";
        default:
            return "unknown";
    }
}

// Get the value of an integral constant
unsigned get_constant_value(shader_module const *src, unsigned id) {
    auto value = src->get_def(id);
    assert(value != src->end());

    if (value.opcode() != spv::OpConstant) {
        // TODO: Either ensure that the specialization transform is already performed on a module we're
        //       considering here, OR -- specialize on the fly now.
        return 1;
    }

    return value.word(3);
}

static void describe_type_inner(std::ostringstream &ss, shader_module const *src, unsigned type) {
    auto insn = src->get_def(type);
    assert(insn != src->end());

    switch (insn.opcode()) {
        case spv::OpTypeBool:
            ss << "bool";
            break;
        case spv::OpTypeInt:
            ss << (insn.word(3) ? 's' : 'u') << "int" << insn.word(2);
            break;
        case spv::OpTypeFloat:
            ss << "float" << insn.word(2);
            break;
        case spv::OpTypeVector:
            ss << "vec" << insn.word(3) << " of ";
            describe_type_inner(ss, src, insn.word(2));
            break;
        case spv::OpTypeMatrix:
            ss << "mat" << insn.word(3) << " of ";
            describe_type_inner(ss, src, insn.word(2));
            break;
        case spv::OpTypeArray:
            ss << "arr[" << get_constant_value(src, insn.word(3)) << "] of ";
            describe_type_inner(ss, src, insn.word(2));
            break;
        case spv::OpTypePointer:
            ss << "ptr to " << storage_class_name(insn.word(2)) << " ";
            describe_type_inner(ss, src, insn.word(3));
            break;
        case spv::OpTypeStruct: {
            ss << "struct of (";
            for (unsigned i = 2; i < insn.len(); i++) {
                describe_type_inner(ss, src, insn.word(i));
                if (i == insn.len() - 1) {
                    ss << ")";
                } else {
                    ss << ", ";
                }
            }
            break;
        }
        case spv::OpTypeSampler:
            ss << "sampler";
            break;
        case spv::OpTypeSampledImage:
            ss << "sampler+";
            describe_type_inner(ss, src, insn.word(2));
            break;
        case spv::OpTypeImage:
            ss << "image(dim=" << insn.word(3) << ", sampled=" << insn.word(7) << ")";
            break;
        default:
            ss << "oddtype";
            break;
    }
}

static std::string describe_type(shader_module const *src, unsigned type) {
    std::ostringstream ss;
    describe_type_inner(ss, src, type);
    return ss.str();
}

static bool is_narrow_numeric_type(spirv_inst_iter type) {
    if (type.opcode() != spv::OpTypeInt && type.opcode() != spv::OpTypeFloat) return false;
    return type.word(2) < 64;
}

static bool types_match(shader_module const *a, shader_module const *b, unsigned a_type, unsigned b_type, bool a_arrayed,
                        bool b_arrayed, bool relaxed) {
    // Walk two type trees together, and complain about differences
    auto a_insn = a->get_def(a_type);
    auto b_insn = b->get_def(b_type);
    assert(a_insn != a->end());
    assert(b_insn != b->end());

    if (a_arrayed && a_insn.opcode() == spv::OpTypeArray) {
        return types_match(a, b, a_insn.word(2), b_type, false, b_arrayed, relaxed);
    }

    if (b_arrayed && b_insn.opcode() == spv::OpTypeArray) {
        // We probably just found the extra level of arrayness in b_type: compare the type inside it to a_type
        return types_match(a, b, a_type, b_insn.word(2), a_arrayed, false, relaxed);
    }

    if (a_insn.opcode() == spv::OpTypeVector && relaxed && is_narrow_numeric_type(b_insn)) {
        return types_match(a, b, a_insn.word(2), b_type, a_arrayed, b_arrayed, false);
    }

    if (a_insn.opcode() != b_insn.opcode()) {
        return false;
    }

    if (a_insn.opcode() == spv::OpTypePointer) {
        // Match on pointee type. storage class is expected to differ
        return types_match(a, b, a_insn.word(3), b_insn.word(3), a_arrayed, b_arrayed, relaxed);
    }

    if (a_arrayed || b_arrayed) {
        // If we havent resolved array-of-verts by here, we're not going to.
        return false;
    }

    switch (a_insn.opcode()) {
        case spv::OpTypeBool:
            return true;
        case spv::OpTypeInt:
            // Match on width, signedness
            return a_insn.word(2) == b_insn.word(2) && a_insn.word(3) == b_insn.word(3);
        case spv::OpTypeFloat:
            // Match on width
            return a_insn.word(2) == b_insn.word(2);
        case spv::OpTypeVector:
            // Match on element type, count.
            if (!types_match(a, b, a_insn.word(2), b_insn.word(2), a_arrayed, b_arrayed, false)) return false;
            if (relaxed && is_narrow_numeric_type(a->get_def(a_insn.word(2)))) {
                return a_insn.word(3) >= b_insn.word(3);
            } else {
                return a_insn.word(3) == b_insn.word(3);
            }
        case spv::OpTypeMatrix:
            // Match on element type, count.
            return types_match(a, b, a_insn.word(2), b_insn.word(2), a_arrayed, b_arrayed, false) &&
                a_insn.word(3) == b_insn.word(3);
        case spv::OpTypeArray:
            // Match on element type, count. these all have the same layout. we don't get here if b_arrayed. This differs from
            // vector & matrix types in that the array size is the id of a constant instruction, * not a literal within OpTypeArray
            return types_match(a, b, a_insn.word(2), b_insn.word(2), a_arrayed, b_arrayed, false) &&
                get_constant_value(a, a_insn.word(3)) == get_constant_value(b, b_insn.word(3));
        case spv::OpTypeStruct:
            // Match on all element types
        {
            if (a_insn.len() != b_insn.len()) {
                return false;  // Structs cannot match if member counts differ
            }

            for (unsigned i = 2; i < a_insn.len(); i++) {
                if (!types_match(a, b, a_insn.word(i), b_insn.word(i), a_arrayed, b_arrayed, false)) {
                    return false;
                }
            }

            return true;
        }
        default:
            // Remaining types are CLisms, or may not appear in the interfaces we are interested in. Just claim no match.
            return false;
    }
}

static unsigned value_or_default(std::unordered_map<unsigned, unsigned> const &map, unsigned id, unsigned def) {
    auto it = map.find(id);
    if (it == map.end())
        return def;
    else
        return it->second;
}

static unsigned get_locations_consumed_by_type(shader_module const *src, unsigned type, bool strip_array_level) {
    auto insn = src->get_def(type);
    assert(insn != src->end());

    switch (insn.opcode()) {
        case spv::OpTypePointer:
            // See through the ptr -- this is only ever at the toplevel for graphics shaders we're never actually passing
            // pointers around.
            return get_locations_consumed_by_type(src, insn.word(3), strip_array_level);
        case spv::OpTypeArray:
            if (strip_array_level) {
                return get_locations_consumed_by_type(src, insn.word(2), false);
            } else {
                return get_constant_value(src, insn.word(3)) * get_locations_consumed_by_type(src, insn.word(2), false);
            }
        case spv::OpTypeMatrix:
            // Num locations is the dimension * element size
            return insn.word(3) * get_locations_consumed_by_type(src, insn.word(2), false);
        case spv::OpTypeVector: {
            auto scalar_type = src->get_def(insn.word(2));
            auto bit_width =
                (scalar_type.opcode() == spv::OpTypeInt || scalar_type.opcode() == spv::OpTypeFloat) ? scalar_type.word(2) : 32;

            // Locations are 128-bit wide; 3- and 4-component vectors of 64 bit types require two.
            return (bit_width * insn.word(3) + 127) / 128;
        }
        default:
            // Everything else is just 1.
            return 1;

            // TODO: extend to handle 64bit scalar types, whose vectors may need multiple locations.
    }
}

static unsigned get_locations_consumed_by_format(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R64G64B64A64_SFLOAT:
        case VK_FORMAT_R64G64B64A64_SINT:
        case VK_FORMAT_R64G64B64A64_UINT:
        case VK_FORMAT_R64G64B64_SFLOAT:
        case VK_FORMAT_R64G64B64_SINT:
        case VK_FORMAT_R64G64B64_UINT:
            return 2;
        default:
            return 1;
    }
}

static unsigned get_format_type(VkFormat fmt) {
    if (FormatIsSInt(fmt))
        return FORMAT_TYPE_SINT;
    if (FormatIsUInt(fmt))
        return FORMAT_TYPE_UINT;
    if (FormatIsDepthAndStencil(fmt))
        return FORMAT_TYPE_FLOAT | FORMAT_TYPE_UINT;
    if (fmt == VK_FORMAT_UNDEFINED)
        return 0;
    // everything else -- UNORM/SNORM/FLOAT/USCALED/SSCALED is all float in the shader.
    return FORMAT_TYPE_FLOAT;
}

// characterizes a SPIR-V type appearing in an interface to a FF stage, for comparison to a VkFormat's characterization above.
static unsigned get_fundamental_type(shader_module const *src, unsigned type) {
    auto insn = src->get_def(type);
    assert(insn != src->end());

    switch (insn.opcode()) {
        case spv::OpTypeInt:
            return insn.word(3) ? FORMAT_TYPE_SINT : FORMAT_TYPE_UINT;
        case spv::OpTypeFloat:
            return FORMAT_TYPE_FLOAT;
        case spv::OpTypeVector:
            return get_fundamental_type(src, insn.word(2));
        case spv::OpTypeMatrix:
            return get_fundamental_type(src, insn.word(2));
        case spv::OpTypeArray:
            return get_fundamental_type(src, insn.word(2));
        case spv::OpTypePointer:
            return get_fundamental_type(src, insn.word(3));
        case spv::OpTypeImage:
            return get_fundamental_type(src, insn.word(2));

        default:
            return 0;
    }
}

static uint32_t get_shader_stage_id(VkShaderStageFlagBits stage) {
    uint32_t bit_pos = uint32_t(u_ffs(stage));
    return bit_pos - 1;
}

static spirv_inst_iter get_struct_type(shader_module const *src, spirv_inst_iter def, bool is_array_of_verts) {
    while (true) {
        if (def.opcode() == spv::OpTypePointer) {
            def = src->get_def(def.word(3));
        } else if (def.opcode() == spv::OpTypeArray && is_array_of_verts) {
            def = src->get_def(def.word(2));
            is_array_of_verts = false;
        } else if (def.opcode() == spv::OpTypeStruct) {
            return def;
        } else {
            return src->end();
        }
    }
}

static bool collect_interface_block_members(shader_module const *src, std::map<location_t, interface_var> *out,
                                            std::unordered_map<unsigned, unsigned> const &blocks, bool is_array_of_verts,
                                            uint32_t id, uint32_t type_id, bool is_patch, int /*first_location*/) {
    // Walk down the type_id presented, trying to determine whether it's actually an interface block.
    auto type = get_struct_type(src, src->get_def(type_id), is_array_of_verts && !is_patch);
    if (type == src->end() || blocks.find(type.word(1)) == blocks.end()) {
        // This isn't an interface block.
        return false;
    }

    std::unordered_map<unsigned, unsigned> member_components;
    std::unordered_map<unsigned, unsigned> member_relaxed_precision;
    std::unordered_map<unsigned, unsigned> member_patch;

    // Walk all the OpMemberDecorate for type's result id -- first pass, collect components.
    for (auto insn : *src) {
        if (insn.opcode() == spv::OpMemberDecorate && insn.word(1) == type.word(1)) {
            unsigned member_index = insn.word(2);

            if (insn.word(3) == spv::DecorationComponent) {
                unsigned component = insn.word(4);
                member_components[member_index] = component;
            }

            if (insn.word(3) == spv::DecorationRelaxedPrecision) {
                member_relaxed_precision[member_index] = 1;
            }

            if (insn.word(3) == spv::DecorationPatch) {
                member_patch[member_index] = 1;
            }
        }
    }

    // TODO: correctly handle location assignment from outside

    // Second pass -- produce the output, from Location decorations
    for (auto insn : *src) {
        if (insn.opcode() == spv::OpMemberDecorate && insn.word(1) == type.word(1)) {
            unsigned member_index = insn.word(2);
            unsigned member_type_id = type.word(2 + member_index);

            if (insn.word(3) == spv::DecorationLocation) {
                unsigned location = insn.word(4);
                unsigned num_locations = get_locations_consumed_by_type(src, member_type_id, false);
                auto component_it = member_components.find(member_index);
                unsigned component = component_it == member_components.end() ? 0 : component_it->second;
                bool is_relaxed_precision = member_relaxed_precision.find(member_index) != member_relaxed_precision.end();
                bool member_is_patch = is_patch || member_patch.count(member_index)>0;

                for (unsigned int offset = 0; offset < num_locations; offset++) {
                    interface_var v = {};
                    v.id = id;
                    // TODO: member index in interface_var too?
                    v.type_id = member_type_id;
                    v.offset = offset;
                    v.is_patch = member_is_patch;
                    v.is_block_member = true;
                    v.is_relaxed_precision = is_relaxed_precision;
                    (*out)[std::make_pair(location + offset, component)] = v;
                }
            }
        }
    }

    return true;
}

static std::map<location_t, interface_var> collect_interface_by_location(shader_module const *src, spirv_inst_iter entrypoint,
                                                                         spv::StorageClass sinterface, bool is_array_of_verts) {
    std::unordered_map<unsigned, unsigned> var_locations;
    std::unordered_map<unsigned, unsigned> var_builtins;
    std::unordered_map<unsigned, unsigned> var_components;
    std::unordered_map<unsigned, unsigned> blocks;
    std::unordered_map<unsigned, unsigned> var_patch;
    std::unordered_map<unsigned, unsigned> var_relaxed_precision;

    for (auto insn : *src) {
        // We consider two interface models: SSO rendezvous-by-location, and builtins. Complain about anything that
        // fits neither model.
        if (insn.opcode() == spv::OpDecorate) {
            if (insn.word(2) == spv::DecorationLocation) {
                var_locations[insn.word(1)] = insn.word(3);
            }

            if (insn.word(2) == spv::DecorationBuiltIn) {
                var_builtins[insn.word(1)] = insn.word(3);
            }

            if (insn.word(2) == spv::DecorationComponent) {
                var_components[insn.word(1)] = insn.word(3);
            }

            if (insn.word(2) == spv::DecorationBlock) {
                blocks[insn.word(1)] = 1;
            }

            if (insn.word(2) == spv::DecorationPatch) {
                var_patch[insn.word(1)] = 1;
            }

            if (insn.word(2) == spv::DecorationRelaxedPrecision) {
                var_relaxed_precision[insn.word(1)] = 1;
            }
        }
    }

    // TODO: handle grouped decorations
    // TODO: handle index=1 dual source outputs from FS -- two vars will have the same location, and we DON'T want to clobber.

    // Find the end of the entrypoint's name string. additional zero bytes follow the actual null terminator, to fill out the
    // rest of the word - so we only need to look at the last byte in the word to determine which word contains the terminator.
    uint32_t word = 3;
    while (entrypoint.word(word) & 0xff000000u) {
        ++word;
    }
    ++word;

    std::map<location_t, interface_var> out;

    for (; word < entrypoint.len(); word++) {
        auto insn = src->get_def(entrypoint.word(word));
        assert(insn != src->end());
        assert(insn.opcode() == spv::OpVariable);

        if (insn.word(3) == static_cast<uint32_t>(sinterface)) {
            unsigned id = insn.word(2);
            unsigned type = insn.word(1);

            int location = value_or_default(var_locations, id, -1);
            int builtin = value_or_default(var_builtins, id, -1);
            unsigned component = value_or_default(var_components, id, 0);  // Unspecified is OK, is 0
            bool is_patch = var_patch.find(id) != var_patch.end();
            bool is_relaxed_precision = var_relaxed_precision.find(id) != var_relaxed_precision.end();

            if (builtin != -1) continue;
            else if (!collect_interface_block_members(src, &out, blocks, is_array_of_verts, id, type, is_patch, location)) {
                // A user-defined interface variable, with a location. Where a variable occupied multiple locations, emit
                // one result for each.
                unsigned num_locations = get_locations_consumed_by_type(src, type, is_array_of_verts && !is_patch);
                for (unsigned int offset = 0; offset < num_locations; offset++) {
                    interface_var v = {};
                    v.id = id;
                    v.type_id = type;
                    v.offset = offset;
                    v.is_patch = is_patch;
                    v.is_relaxed_precision = is_relaxed_precision;
                    out[std::make_pair(location + offset, component)] = v;
                }
            }
        }
    }

    return out;
}

static std::vector<std::pair<uint32_t, interface_var>> collect_interface_by_input_attachment_index(
    shader_module const *src, std::unordered_set<uint32_t> const &accessible_ids) {
    std::vector<std::pair<uint32_t, interface_var>> out;

    for (auto insn : *src) {
        if (insn.opcode() == spv::OpDecorate) {
            if (insn.word(2) == spv::DecorationInputAttachmentIndex) {
                auto attachment_index = insn.word(3);
                auto id = insn.word(1);

                if (accessible_ids.count(id)) {
                    auto def = src->get_def(id);
                    assert(def != src->end());

                    if (def.opcode() == spv::OpVariable && insn.word(3) == spv::StorageClassUniformConstant) {
                        auto num_locations = get_locations_consumed_by_type(src, def.word(1), false);
                        for (unsigned int offset = 0; offset < num_locations; offset++) {
                            interface_var v = {};
                            v.id = id;
                            v.type_id = def.word(1);
                            v.offset = offset;
                            out.emplace_back(attachment_index + offset, v);
                        }
                    }
                }
            }
        }
    }

    return out;
}

static std::vector<std::pair<descriptor_slot_t, interface_var>> collect_interface_by_descriptor_slot(
    debug_report_data const *report_data, shader_module const *src, std::unordered_set<uint32_t> const &accessible_ids) {
    std::unordered_map<unsigned, unsigned> var_sets;
    std::unordered_map<unsigned, unsigned> var_bindings;

    for (auto insn : *src) {
        // All variables in the Uniform or UniformConstant storage classes are required to be decorated with both
        // DecorationDescriptorSet and DecorationBinding.
        if (insn.opcode() == spv::OpDecorate) {
            if (insn.word(2) == spv::DecorationDescriptorSet) {
                var_sets[insn.word(1)] = insn.word(3);
            }

            if (insn.word(2) == spv::DecorationBinding) {
                var_bindings[insn.word(1)] = insn.word(3);
            }
        }
    }

    std::vector<std::pair<descriptor_slot_t, interface_var>> out;

    for (auto id : accessible_ids) {
        auto insn = src->get_def(id);
        assert(insn != src->end());

        if (insn.opcode() == spv::OpVariable &&
            (insn.word(3) == spv::StorageClassUniform || insn.word(3) == spv::StorageClassUniformConstant)) {
            unsigned set = value_or_default(var_sets, insn.word(2), 0);
            unsigned binding = value_or_default(var_bindings, insn.word(2), 0);

            interface_var v = {};
            v.id = insn.word(2);
            v.type_id = insn.word(1);
            out.emplace_back(std::make_pair(set, binding), v);
        }
    }

    return out;
}



static bool validate_vi_consistency(debug_report_data const *report_data, VkPipelineVertexInputStateCreateInfo const *vi) {
    // Walk the binding descriptions, which describe the step rate and stride of each vertex buffer.  Each binding should
    // be specified only once.
    std::unordered_map<uint32_t, VkVertexInputBindingDescription const *> bindings;
    bool skip = false;

    for (unsigned i = 0; i < vi->vertexBindingDescriptionCount; i++) {
        auto desc = &vi->pVertexBindingDescriptions[i];
        auto &binding = bindings[desc->binding];
        if (binding) {
            // TODO: VALIDATION_ERROR_096005cc perhaps?
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_INCONSISTENT_VI, "SC", "Duplicate vertex input binding descriptions for binding %d",
                            desc->binding);
        } else {
            binding = desc;
        }
    }

    return skip;
}

static bool validate_vi_against_vs_inputs(debug_report_data const *report_data, VkPipelineVertexInputStateCreateInfo const *vi,
                                          shader_module const *vs, spirv_inst_iter entrypoint) {
    bool skip = false;

    auto inputs = collect_interface_by_location(vs, entrypoint, spv::StorageClassInput, false);

    // Build index by location
    std::map<uint32_t, VkVertexInputAttributeDescription const *> attribs;
    if (vi) {
        for (unsigned i = 0; i < vi->vertexAttributeDescriptionCount; i++) {
            auto num_locations = get_locations_consumed_by_format(vi->pVertexAttributeDescriptions[i].format);
            for (auto j = 0u; j < num_locations; j++) {
                attribs[vi->pVertexAttributeDescriptions[i].location + j] = &vi->pVertexAttributeDescriptions[i];
            }
        }
    }

    auto it_a = attribs.begin();
    auto it_b = inputs.begin();
    bool used = false;

    while ((attribs.size() > 0 && it_a != attribs.end()) || (inputs.size() > 0 && it_b != inputs.end())) {
        bool a_at_end = attribs.size() == 0 || it_a == attribs.end();
        bool b_at_end = inputs.size() == 0 || it_b == inputs.end();
        auto a_first = a_at_end ? 0 : it_a->first;
        auto b_first = b_at_end ? 0 : it_b->first.first;
        if (!a_at_end && (b_at_end || a_first < b_first)) {
            if (!used && log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,
                                 0, __LINE__, SHADER_CHECKER_OUTPUT_NOT_CONSUMED, "SC",
                                 "Vertex attribute at location %d not consumed by vertex shader", a_first)) {
                skip = true;
            }
            used = false;
            it_a++;
        } else if (!b_at_end && (a_at_end || b_first < a_first)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT, 0, __LINE__,
                            SHADER_CHECKER_INPUT_NOT_PRODUCED, "SC", "Vertex shader consumes input at location %d but not provided",
                            b_first);
            it_b++;
        } else {
            unsigned attrib_type = get_format_type(it_a->second->format);
            unsigned input_type = get_fundamental_type(vs, it_b->second.type_id);

            // Type checking
            if (!(attrib_type & input_type)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                SHADER_CHECKER_INTERFACE_TYPE_MISMATCH, "SC",
                                "Attribute type of `%s` at location %d does not match vertex shader input type of `%s`",
                                string_VkFormat(it_a->second->format), a_first, describe_type(vs, it_b->second.type_id).c_str());
            }

            // OK!
            used = true;
            it_b++;
        }
    }

    return skip;
}

static bool validate_fs_outputs_against_render_pass(debug_report_data const *report_data, shader_module const *fs,
                                                    spirv_inst_iter entrypoint, PIPELINE_STATE const *pipeline,
                                                    uint32_t subpass_index) {
    auto rpci = pipeline->render_pass_ci.ptr();

    std::map<uint32_t, VkFormat> color_attachments;
    auto subpass = rpci->pSubpasses[subpass_index];
    for (auto i = 0u; i < subpass.colorAttachmentCount; ++i) {
        uint32_t attachment = subpass.pColorAttachments[i].attachment;
        if (attachment == VK_ATTACHMENT_UNUSED) continue;
        if (rpci->pAttachments[attachment].format != VK_FORMAT_UNDEFINED) {
            color_attachments[i] = rpci->pAttachments[attachment].format;
        }
    }

    bool skip = false;

    // TODO: dual source blend index (spv::DecIndex, zero if not provided)

    auto outputs = collect_interface_by_location(fs, entrypoint, spv::StorageClassOutput, false);

    auto it_a = outputs.begin();
    auto it_b = color_attachments.begin();

    // Walk attachment list and outputs together

    while ((outputs.size() > 0 && it_a != outputs.end()) || (color_attachments.size() > 0 && it_b != color_attachments.end())) {
        bool a_at_end = outputs.size() == 0 || it_a == outputs.end();
        bool b_at_end = color_attachments.size() == 0 || it_b == color_attachments.end();

        if (!a_at_end && (b_at_end || it_a->first.first < it_b->first)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_OUTPUT_NOT_CONSUMED, "SC",
                            "fragment shader writes to output location %d with no matching attachment", it_a->first.first);
            it_a++;
        } else if (!b_at_end && (a_at_end || it_a->first.first > it_b->first)) {
            // Only complain if there are unmasked channels for this attachment. If the writemask is 0, it's acceptable for the
            // shader to not produce a matching output.
            if (pipeline->attachments[it_b->first].colorWriteMask != 0) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_INPUT_NOT_PRODUCED, "SC", "Attachment %d not written by fragment shader", it_b->first);
            }
            it_b++;
        } else {
            unsigned output_type = get_fundamental_type(fs, it_a->second.type_id);
            unsigned att_type = get_format_type(it_b->second);

            // Type checking
            if (!(output_type & att_type)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                SHADER_CHECKER_INTERFACE_TYPE_MISMATCH, "SC",
                                "Attachment %d of type `%s` does not match fragment shader output type of `%s`", it_b->first,
                                string_VkFormat(it_b->second), describe_type(fs, it_a->second.type_id).c_str());
            }

            // OK!
            it_a++;
            it_b++;
        }
    }

    return skip;
}

// For some analyses, we need to know about all ids referenced by the static call tree of a particular entrypoint. This is
// important for identifying the set of shader resources actually used by an entrypoint, for example.
// Note: we only explore parts of the image which might actually contain ids we care about for the above analyses.
//  - NOT the shader input/output interfaces.
//
// TODO: The set of interesting opcodes here was determined by eyeballing the SPIRV spec. It might be worth
// converting parts of this to be generated from the machine-readable spec instead.
static std::unordered_set<uint32_t> mark_accessible_ids(shader_module const *src, spirv_inst_iter entrypoint) {
    std::unordered_set<uint32_t> ids;
    std::unordered_set<uint32_t> worklist;
    worklist.insert(entrypoint.word(2));

    while (!worklist.empty()) {
        auto id_iter = worklist.begin();
        auto id = *id_iter;
        worklist.erase(id_iter);

        auto insn = src->get_def(id);
        if (insn == src->end()) {
            // ID is something we didn't collect in build_def_index. that's OK -- we'll stumble across all kinds of things here
            // that we may not care about.
            continue;
        }

        // Try to add to the output set
        if (!ids.insert(id).second) {
            continue;  // If we already saw this id, we don't want to walk it again.
        }

        switch (insn.opcode()) {
            case spv::OpFunction:
                // Scan whole body of the function, enlisting anything interesting
                while (++insn, insn.opcode() != spv::OpFunctionEnd) {
                    switch (insn.opcode()) {
                        case spv::OpLoad:
                        case spv::OpAtomicLoad:
                        case spv::OpAtomicExchange:
                        case spv::OpAtomicCompareExchange:
                        case spv::OpAtomicCompareExchangeWeak:
                        case spv::OpAtomicIIncrement:
                        case spv::OpAtomicIDecrement:
                        case spv::OpAtomicIAdd:
                        case spv::OpAtomicISub:
                        case spv::OpAtomicSMin:
                        case spv::OpAtomicUMin:
                        case spv::OpAtomicSMax:
                        case spv::OpAtomicUMax:
                        case spv::OpAtomicAnd:
                        case spv::OpAtomicOr:
                        case spv::OpAtomicXor:
                            worklist.insert(insn.word(3));  // ptr
                            break;
                        case spv::OpStore:
                        case spv::OpAtomicStore:
                            worklist.insert(insn.word(1));  // ptr
                            break;
                        case spv::OpAccessChain:
                        case spv::OpInBoundsAccessChain:
                            worklist.insert(insn.word(3));  // base ptr
                            break;
                        case spv::OpSampledImage:
                        case spv::OpImageSampleImplicitLod:
                        case spv::OpImageSampleExplicitLod:
                        case spv::OpImageSampleDrefImplicitLod:
                        case spv::OpImageSampleDrefExplicitLod:
                        case spv::OpImageSampleProjImplicitLod:
                        case spv::OpImageSampleProjExplicitLod:
                        case spv::OpImageSampleProjDrefImplicitLod:
                        case spv::OpImageSampleProjDrefExplicitLod:
                        case spv::OpImageFetch:
                        case spv::OpImageGather:
                        case spv::OpImageDrefGather:
                        case spv::OpImageRead:
                        case spv::OpImage:
                        case spv::OpImageQueryFormat:
                        case spv::OpImageQueryOrder:
                        case spv::OpImageQuerySizeLod:
                        case spv::OpImageQuerySize:
                        case spv::OpImageQueryLod:
                        case spv::OpImageQueryLevels:
                        case spv::OpImageQuerySamples:
                        case spv::OpImageSparseSampleImplicitLod:
                        case spv::OpImageSparseSampleExplicitLod:
                        case spv::OpImageSparseSampleDrefImplicitLod:
                        case spv::OpImageSparseSampleDrefExplicitLod:
                        case spv::OpImageSparseSampleProjImplicitLod:
                        case spv::OpImageSparseSampleProjExplicitLod:
                        case spv::OpImageSparseSampleProjDrefImplicitLod:
                        case spv::OpImageSparseSampleProjDrefExplicitLod:
                        case spv::OpImageSparseFetch:
                        case spv::OpImageSparseGather:
                        case spv::OpImageSparseDrefGather:
                        case spv::OpImageTexelPointer:
                            worklist.insert(insn.word(3));  // Image or sampled image
                            break;
                        case spv::OpImageWrite:
                            worklist.insert(insn.word(1));  // Image -- different operand order to above
                            break;
                        case spv::OpFunctionCall:
                            for (uint32_t i = 3; i < insn.len(); i++) {
                                worklist.insert(insn.word(i));  // fn itself, and all args
                            }
                            break;

                        case spv::OpExtInst:
                            for (uint32_t i = 5; i < insn.len(); i++) {
                                worklist.insert(insn.word(i));  // Operands to ext inst
                            }
                            break;
                    }
                }
                break;
        }
    }

    return ids;
}

static bool validate_push_constant_block_against_pipeline(debug_report_data const *report_data,
                                                          std::vector<VkPushConstantRange> const *push_constant_ranges,
                                                          shader_module const *src, spirv_inst_iter type,
                                                          VkShaderStageFlagBits stage) {
    bool skip = false;

    // Strip off ptrs etc
    type = get_struct_type(src, type, false);
    assert(type != src->end());

    // Validate directly off the offsets. this isn't quite correct for arrays and matrices, but is a good first step.
    // TODO: arrays, matrices, weird sizes
    for (auto insn : *src) {
        if (insn.opcode() == spv::OpMemberDecorate && insn.word(1) == type.word(1)) {
            if (insn.word(3) == spv::DecorationOffset) {
                unsigned offset = insn.word(4);
                auto size = 4;  // Bytes; TODO: calculate this based on the type

                bool found_range = false;
                for (auto const &range : *push_constant_ranges) {
                    if (range.offset <= offset && range.offset + range.size >= offset + size) {
                        found_range = true;

                        if ((range.stageFlags & stage) == 0) {
                            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                            __LINE__, SHADER_CHECKER_PUSH_CONSTANT_NOT_ACCESSIBLE_FROM_STAGE, "SC",
                                            "Push constant range covering variable starting at "
                                                "offset %u not accessible from stage %s",
                                            offset, string_VkShaderStageFlagBits(stage));
                        }

                        break;
                    }
                }

                if (!found_range) {
                    skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                                    __LINE__, SHADER_CHECKER_PUSH_CONSTANT_OUT_OF_RANGE, "SC",
                                    "Push constant range covering variable starting at "
                                        "offset %u not declared in layout",
                                    offset);
                }
            }
        }
    }

    return skip;
}

static bool validate_push_constant_usage(debug_report_data const *report_data,
                                         std::vector<VkPushConstantRange> const *push_constant_ranges, shader_module const *src,
                                         std::unordered_set<uint32_t> accessible_ids, VkShaderStageFlagBits stage) {
    bool skip = false;

    for (auto id : accessible_ids) {
        auto def_insn = src->get_def(id);
        if (def_insn.opcode() == spv::OpVariable && def_insn.word(3) == spv::StorageClassPushConstant) {
            skip |= validate_push_constant_block_against_pipeline(report_data, push_constant_ranges, src,
                                                                  src->get_def(def_insn.word(1)), stage);
        }
    }

    return skip;
}

// Validate that data for each specialization entry is fully contained within the buffer.
static bool validate_specialization_offsets(debug_report_data const *report_data, VkPipelineShaderStageCreateInfo const *info) {
    bool skip = false;

    VkSpecializationInfo const *spec = info->pSpecializationInfo;

    if (spec) {
        for (auto i = 0u; i < spec->mapEntryCount; i++) {
            // TODO: This is a good place for VALIDATION_ERROR_1360060a.
            if (spec->pMapEntries[i].offset + spec->pMapEntries[i].size > spec->dataSize) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT, 0, __LINE__,
                                VALIDATION_ERROR_1360060c, "SC",
                                "Specialization entry %u (for constant id %u) references memory outside provided "
                                    "specialization data (bytes %u.." PRINTF_SIZE_T_SPECIFIER "; " PRINTF_SIZE_T_SPECIFIER
                " bytes provided). %s.",
                    i, spec->pMapEntries[i].constantID, spec->pMapEntries[i].offset,
                    spec->pMapEntries[i].offset + spec->pMapEntries[i].size - 1, spec->dataSize,
                    validation_error_map[VALIDATION_ERROR_1360060c]);
            }
        }
    }

    return skip;
}

static bool descriptor_type_match(shader_module const *module, uint32_t type_id, VkDescriptorType descriptor_type,
                                  unsigned &descriptor_count) {
    auto type = module->get_def(type_id);

    descriptor_count = 1;

    // Strip off any array or ptrs. Where we remove array levels, adjust the  descriptor count for each dimension.
    while (type.opcode() == spv::OpTypeArray || type.opcode() == spv::OpTypePointer) {
        if (type.opcode() == spv::OpTypeArray) {
            descriptor_count *= get_constant_value(module, type.word(3));
            type = module->get_def(type.word(2));
        } else {
            type = module->get_def(type.word(3));
        }
    }

    switch (type.opcode()) {
        case spv::OpTypeStruct: {
            for (auto insn : *module) {
                if (insn.opcode() == spv::OpDecorate && insn.word(1) == type.word(1)) {
                    if (insn.word(2) == spv::DecorationBlock) {
                        return descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                            descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                    } else if (insn.word(2) == spv::DecorationBufferBlock) {
                        return descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                            descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
                    }
                }
            }

            // Invalid
            return false;
        }

        case spv::OpTypeSampler:
            return descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER || descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        case spv::OpTypeSampledImage:
            if (descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
                // Slight relaxation for some GLSL historical madness: samplerBuffer doesn't really have a sampler, and a texel
                // buffer descriptor doesn't really provide one. Allow this slight mismatch.
                auto image_type = module->get_def(type.word(2));
                auto dim = image_type.word(3);
                auto sampled = image_type.word(7);
                return dim == spv::DimBuffer && sampled == 1;
            }
            return descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        case spv::OpTypeImage: {
            // Many descriptor types backing image types-- depends on dimension and whether the image will be used with a sampler.
            // SPIRV for Vulkan requires that sampled be 1 or 2 -- leaving the decision to runtime is unacceptable.
            auto dim = type.word(3);
            auto sampled = type.word(7);

            if (dim == spv::DimSubpassData) {
                return descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            } else if (dim == spv::DimBuffer) {
                if (sampled == 1) {
                    return descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                } else {
                    return descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                }
            } else if (sampled == 1) {
                return descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
                    descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            } else {
                return descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            }
        }

            // We shouldn't really see any other junk types -- but if we do, they're a mismatch.
        default:
            return false;  // Mismatch
    }
}

static bool require_feature(debug_report_data const *report_data, VkBool32 feature, char const *feature_name) {
    if (!feature) {
        if (log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    SHADER_CHECKER_FEATURE_NOT_ENABLED, "SC",
                    "Shader requires VkPhysicalDeviceFeatures::%s but is not "
                        "enabled on the device",
                    feature_name)) {
            return true;
        }
    }

    return false;
}

static bool require_extension(debug_report_data const *report_data, bool extension, char const *extension_name) {
    if (!extension) {
        if (log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    SHADER_CHECKER_FEATURE_NOT_ENABLED, "SC",
                    "Shader requires extension %s but is not "
                        "enabled on the device",
                    extension_name)) {
            return true;
        }
    }

    return false;
}

static bool validate_shader_capabilities(layer_data *dev_data, shader_module const *src) {
    bool skip = false;

    auto report_data = GetReportData(dev_data);
    auto const & enabledFeatures = GetEnabledFeatures(dev_data);
    auto const & extensions = GetEnabledExtensions(dev_data);

    struct CapabilityInfo {
        char const *name;
        VkBool32 const VkPhysicalDeviceFeatures::*feature;
        bool const DeviceExtensions::*extension;
    };

    using F = VkPhysicalDeviceFeatures;
    using E = DeviceExtensions;

    // clang-format off
    static const std::unordered_map<uint32_t, CapabilityInfo> capabilities = {
        // Capabilities always supported by a Vulkan 1.0 implementation -- no
        // feature bits.
        {spv::CapabilityMatrix, {nullptr}},
        {spv::CapabilityShader, {nullptr}},
        {spv::CapabilityInputAttachment, {nullptr}},
        {spv::CapabilitySampled1D, {nullptr}},
        {spv::CapabilityImage1D, {nullptr}},
        {spv::CapabilitySampledBuffer, {nullptr}},
        {spv::CapabilityImageQuery, {nullptr}},
        {spv::CapabilityDerivativeControl, {nullptr}},

        // Capabilities that are optionally supported, but require a feature to
        // be enabled on the device
        {spv::CapabilityGeometry, {"geometryShader", &F::geometryShader}},
        {spv::CapabilityTessellation, {"tessellationShader", &F::tessellationShader}},
        {spv::CapabilityFloat64, {"shaderFloat64", &F::shaderFloat64}},
        {spv::CapabilityInt64, {"shaderInt64", &F::shaderInt64}},
        {spv::CapabilityTessellationPointSize, {"shaderTessellationAndGeometryPointSize", &F::shaderTessellationAndGeometryPointSize}},
        {spv::CapabilityGeometryPointSize, {"shaderTessellationAndGeometryPointSize", &F::shaderTessellationAndGeometryPointSize}},
        {spv::CapabilityImageGatherExtended, {"shaderImageGatherExtended", &F::shaderImageGatherExtended}},
        {spv::CapabilityStorageImageMultisample, {"shaderStorageImageMultisample", &F::shaderStorageImageMultisample}},
        {spv::CapabilityUniformBufferArrayDynamicIndexing, {"shaderUniformBufferArrayDynamicIndexing", &F::shaderUniformBufferArrayDynamicIndexing}},
        {spv::CapabilitySampledImageArrayDynamicIndexing, {"shaderSampledImageArrayDynamicIndexing", &F::shaderSampledImageArrayDynamicIndexing}},
        {spv::CapabilityStorageBufferArrayDynamicIndexing, {"shaderStorageBufferArrayDynamicIndexing", &F::shaderStorageBufferArrayDynamicIndexing}},
        {spv::CapabilityStorageImageArrayDynamicIndexing, {"shaderStorageImageArrayDynamicIndexing", &F::shaderStorageBufferArrayDynamicIndexing}},
        {spv::CapabilityClipDistance, {"shaderClipDistance", &F::shaderClipDistance}},
        {spv::CapabilityCullDistance, {"shaderCullDistance", &F::shaderCullDistance}},
        {spv::CapabilityImageCubeArray, {"imageCubeArray", &F::imageCubeArray}},
        {spv::CapabilitySampleRateShading, {"sampleRateShading", &F::sampleRateShading}},
        {spv::CapabilitySparseResidency, {"shaderResourceResidency", &F::shaderResourceResidency}},
        {spv::CapabilityMinLod, {"shaderResourceMinLod", &F::shaderResourceMinLod}},
        {spv::CapabilitySampledCubeArray, {"imageCubeArray", &F::imageCubeArray}},
        {spv::CapabilityImageMSArray, {"shaderStorageImageMultisample", &F::shaderStorageImageMultisample}},
        {spv::CapabilityStorageImageExtendedFormats, {"shaderStorageImageExtendedFormats", &F::shaderStorageImageExtendedFormats}},
        {spv::CapabilityInterpolationFunction, {"sampleRateShading", &F::sampleRateShading}},
        {spv::CapabilityStorageImageReadWithoutFormat, {"shaderStorageImageReadWithoutFormat", &F::shaderStorageImageReadWithoutFormat}},
        {spv::CapabilityStorageImageWriteWithoutFormat, {"shaderStorageImageWriteWithoutFormat", &F::shaderStorageImageWriteWithoutFormat}},
        {spv::CapabilityMultiViewport, {"multiViewport", &F::multiViewport}},

        // Capabilities that require an extension
        {spv::CapabilityDrawParameters, {VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME, nullptr, &E::vk_khr_shader_draw_parameters}},
        {spv::CapabilityGeometryShaderPassthroughNV, {VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME, nullptr, &E::vk_nv_geometry_shader_passthrough}},
        {spv::CapabilitySampleMaskOverrideCoverageNV, {VK_NV_SAMPLE_MASK_OVERRIDE_COVERAGE_EXTENSION_NAME, nullptr, &E::vk_nv_sample_mask_override_coverage}},
        {spv::CapabilityShaderViewportIndexLayerNV, {VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME, nullptr, &E::vk_nv_viewport_array2}},
        {spv::CapabilityShaderViewportMaskNV, {VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME, nullptr, &E::vk_nv_viewport_array2}},
        {spv::CapabilitySubgroupBallotKHR, {VK_EXT_SHADER_SUBGROUP_BALLOT_EXTENSION_NAME, nullptr, &E::vk_ext_shader_subgroup_ballot }},
        {spv::CapabilitySubgroupVoteKHR, {VK_EXT_SHADER_SUBGROUP_VOTE_EXTENSION_NAME, nullptr, &E::vk_ext_shader_subgroup_vote }},
    };
    // clang-format on

    for (auto insn : *src) {
        if (insn.opcode() == spv::OpCapability) {
            auto it = capabilities.find(insn.word(1));
            if (it != capabilities.end()) {
                if (it->second.feature) {
                    skip |= require_feature(report_data, enabledFeatures->*(it->second.feature), it->second.name);
                }
                if (it->second.extension) {
                    skip |= require_extension(report_data, extensions->*(it->second.extension), it->second.name);
                }
            }
        }
    }

    return skip;
}

static uint32_t descriptor_type_to_reqs(shader_module const *module, uint32_t type_id) {
    auto type = module->get_def(type_id);

    while (true) {
        switch (type.opcode()) {
            case spv::OpTypeArray:
            case spv::OpTypeSampledImage:
                type = module->get_def(type.word(2));
                break;
            case spv::OpTypePointer:
                type = module->get_def(type.word(3));
                break;
            case spv::OpTypeImage: {
                auto dim = type.word(3);
                auto arrayed = type.word(5);
                auto msaa = type.word(6);

                switch (dim) {
                    case spv::Dim1D:
                        return arrayed ? DESCRIPTOR_REQ_VIEW_TYPE_1D_ARRAY : DESCRIPTOR_REQ_VIEW_TYPE_1D;
                    case spv::Dim2D:
                        return (msaa ? DESCRIPTOR_REQ_MULTI_SAMPLE : DESCRIPTOR_REQ_SINGLE_SAMPLE) |
                            (arrayed ? DESCRIPTOR_REQ_VIEW_TYPE_2D_ARRAY : DESCRIPTOR_REQ_VIEW_TYPE_2D);
                    case spv::Dim3D:
                        return DESCRIPTOR_REQ_VIEW_TYPE_3D;
                    case spv::DimCube:
                        return arrayed ? DESCRIPTOR_REQ_VIEW_TYPE_CUBE_ARRAY : DESCRIPTOR_REQ_VIEW_TYPE_CUBE;
                    case spv::DimSubpassData:
                        return msaa ? DESCRIPTOR_REQ_MULTI_SAMPLE : DESCRIPTOR_REQ_SINGLE_SAMPLE;
                    default:  // buffer, etc.
                        return 0;
                }
            }
            default:
                return 0;
        }
    }
}

// For given pipelineLayout verify that the set_layout_node at slot.first
//  has the requested binding at slot.second and return ptr to that binding
static VkDescriptorSetLayoutBinding const *get_descriptor_binding(PIPELINE_LAYOUT_NODE const *pipelineLayout,
                                                                  descriptor_slot_t slot) {
    if (!pipelineLayout) return nullptr;

    if (slot.first >= pipelineLayout->set_layouts.size()) return nullptr;

    return pipelineLayout->set_layouts[slot.first]->GetDescriptorSetLayoutBindingPtrFromBinding(slot.second);
}


static bool validate_pipeline_shader_stage(
    layer_data *dev_data, VkPipelineShaderStageCreateInfo const *pStage, PIPELINE_STATE *pipeline,
    shader_module const **out_module, spirv_inst_iter *out_entrypoint) {
    bool skip = false;
    auto module = *out_module = GetShaderModuleState(dev_data, pStage->module);
    auto report_data = GetReportData(dev_data);

    if (!module->has_valid_spirv) return false;

    // Find the entrypoint
    auto entrypoint = *out_entrypoint = find_entrypoint(module, pStage->pName, pStage->stage);
    if (entrypoint == module->end()) {
        if (log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                    VALIDATION_ERROR_10600586, "SC", "No entrypoint found named `%s` for stage %s. %s.", pStage->pName,
                    string_VkShaderStageFlagBits(pStage->stage), validation_error_map[VALIDATION_ERROR_10600586])) {
            return true;  // no point continuing beyond here, any analysis is just going to be garbage.
        }
    }

    // Validate shader capabilities against enabled device features
    skip |= validate_shader_capabilities(dev_data, module);

    // Mark accessible ids
    auto accessible_ids = mark_accessible_ids(module, entrypoint);

    // Validate descriptor set layout against what the entrypoint actually uses
    auto descriptor_uses = collect_interface_by_descriptor_slot(report_data, module, accessible_ids);

    skip |= validate_specialization_offsets(report_data, pStage);
    skip |= validate_push_constant_usage(report_data, &pipeline->pipeline_layout.push_constant_ranges, module, accessible_ids, pStage->stage);

    // Validate descriptor use
    for (auto use : descriptor_uses) {
        // While validating shaders capture which slots are used by the pipeline
        auto &reqs = pipeline->active_slots[use.first.first][use.first.second];
        reqs = descriptor_req(reqs | descriptor_type_to_reqs(module, use.second.type_id));

        // Verify given pipelineLayout has requested setLayout with requested binding
        const auto &binding = get_descriptor_binding(&pipeline->pipeline_layout, use.first);
        unsigned required_descriptor_count;

        if (!binding) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_MISSING_DESCRIPTOR, "SC",
                            "Shader uses descriptor slot %u.%u (used as type `%s`) but not declared in pipeline layout",
                            use.first.first, use.first.second, describe_type(module, use.second.type_id).c_str());
        } else if (~binding->stageFlags & pStage->stage) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT, 0, __LINE__,
                            SHADER_CHECKER_DESCRIPTOR_NOT_ACCESSIBLE_FROM_STAGE, "SC",
                            "Shader uses descriptor slot %u.%u (used "
                                "as type `%s`) but descriptor not "
                                "accessible from stage %s",
                            use.first.first, use.first.second, describe_type(module, use.second.type_id).c_str(),
                            string_VkShaderStageFlagBits(pStage->stage));
        } else if (!descriptor_type_match(module, use.second.type_id, binding->descriptorType, required_descriptor_count)) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_DESCRIPTOR_TYPE_MISMATCH, "SC",
                            "Type mismatch on descriptor slot "
                                "%u.%u (used as type `%s`) but "
                                "descriptor of type %s",
                            use.first.first, use.first.second, describe_type(module, use.second.type_id).c_str(),
                            string_VkDescriptorType(binding->descriptorType));
        } else if (binding->descriptorCount < required_descriptor_count) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_DESCRIPTOR_TYPE_MISMATCH, "SC",
                            "Shader expects at least %u descriptors for binding %u.%u (used as type `%s`) but only %u provided",
                            required_descriptor_count, use.first.first, use.first.second,
                            describe_type(module, use.second.type_id).c_str(), binding->descriptorCount);
        }
    }

    // Validate use of input attachments against subpass structure
    if (pStage->stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
        auto input_attachment_uses = collect_interface_by_input_attachment_index(module, accessible_ids);

        auto rpci = pipeline->render_pass_ci.ptr();
        auto subpass = pipeline->graphicsPipelineCI.subpass;

        for (auto use : input_attachment_uses) {
            auto input_attachments = rpci->pSubpasses[subpass].pInputAttachments;
            auto index = (input_attachments && use.first < rpci->pSubpasses[subpass].inputAttachmentCount)
                         ? input_attachments[use.first].attachment
                         : VK_ATTACHMENT_UNUSED;

            if (index == VK_ATTACHMENT_UNUSED) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                SHADER_CHECKER_MISSING_INPUT_ATTACHMENT, "SC",
                                "Shader consumes input attachment index %d but not provided in subpass", use.first);
            } else if (!(get_format_type(rpci->pAttachments[index].format) & get_fundamental_type(module, use.second.type_id))) {
                skip |=
                    log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_INPUT_ATTACHMENT_TYPE_MISMATCH, "SC",
                            "Subpass input attachment %u format of %s does not match type used in shader `%s`", use.first,
                            string_VkFormat(rpci->pAttachments[index].format), describe_type(module, use.second.type_id).c_str());
            }
        }
    }

    return skip;
}

static bool validate_interface_between_stages(debug_report_data const *report_data, shader_module const *producer,
                                              spirv_inst_iter producer_entrypoint, shader_stage_attributes const *producer_stage,
                                              shader_module const *consumer, spirv_inst_iter consumer_entrypoint,
                                              shader_stage_attributes const *consumer_stage) {
    bool skip = false;

    auto outputs =
        collect_interface_by_location(producer, producer_entrypoint, spv::StorageClassOutput, producer_stage->arrayed_output);
    auto inputs =
        collect_interface_by_location(consumer, consumer_entrypoint, spv::StorageClassInput, consumer_stage->arrayed_input);

    auto a_it = outputs.begin();
    auto b_it = inputs.begin();

    // Maps sorted by key (location); walk them together to find mismatches
    while ((outputs.size() > 0 && a_it != outputs.end()) || (inputs.size() && b_it != inputs.end())) {
        bool a_at_end = outputs.size() == 0 || a_it == outputs.end();
        bool b_at_end = inputs.size() == 0 || b_it == inputs.end();
        auto a_first = a_at_end ? std::make_pair(0u, 0u) : a_it->first;
        auto b_first = b_at_end ? std::make_pair(0u, 0u) : b_it->first;

        if (b_at_end || ((!a_at_end) && (a_first < b_first))) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                            __LINE__, SHADER_CHECKER_OUTPUT_NOT_CONSUMED, "SC",
                            "%s writes to output location %u.%u which is not consumed by %s", producer_stage->name, a_first.first,
                            a_first.second, consumer_stage->name);
            a_it++;
        } else if (a_at_end || a_first > b_first) {
            skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                            SHADER_CHECKER_INPUT_NOT_PRODUCED, "SC", "%s consumes input location %u.%u which is not written by %s",
                            consumer_stage->name, b_first.first, b_first.second, producer_stage->name);
            b_it++;
        } else {
            // subtleties of arrayed interfaces:
            // - if is_patch, then the member is not arrayed, even though the interface may be.
            // - if is_block_member, then the extra array level of an arrayed interface is not
            //   expressed in the member type -- it's expressed in the block type.
            if (!types_match(producer, consumer, a_it->second.type_id, b_it->second.type_id,
                             producer_stage->arrayed_output && !a_it->second.is_patch && !a_it->second.is_block_member,
                             consumer_stage->arrayed_input && !b_it->second.is_patch && !b_it->second.is_block_member, true)) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__,
                                SHADER_CHECKER_INTERFACE_TYPE_MISMATCH, "SC", "Type mismatch on location %u.%u: '%s' vs '%s'",
                                a_first.first, a_first.second, describe_type(producer, a_it->second.type_id).c_str(),
                                describe_type(consumer, b_it->second.type_id).c_str());
            }
            if (a_it->second.is_patch != b_it->second.is_patch) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT, 0, __LINE__,
                                SHADER_CHECKER_INTERFACE_TYPE_MISMATCH, "SC",
                                "Decoration mismatch on location %u.%u: is per-%s in %s stage but "
                                    "per-%s in %s stage",
                                a_first.first, a_first.second, a_it->second.is_patch ? "patch" : "vertex", producer_stage->name,
                                b_it->second.is_patch ? "patch" : "vertex", consumer_stage->name);
            }
            if (a_it->second.is_relaxed_precision != b_it->second.is_relaxed_precision) {
                skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT, 0, __LINE__,
                                SHADER_CHECKER_INTERFACE_TYPE_MISMATCH, "SC",
                                "Decoration mismatch on location %u.%u: %s and %s stages differ in precision", a_first.first,
                                a_first.second, producer_stage->name, consumer_stage->name);
            }
            a_it++;
            b_it++;
        }
    }

    return skip;
}

// Validate that the shaders used by the given pipeline and store the active_slots
//  that are actually used by the pipeline into pPipeline->active_slots
bool validate_and_capture_pipeline_shader_state(layer_data *dev_data, PIPELINE_STATE *pipeline) {
    auto pCreateInfo = pipeline->graphicsPipelineCI.ptr();
    int vertex_stage = get_shader_stage_id(VK_SHADER_STAGE_VERTEX_BIT);
    int fragment_stage = get_shader_stage_id(VK_SHADER_STAGE_FRAGMENT_BIT);
    auto report_data = GetReportData(dev_data);

    shader_module const *shaders[5];
    memset(shaders, 0, sizeof(shaders));
    spirv_inst_iter entrypoints[5];
    memset(entrypoints, 0, sizeof(entrypoints));
    bool skip = false;

    for (uint32_t i = 0; i < pCreateInfo->stageCount; i++) {
        auto pStage = &pCreateInfo->pStages[i];
        auto stage_id = get_shader_stage_id(pStage->stage);
        skip |= validate_pipeline_shader_stage(dev_data, pStage, pipeline, &shaders[stage_id], &entrypoints[stage_id]);
    }

    // if the shader stages are no good individually, cross-stage validation is pointless.
    if (skip) return true;

    auto vi = pCreateInfo->pVertexInputState;

    if (vi) {
        skip |= validate_vi_consistency(report_data, vi);
    }

    if (shaders[vertex_stage] && shaders[vertex_stage]->has_valid_spirv) {
        skip |= validate_vi_against_vs_inputs(report_data, vi, shaders[vertex_stage], entrypoints[vertex_stage]);
    }

    int producer = get_shader_stage_id(VK_SHADER_STAGE_VERTEX_BIT);
    int consumer = get_shader_stage_id(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);

    while (!shaders[producer] && producer != fragment_stage) {
        producer++;
        consumer++;
    }

    for (; producer != fragment_stage && consumer <= fragment_stage; consumer++) {
        assert(shaders[producer]);
        if (shaders[consumer] && shaders[consumer]->has_valid_spirv && shaders[producer]->has_valid_spirv) {
            skip |= validate_interface_between_stages(report_data, shaders[producer], entrypoints[producer],
                                                      &shader_stage_attribs[producer], shaders[consumer], entrypoints[consumer],
                                                      &shader_stage_attribs[consumer]);

            producer = consumer;
        }
    }

    if (shaders[fragment_stage] && shaders[fragment_stage]->has_valid_spirv) {
        skip |= validate_fs_outputs_against_render_pass(report_data, shaders[fragment_stage], entrypoints[fragment_stage],
                                                        pipeline, pCreateInfo->subpass);
    }

    return skip;
}

bool validate_compute_pipeline(layer_data *dev_data, PIPELINE_STATE *pipeline) {
    auto pCreateInfo = pipeline->computePipelineCI.ptr();

    shader_module const *module;
    spirv_inst_iter entrypoint;

    return validate_pipeline_shader_stage(dev_data, &pCreateInfo->stage, pipeline, &module, &entrypoint);
}

bool PreCallValidateCreateShaderModule(layer_data *dev_data, VkShaderModuleCreateInfo const *pCreateInfo, bool *spirv_valid) {
    bool skip = false;
    spv_result_t spv_valid = SPV_SUCCESS;
    auto report_data = GetReportData(dev_data);

    if (GetDisables(dev_data)->shader_validation) {
        return false;
    }

    auto have_glsl_shader = GetEnabledExtensions(dev_data)->vk_nv_glsl_shader;

    if (!have_glsl_shader && (pCreateInfo->codeSize % 4)) {
        skip |= log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0,
                        __LINE__, VALIDATION_ERROR_12a00ac0, "SC",
                        "SPIR-V module not valid: Codesize must be a multiple of 4 but is " PRINTF_SIZE_T_SPECIFIER ". %s",
                        pCreateInfo->codeSize, validation_error_map[VALIDATION_ERROR_12a00ac0]);
    } else {
        // Use SPIRV-Tools validator to try and catch any issues with the module itself
        spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_0);
        spv_const_binary_t binary{ pCreateInfo->pCode, pCreateInfo->codeSize / sizeof(uint32_t) };
        spv_diagnostic diag = nullptr;

        spv_valid = spvValidate(ctx, &binary, &diag);
        if (spv_valid != SPV_SUCCESS) {
            if (!have_glsl_shader || (pCreateInfo->pCode[0] == spv::MagicNumber)) {
                skip |= log_msg(report_data,
                                spv_valid == SPV_WARNING ? VK_DEBUG_REPORT_WARNING_BIT_EXT : VK_DEBUG_REPORT_ERROR_BIT_EXT,
                                VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, 0, __LINE__, SHADER_CHECKER_INCONSISTENT_SPIRV, "SC",
                                "SPIR-V module not valid: %s", diag && diag->error ? diag->error : "(no error text)");
            }
        }

        spvDiagnosticDestroy(diag);
        spvContextDestroy(ctx);
    }

    *spirv_valid = (spv_valid == SPV_SUCCESS);
    return skip;
}

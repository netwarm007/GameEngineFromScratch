# Copyright (c) 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_spvtools_unittest(TARGET instruction
  SRCS instruction_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET ir_loader
  SRCS ir_loader_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_manager
  SRCS module_utils.h
       pass_manager_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET optimizer
  SRCS optimizer_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_strip_debug_info
  SRCS strip_debug_info_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_compact_ids
  SRCS compact_ids_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_flatten_decoration
  SRCS flatten_decoration_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_freeze_spec_const
  SRCS freeze_spec_const_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_block_merge
  SRCS block_merge_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_inline
  SRCS inline_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_insert_extract_elim
  SRCS insert_extract_elim_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_local_ssa_elim
  SRCS local_ssa_elim_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_local_single_block_elim
  SRCS local_single_block_elim.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_local_access_chain_convert
  SRCS local_access_chain_convert_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_local_single_store_elim
  SRCS local_single_store_elim_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_dead_branch_elim
  SRCS dead_branch_elim_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_aggressive_dce
  SRCS aggressive_dead_code_elim_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_common_uniform_elim
  SRCS common_uniform_elim_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_eliminate_dead_const
  SRCS eliminate_dead_const_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_utils
  SRCS utils_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET def_use
  SRCS def_use_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET assembly_builder
  SRCS assembly_builder_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET types
  SRCS types_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET type_manager
  SRCS type_manager_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET iterator
  SRCS iterator_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET module
  SRCS module_utils.h
       module_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_fold_spec_const_op_composite
  SRCS fold_spec_const_op_composite_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_unify_const
  SRCS unify_const_test.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET pass_set_spec_const_default_value
  SRCS set_spec_const_default_value_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET line_debug_info
  SRCS line_debug_info_test.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt
)

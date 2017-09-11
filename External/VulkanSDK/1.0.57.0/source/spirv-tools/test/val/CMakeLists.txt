# Copyright (c) 2016 The Khronos Group Inc.
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

set(VAL_TEST_COMMON_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../test_fixture.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_spirv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_spirv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val_fixtures.h
  ${CMAKE_CURRENT_SOURCE_DIR}/val_fixtures.cpp
)


add_spvtools_unittest(TARGET val_capability
  SRCS val_capability_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_cfg
  SRCS val_cfg_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_id
  SRCS val_id_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_layout
  SRCS val_layout_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_ssa
  SRCS val_ssa_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_storage
  SRCS val_storage_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_state
  SRCS val_state_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_data
	SRCS val_data_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_type_unique
	SRCS val_type_unique_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_limits
	SRCS val_limits_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_validation_state
	SRCS val_validation_state_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_decoration
	SRCS val_decoration_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_instructions
	SRCS val_instructions_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_extensions
	SRCS val_extensions_test.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests for OpExtension validator rules.

#include <string>

#include "enum_string_mapping.h"
#include "extensions.h"
#include "gmock/gmock.h"
#include "test_fixture.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using ::libspirv::Extension;

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Values;

using std::string;

using ValidateKnownExtensions = spvtest::ValidateBase<string>;
using ValidateUnknownExtensions = spvtest::ValidateBase<string>;
using ValidateExtensionCapabilities = spvtest::ValidateBase<bool>;

// Returns expected error string if |extension| is not recognized.
string GetErrorString(const std::string& extension) {
  return "Found unrecognized extension " + extension;
}

INSTANTIATE_TEST_CASE_P(ExpectSuccess, ValidateKnownExtensions, Values(
    "SPV_AMD_shader_explicit_vertex_parameter",
    "SPV_AMD_shader_trinary_minmax",
    "SPV_AMD_gcn_shader",
    "SPV_AMD_shader_ballot",
    "SPV_AMD_gpu_shader_half_float",
    "SPV_AMD_texture_gather_bias_lod",
    "SPV_AMD_gpu_shader_int16",
    "SPV_KHR_shader_ballot",
    "SPV_KHR_shader_draw_parameters",
    "SPV_KHR_subgroup_vote",
    "SPV_KHR_16bit_storage",
    "SPV_KHR_device_group",
    "SPV_KHR_multiview",
    "SPV_NV_sample_mask_override_coverage",
    "SPV_NV_geometry_shader_passthrough",
    "SPV_NV_viewport_array2",
    "SPV_NV_stereo_view_rendering",
    "SPV_NVX_multiview_per_view_attributes"
    ));

INSTANTIATE_TEST_CASE_P(FailSilently, ValidateUnknownExtensions, Values(
    "ERROR_unknown_extension",
    "SPV_KHR_",
    "SPV_KHR_shader_ballot_ERROR"
    ));

TEST_P(ValidateKnownExtensions, ExpectSuccess) {
  const std::string extension = GetParam();
  const string str =
      "OpCapability Shader\nOpCapability Linkage\nOpExtension \"" + extension +
      "\"\nOpMemoryModel Logical GLSL450";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), Not(HasSubstr(GetErrorString(extension))));
}

TEST_P(ValidateUnknownExtensions, FailSilently) {
  const std::string extension = GetParam();
  const string str =
      "OpCapability Shader\nOpCapability Linkage\nOpExtension \"" + extension +
      "\"\nOpMemoryModel Logical GLSL450";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(GetErrorString(extension)));
}

TEST_F(ValidateExtensionCapabilities, DeclCapabilitySuccess) {
  const string str =
      "OpCapability Shader\nOpCapability Linkage\nOpCapability DeviceGroup\n"
      "OpExtension \"SPV_KHR_device_group\""
      "\nOpMemoryModel Logical GLSL450";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtensionCapabilities, DeclCapabilityFailure) {
  const string str =
      "OpCapability Shader\nOpCapability Linkage\nOpCapability DeviceGroup\n"
      "\nOpMemoryModel Logical GLSL450";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_MISSING_EXTENSION, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("1st operand of Capability"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("requires one of these extensions"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("SPV_KHR_device_group"));
}

}  // anonymous namespace

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

// Validation tests for illegal instructions

#include <sstream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "val_fixtures.h"

using ::testing::HasSubstr;

using ImageSampleTestParameter = std::pair<std::string, std::string>;

const std::string& Opcode(const ImageSampleTestParameter& p) { return p.first; }
const std::string& Operands(const ImageSampleTestParameter& p) {
  return p.second;
}

using ValidateIns = spvtest::ValidateBase<ImageSampleTestParameter>;

namespace {

TEST_P(ValidateIns, Reserved) {
  const auto& param = GetParam();

  std::string str = R"(
             OpCapability Shader
             OpCapability SparseResidency
             OpMemoryModel Logical GLSL450
             OpEntryPoint Fragment %main "main" %img
%void      = OpTypeVoid
%int       = OpTypeInt 32 0
%float     = OpTypeFloat 32
%fnt       = OpTypeFunction %void
%coord     = OpConstantNull %float
%lod       = OpConstantNull %float
%dref      = OpConstantNull %float
%imgt      = OpTypeImage %float 2D 0 0 0 0 Rgba32f
%sampledt  = OpTypeSampledImage %imgt
%sampledp  = OpTypePointer Uniform %sampledt
%img       = OpVariable %sampledp Input
%main      = OpFunction %void None %fnt
%label     = OpLabel
%sample    = Op)" + Opcode(param) +
                    " " + Operands(param) + R"(
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_INVALID_VALUE, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(Opcode(param) + " is reserved for future use."));
}

#define CASE(NAME, ARGS) \
  { "ImageSparseSampleProj" #NAME, ARGS }

INSTANTIATE_TEST_CASE_P(
    OpImageSparseSampleProj, ValidateIns,
    ::testing::ValuesIn(std::vector<ImageSampleTestParameter>{
        CASE(ImplicitLod, "%float %img %coord"),
        CASE(ExplicitLod, "%float %img %coord Lod %lod"),
        CASE(DrefImplicitLod, "%float %img %coord %dref"),
        CASE(DrefExplicitLod, "%float %img %coord %dref Lod %lod"),
    }), );
#undef CASE
}

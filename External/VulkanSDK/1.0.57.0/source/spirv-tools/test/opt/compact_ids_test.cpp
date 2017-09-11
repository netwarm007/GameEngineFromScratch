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

#include <gmock/gmock.h>

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using CompactIdsTest = PassTest<::testing::Test>;

TEST_F(CompactIdsTest, PassOff) {
  const std::string before =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%99 = OpTypeInt 32 0
%10 = OpTypeVector %99 2
%20 = OpConstant %99 2
%30 = OpTypeArray %99 %20
)";

  const std::string after = before;

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::NullPass>(before, after, false, false);
}

TEST_F(CompactIdsTest, PassOn) {
  const std::string before =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %3 "simple_kernel"
%99 = OpTypeInt 32 0
%10 = OpTypeVector %99 2
%20 = OpConstant %99 2
%30 = OpTypeArray %99 %20
%40 = OpTypeVoid
%50 = OpTypeFunction %40
 %3 = OpFunction %40 None %50
%70 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "simple_kernel"
%2 = OpTypeInt 32 0
%3 = OpTypeVector %2 2
%4 = OpConstant %2 2
%5 = OpTypeArray %2 %4
%6 = OpTypeVoid
%7 = OpTypeFunction %6
%1 = OpFunction %6 None %7
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::CompactIdsPass>(before, after, false, false);
}

}  // anonymous namespace

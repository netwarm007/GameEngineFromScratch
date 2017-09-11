// Copyright (c) 2016 Google Inc.
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

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "message.h"
#include "opt/build_module.h"
#include "opt/module.h"
#include "spirv-tools/libspirv.hpp"

#include "module_utils.h"

namespace {

using spvtools::ir::Module;
using spvtest::GetIdBound;
using ::testing::Eq;

TEST(ModuleTest, SetIdBound) {
  Module m;
  // It's initialized to 0.
  EXPECT_EQ(0u, GetIdBound(m));

  m.SetIdBound(19);
  EXPECT_EQ(19u, GetIdBound(m));

  m.SetIdBound(102);
  EXPECT_EQ(102u, GetIdBound(m));
}

// Returns a module formed by assembling the given text,
// then loading the result.
inline std::unique_ptr<Module> BuildModule(std::string text) {
  return spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
}

TEST(ModuleTest, ComputeIdBound) {
  // Emtpy module case.
  EXPECT_EQ(1u, BuildModule("")->ComputeIdBound());
  // Sensitive to result id
  EXPECT_EQ(2u, BuildModule("%void = OpTypeVoid")->ComputeIdBound());
  // Sensitive to type id
  EXPECT_EQ(1000u, BuildModule("%a = OpTypeArray !999 3")->ComputeIdBound());
  // Sensitive to a regular Id parameter
  EXPECT_EQ(2000u, BuildModule("OpDecorate !1999 0")->ComputeIdBound());
  // Sensitive to a scope Id parameter.
  EXPECT_EQ(3000u,
            BuildModule("%f = OpFunction %void None %fntype %a = OpLabel "
                        "OpMemoryBarrier !2999 %b\n")
                ->ComputeIdBound());
  // Sensitive to a semantics Id parameter
  EXPECT_EQ(4000u,
            BuildModule("%f = OpFunction %void None %fntype %a = OpLabel "
                        "OpMemoryBarrier %b !3999\n")
                ->ComputeIdBound());
}

}  // anonymous namespace

// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using InsertExtractElimTest = PassTest<::testing::Test>;

TEST_F(InsertExtractElimTest, Simple) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // 
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  // 
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     gl_FragColor = s0.v1;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpLoad %S_t %s0
%20 = OpCompositeInsert %S_t %18 %19 1
OpStore %s0 %20
%21 = OpCompositeExtract %v4float %20 1
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpLoad %S_t %s0
%20 = OpCompositeInsert %S_t %18 %19 1
OpStore %s0 %20
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(predefs + before, 
      predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, OptimizeAcrossNonConflictingInsert) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // 
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  // 
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     s0.v0[2] = 0.0;
  //     gl_FragColor = s0.v1;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 0 2
OpStore %s0 %22
%23 = OpCompositeExtract %v4float %22 1
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 0 2
OpStore %s0 %22
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(predefs + before, 
      predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, ConflictingInsertPreventsOptimization) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // 
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  // 
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     s0.v1[2] = 0.0;
  //     gl_FragColor = s0.v1;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 1 2
OpStore %s0 %22
%23 = OpCompositeExtract %v4float %22 1
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(assembly, 
      assembly, true, true);
}

TEST_F(InsertExtractElimTest, ConflictingInsertPreventsOptimization2) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // 
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  // 
  // void main()
  // {
  //     S_t s0;
  //     s0.v1[1] = 1.0;
  //     s0.v1 = Baseline;
  //     gl_FragColor = vec4(s0.v1[1], 0.0, 0.0, 0.0);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%main = OpFunction %void None %8
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %S_t %s0
%24 = OpCompositeInsert %S_t %float_1 %23 1 1
%25 = OpLoad %v4float %BaseColor
%26 = OpCompositeInsert %S_t %25 %24 1
%27 = OpCompositeExtract %float %26 1 1
%28 = OpCompositeConstruct %v4float %27 %float_0 %float_0 %float_0
OpStore %gl_FragColor %28
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(assembly, 
      assembly, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//

}  // anonymous namespace

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

using BlockMergeTest = PassTest<::testing::Test>;

TEST_F(BlockMergeTest, Simple) {
  // Note: SPIR-V hand edited to insert block boundary
  // between two statements in main.
  //
  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      gl_FragColor = v;
  //  }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
OpBranch %15
%15 = OpLabel
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::BlockMergePass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(BlockMergeTest, EmptyBlock) {
  // Note: SPIR-V hand edited to insert empty block
  // after two statements in main.
  //
  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      gl_FragColor = v;
  //  }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
OpBranch %15
%15 = OpLabel
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpBranch %17
%17 = OpLabel
OpBranch %18
%18 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::BlockMergePass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(BlockMergeTest, NoOptOfMergeOrContinueBlock) {
  // Note: SPIR-V hand edited remove dead branch and add block
  // before continue block
  //
  // #version 140
  // in vec4 BaseColor;
  // 
  // void main()
  // {
  //     while (true) {
  //         break;
  //     }
  //     gl_FragColor = BaseColor;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %6
%13 = OpLabel
OpBranch %14
%14 = OpLabel
OpLoopMerge %15 %16 None
OpBranch %17
%17 = OpLabel
OpBranch %15
%18 = OpLabel
OpBranch %16
%16 = OpLabel
OpBranch %14
%15 = OpLabel
%19 = OpLoad %v4float %BaseColor
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::BlockMergePass>(
      assembly, assembly, true, true);
}

TEST_F(BlockMergeTest, NestedInControlFlow) {
  // Note: SPIR-V hand edited to insert block boundary
  // between OpFMul and OpStore in then-part.
  //
  // #version 140
  // in vec4 BaseColor;
  // 
  // layout(std140) uniform U_t
  // {
  //     bool g_B ;
  // } ;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (g_B) 
  //       vec4 v = v * 0.25;
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpName %_ ""
OpName %v_0 "v"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%U_t = OpTypeStruct %uint
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_0_25 = OpConstant %float 0.25
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%24 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%v_0 = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %BaseColor
OpStore %v %25
%26 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%27 = OpLoad %uint %26
%28 = OpINotEqual %bool %27 %uint_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_25
OpBranch %33
%33 = OpLabel
OpStore %v_0 %32
OpBranch %29
%29 = OpLabel
%34 = OpLoad %v4float %v
OpStore %gl_FragColor %34
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%24 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%v_0 = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %BaseColor
OpStore %v %25
%26 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%27 = OpLoad %uint %26
%28 = OpINotEqual %bool %27 %uint_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_25
OpStore %v_0 %32
OpBranch %29
%29 = OpLabel
%34 = OpLoad %v4float %v
OpStore %gl_FragColor %34
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::BlockMergePass>(
      predefs + before, predefs + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    More complex control flow
//    Others?

}  // anonymous namespace

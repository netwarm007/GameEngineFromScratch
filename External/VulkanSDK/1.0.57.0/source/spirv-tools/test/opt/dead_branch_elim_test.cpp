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

using DeadBranchElimTest = PassTest<::testing::Test>;

TEST_F(DeadBranchElimTest, IfThenElseTrue) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (true)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%16 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpSelectionMerge %20 None
OpBranchConditional %true %21 %22
%21 = OpLabel
OpStore %v %14
OpBranch %20
%22 = OpLabel
OpStore %v %16
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpBranch %21
%21 = OpLabel
OpStore %v %14
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, IfThenElseFalse) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (false)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%16 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpSelectionMerge %20 None
OpBranchConditional %false %21 %22
%21 = OpLabel
OpStore %v %14
OpBranch %20
%22 = OpLabel
OpStore %v %16
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpBranch %22
%22 = OpLabel
OpStore %v %16
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, IfThenTrue) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (true)
  //       v = v * vec4(0.5,0.5,0.5,0.5);
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
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float_0_5 = OpConstant %float 0.5
%15 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18 
OpSelectionMerge %19 None 
OpBranchConditional %true %20 %19
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22 
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpBranch %20
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, IfThenFalse) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (false)
  //       v = v * vec4(0.5,0.5,0.5,0.5);
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
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float_0_5 = OpConstant %float 0.5
%15 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18 
OpSelectionMerge %19 None 
OpBranchConditional %false %20 %19
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22 
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, IfThenElsePhiTrue) {
  // Test handling of phi in merge block after dead branch elimination.
  // Note: The SPIR-V has had store/load elimination and phi insertion
  //
  // #version 140
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (true)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%12 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%14 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
)";

  const std::string before =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpSelectionMerge %18 None
OpBranchConditional %true %19 %20
%19 = OpLabel
OpBranch %18
%20 = OpLabel
OpBranch %18
%18 = OpLabel
%21 = OpPhi %v4float %12 %19 %14 %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpBranch %19
%19 = OpLabel
OpBranch %18
%18 = OpLabel
OpStore %gl_FragColor %12
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, IfThenElsePhiFalse) {
  // Test handling of phi in merge block after dead branch elimination.
  // Note: The SPIR-V has had store/load elimination and phi insertion
  //
  // #version 140
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (true)
  //       v = vec4(0.0,0.0,0.0,0.0);
  //     else
  //       v = vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%12 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%14 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
)";

  const std::string before =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpSelectionMerge %18 None
OpBranchConditional %false %19 %20
%19 = OpLabel
OpBranch %18
%20 = OpLabel
OpBranch %18
%18 = OpLabel
%21 = OpPhi %v4float %12 %19 %14 %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %5
%17 = OpLabel
OpBranch %20
%20 = OpLabel
OpBranch %18
%18 = OpLabel
OpStore %gl_FragColor %14
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, CompoundIfThenElseFalse) {
  // #version 140
  // 
  // layout(std140) uniform U_t
  // {
  //     bool g_B ;
  // } ;
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (false) {
  //       if (g_B)
  //         v = vec4(0.0,0.0,0.0,0.0);
  //       else
  //         v = vec4(1.0,1.0,1.0,1.0);
  //     } else {
  //       if (g_B)
  //         v = vec4(1.0,1.0,1.0,1.0);
  //       else
  //         v = vec4(0.0,0.0,0.0,0.0);
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpName %_ ""
OpName %v "v"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%uint = OpTypeInt 32 0
%U_t = OpTypeStruct %uint
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%uint_0 = OpConstant %uint 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%float_0 = OpConstant %float 0
%21 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%25 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpSelectionMerge %26 None
OpBranchConditional %false %27 %28 
%27 = OpLabel
%29 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%30 = OpLoad %uint %29
%31 = OpINotEqual %bool %30 %uint_0
OpSelectionMerge %32 None
OpBranchConditional %31 %33 %34
%33 = OpLabel
OpStore %v %21
OpBranch %32
%34 = OpLabel
OpStore %v %23
OpBranch %32
%32 = OpLabel
OpBranch %26
%28 = OpLabel
%35 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%36 = OpLoad %uint %35
%37 = OpINotEqual %bool %36 %uint_0
OpSelectionMerge %38 None
OpBranchConditional %37 %39 %40
%39 = OpLabel
OpStore %v %23
OpBranch %38
%40 = OpLabel
OpStore %v %21
OpBranch %38
%38 = OpLabel
OpBranch %26
%26 = OpLabel
%41 = OpLoad %v4float %v
OpStore %gl_FragColor %41
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%25 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
OpBranch %28
%28 = OpLabel
%35 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%36 = OpLoad %uint %35
%37 = OpINotEqual %bool %36 %uint_0
OpSelectionMerge %38 None
OpBranchConditional %37 %39 %40
%39 = OpLabel
OpStore %v %23
OpBranch %38
%40 = OpLabel
OpStore %v %21
OpBranch %38
%38 = OpLabel
OpBranch %26
%26 = OpLabel
%41 = OpLoad %v4float %v
OpStore %gl_FragColor %41
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, NoOrphanMerge) {

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
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float_0_5 = OpConstant %float 0.5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
OpSelectionMerge %18 None
OpBranchConditional %true %19 %20
%19 = OpLabel
OpKill
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpVectorTimesScalar %v4float %21 %float_0_5
OpStore %v %22
OpBranch %18
%18 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
OpSelectionMerge %18 None
OpBranchConditional %true %19 %18
%19 = OpLabel
OpKill
%18 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, KeepContinueTargetWhenKillAfterMerge) {
  // #version 450
  // void main() {
  //   bool c;
  //   bool d;
  //   while(c) {
  //     if(d) {
  //      continue;
  //     }
  //     if(false) {
  //      continue;
  //     }
  //     discard;
  //   }
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %c "c"
OpName %d "d"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
%false = OpConstantFalse %bool
)";

  const std::string before =
      R"(%main = OpFunction %void None %6
%10 = OpLabel
%c = OpVariable %_ptr_Function_bool Function
%d = OpVariable %_ptr_Function_bool Function
OpBranch %11
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranch %14
%14 = OpLabel
%15 = OpLoad %bool %c
OpBranchConditional %15 %16 %12
%16 = OpLabel
%17 = OpLoad %bool %d
OpSelectionMerge %18 None
OpBranchConditional %17 %19 %18
%19 = OpLabel
OpBranch %13
%18 = OpLabel
OpSelectionMerge %20 None
OpBranchConditional %false %21 %20 
%21 = OpLabel
OpBranch %13
%20 = OpLabel
OpKill
%13 = OpLabel
OpBranch %11
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %6
%10 = OpLabel
%c = OpVariable %_ptr_Function_bool Function
%d = OpVariable %_ptr_Function_bool Function
OpBranch %11
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranch %14
%14 = OpLabel
%15 = OpLoad %bool %c
OpBranchConditional %15 %16 %12
%16 = OpLabel
%17 = OpLoad %bool %d
OpSelectionMerge %18 None
OpBranchConditional %17 %19 %18
%19 = OpLabel
OpBranch %13
%18 = OpLabel
OpBranch %20
%20 = OpLabel
OpKill
%13 = OpLabel
OpBranch %11
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(DeadBranchElimTest, DecorateDeleted) {
  // Note: SPIR-V hand-edited to add decoration
  // #version 140
  // 
  // in vec4 BaseColor;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (false)
  //       v = v * vec4(0.5,0.5,0.5,0.5);
  //     gl_FragColor = v;
  // }

  const std::string predefs_before =
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
OpDecorate %22 RelaxedPrecision
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float_0_5 = OpConstant %float 0.5
%15 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
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
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%bool = OpTypeBool
%false = OpConstantFalse %bool
%float_0_5 = OpConstant %float 0.5
%16 = OpConstantComposite %v4float %float_0_5 %float_0_5 %float_0_5 %float_0_5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18 
OpSelectionMerge %19 None 
OpBranchConditional %false %20 %19
%20 = OpLabel
%21 = OpLoad %v4float %v
%22 = OpFMul %v4float %21 %15
OpStore %v %22 
OpBranch %19
%19 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
OpBranch %20
%20 = OpLabel
%23 = OpLoad %v4float %v
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::DeadBranchElimPass>(
      predefs_before + before, predefs_after + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    More complex control flow
//    Others?

}  // anonymous namespace

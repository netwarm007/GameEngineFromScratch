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

using LocalSSAElimTest = PassTest<::testing::Test>;

TEST_F(LocalSSAElimTest, ForLoop) {
  // #version 140
  // 
  // in vec4 BC;
  // out float fo;
  // 
  // void main()
  // {
  //     float f = 0.0;
  //     for (int i=0; i<4; i++) {
  //       f = f + BC[i];
  //     }
  //     fo = f;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %fo
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %f "f"
OpName %i "i"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%fo = OpVariable %_ptr_Output_float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %f %float_0
OpStore %i %int_0
OpBranch %23
%23 = OpLabel
OpLoopMerge %24 %25 None
OpBranch %26
%26 = OpLabel
%27 = OpLoad %int %i
%28 = OpSLessThan %bool %27 %int_4
OpBranchConditional %28 %29 %24
%29 = OpLabel
%30 = OpLoad %float %f
%31 = OpLoad %int %i
%32 = OpAccessChain %_ptr_Input_float %BC %31
%33 = OpLoad %float %32
%34 = OpFAdd %float %30 %33
OpStore %f %34
OpBranch %25
%25 = OpLabel
%35 = OpLoad %int %i
%36 = OpIAdd %int %35 %int_1
OpStore %i %36
OpBranch %23
%24 = OpLabel
%37 = OpLoad %float %f
OpStore %fo %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
OpBranch %23
%23 = OpLabel
%38 = OpPhi %float %float_0 %22 %34 %25
%39 = OpPhi %int %int_0 %22 %36 %25
OpLoopMerge %24 %25 None
OpBranch %26
%26 = OpLabel
%28 = OpSLessThan %bool %39 %int_4
OpBranchConditional %28 %29 %24
%29 = OpLabel
%32 = OpAccessChain %_ptr_Input_float %BC %39
%33 = OpLoad %float %32
%34 = OpFAdd %float %38 %33
OpBranch %25
%25 = OpLabel
%36 = OpIAdd %int %39 %int_1
OpBranch %23
%24 = OpLabel
OpStore %fo %38
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, ForLoopWithContinue) {
  // #version 140
  // 
  // in vec4 BC;
  // out float fo;
  // 
  // void main()
  // {
  //     float f = 0.0;
  //     for (int i=0; i<4; i++) {
  //       float t = BC[i];
  //       if (t < 0.0)
  //         continue;
  //       f = f + t;
  //     }
  //     fo = f;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %fo
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %f "f"
OpName %i "i"
OpName %t "t"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%fo = OpVariable %_ptr_Output_float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%23 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_float Function
OpStore %f %float_0
OpStore %i %int_0
OpBranch %24
%24 = OpLabel
OpLoopMerge %25 %26 None
OpBranch %27
%27 = OpLabel
%28 = OpLoad %int %i
%29 = OpSLessThan %bool %28 %int_4
OpBranchConditional %29 %30 %25
%30 = OpLabel
%31 = OpLoad %int %i
%32 = OpAccessChain %_ptr_Input_float %BC %31
%33 = OpLoad %float %32
OpStore %t %33
%34 = OpLoad %float %t
%35 = OpFOrdLessThan %bool %34 %float_0
OpSelectionMerge %36 None
OpBranchConditional %35 %37 %36
%37 = OpLabel
OpBranch %26
%36 = OpLabel
%38 = OpLoad %float %f
%39 = OpLoad %float %t
%40 = OpFAdd %float %38 %39
OpStore %f %40
OpBranch %26
%26 = OpLabel
%41 = OpLoad %int %i
%42 = OpIAdd %int %41 %int_1
OpStore %i %42
OpBranch %24
%25 = OpLabel
%43 = OpLoad %float %f
OpStore %fo %43
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%46 = OpUndef %float
%main = OpFunction %void None %9
%23 = OpLabel
OpBranch %24
%24 = OpLabel
%44 = OpPhi %float %float_0 %23 %48 %26
%45 = OpPhi %int %int_0 %23 %42 %26
%47 = OpPhi %float %46 %23 %33 %26
OpLoopMerge %25 %26 None
OpBranch %27
%27 = OpLabel
%29 = OpSLessThan %bool %45 %int_4
OpBranchConditional %29 %30 %25
%30 = OpLabel
%32 = OpAccessChain %_ptr_Input_float %BC %45
%33 = OpLoad %float %32
%35 = OpFOrdLessThan %bool %33 %float_0
OpSelectionMerge %36 None
OpBranchConditional %35 %37 %36
%37 = OpLabel
OpBranch %26
%36 = OpLabel
%40 = OpFAdd %float %44 %33
OpBranch %26
%26 = OpLabel
%48 = OpPhi %float %44 %37 %40 %36
%42 = OpIAdd %int %45 %int_1
OpBranch %24
%25 = OpLabel
OpStore %fo %44
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, ForLoopWithBreak) {
  // #version 140
  // 
  // in vec4 BC;
  // out float fo;
  // 
  // void main()
  // {
  //     float f = 0.0;
  //     for (int i=0; i<4; i++) {
  //       float t = f + BC[i];
  //       if (t > 1.0)
  //         break;
  //       f = t;
  //     }
  //     fo = f;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %fo
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %f "f"
OpName %i "i"
OpName %t "t"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%fo = OpVariable %_ptr_Output_float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%24 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_float Function
OpStore %f %float_0
OpStore %i %int_0
OpBranch %25
%25 = OpLabel
OpLoopMerge %26 %27 None
OpBranch %28
%28 = OpLabel
%29 = OpLoad %int %i
%30 = OpSLessThan %bool %29 %int_4
OpBranchConditional %30 %31 %26
%31 = OpLabel
%32 = OpLoad %float %f
%33 = OpLoad %int %i
%34 = OpAccessChain %_ptr_Input_float %BC %33
%35 = OpLoad %float %34
%36 = OpFAdd %float %32 %35
OpStore %t %36
%37 = OpLoad %float %t
%38 = OpFOrdGreaterThan %bool %37 %float_1
OpSelectionMerge %39 None
OpBranchConditional %38 %40 %39
%40 = OpLabel
OpBranch %26
%39 = OpLabel
%41 = OpLoad %float %t
OpStore %f %41
OpBranch %27
%27 = OpLabel
%42 = OpLoad %int %i
%43 = OpIAdd %int %42 %int_1
OpStore %i %43
OpBranch %25
%26 = OpLabel
%44 = OpLoad %float %f
OpStore %fo %44
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%47 = OpUndef %float
%main = OpFunction %void None %9
%24 = OpLabel
OpBranch %25
%25 = OpLabel
%45 = OpPhi %float %float_0 %24 %36 %27
%46 = OpPhi %int %int_0 %24 %43 %27
%48 = OpPhi %float %47 %24 %36 %27
OpLoopMerge %26 %27 None
OpBranch %28
%28 = OpLabel
%30 = OpSLessThan %bool %46 %int_4
OpBranchConditional %30 %31 %26
%31 = OpLabel
%34 = OpAccessChain %_ptr_Input_float %BC %46
%35 = OpLoad %float %34
%36 = OpFAdd %float %45 %35
%38 = OpFOrdGreaterThan %bool %36 %float_1
OpSelectionMerge %39 None
OpBranchConditional %38 %40 %39
%40 = OpLabel
OpBranch %26
%39 = OpLabel
OpBranch %27
%27 = OpLabel
%43 = OpIAdd %int %46 %int_1
OpBranch %25
%26 = OpLabel
%49 = OpPhi %float %48 %28 %36 %40
OpStore %fo %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, SwapProblem) {
  // #version 140
  // 
  // in float fe;
  // out float fo;
  // 
  // void main()
  // {
  //     float f1 = 0.0;
  //     float f2 = 1.0;
  //     int ie = int(fe);
  //     for (int i=0; i<ie; i++) {
  //       float t = f1;
  //       f1 = f2;
  //       f2 = t;
  //     }
  //     fo = f1;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %fe %fo
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %f1 "f1"
OpName %f2 "f2"
OpName %ie "ie"
OpName %fe "fe"
OpName %i "i"
OpName %t "t"
OpName %fo "fo"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %fe "fe"
OpName %fo "fo"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_float = OpTypePointer Input %float
%fe = OpVariable %_ptr_Input_float Input
%int_0 = OpConstant %int 0
%bool = OpTypeBool
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%fo = OpVariable %_ptr_Output_float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%23 = OpLabel
%f1 = OpVariable %_ptr_Function_float Function
%f2 = OpVariable %_ptr_Function_float Function
%ie = OpVariable %_ptr_Function_int Function
%i = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_float Function
OpStore %f1 %float_0
OpStore %f2 %float_1
%24 = OpLoad %float %fe
%25 = OpConvertFToS %int %24
OpStore %ie %25
OpStore %i %int_0
OpBranch %26
%26 = OpLabel
OpLoopMerge %27 %28 None
OpBranch %29
%29 = OpLabel
%30 = OpLoad %int %i
%31 = OpLoad %int %ie 
%32 = OpSLessThan %bool %30 %31
OpBranchConditional %32 %33 %27
%33 = OpLabel
%34 = OpLoad %float %f1
OpStore %t %34
%35 = OpLoad %float %f2
OpStore %f1 %35
%36 = OpLoad %float %t
OpStore %f2 %36
OpBranch %28
%28 = OpLabel
%37 = OpLoad %int %i
%38 = OpIAdd %int %37 %int_1
OpStore %i %38
OpBranch %26
%27 = OpLabel
%39 = OpLoad %float %f1
OpStore %fo %39
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%43 = OpUndef %float
%main = OpFunction %void None %11
%23 = OpLabel
%24 = OpLoad %float %fe
%25 = OpConvertFToS %int %24
OpBranch %26
%26 = OpLabel
%40 = OpPhi %float %float_0 %23 %41 %28
%41 = OpPhi %float %float_1 %23 %40 %28
%42 = OpPhi %int %int_0 %23 %38 %28
%44 = OpPhi %float %43 %23 %40 %28
OpLoopMerge %27 %28 None
OpBranch %29
%29 = OpLabel
%32 = OpSLessThan %bool %42 %25
OpBranchConditional %32 %33 %27
%33 = OpLabel
OpBranch %28
%28 = OpLabel
%38 = OpIAdd %int %42 %int_1
OpBranch %26
%27 = OpLabel
OpStore %fo %40
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, LostCopyProblem) {
  // #version 140
  // 
  // in vec4 BC;
  // out float fo;
  // 
  // void main()
  // {
  //     float f = 0.0;
  //     float t;
  //     for (int i=0; i<4; i++) {
  //       t = f;
  //       f = f + BC[i];
  //       if (f > 1.0)
  //         break;
  //     }
  //     fo = t;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %fo
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %f "f"
OpName %i "i"
OpName %t "t"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BC "BC"
OpName %fo "fo"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%_ptr_Output_float = OpTypePointer Output %float
%fo = OpVariable %_ptr_Output_float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%24 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_float Function
OpStore %f %float_0
OpStore %i %int_0
OpBranch %25
%25 = OpLabel
OpLoopMerge %26 %27 None
OpBranch %28
%28 = OpLabel
%29 = OpLoad %int %i
%30 = OpSLessThan %bool %29 %int_4
OpBranchConditional %30 %31 %26
%31 = OpLabel
%32 = OpLoad %float %f
OpStore %t %32
%33 = OpLoad %float %f
%34 = OpLoad %int %i
%35 = OpAccessChain %_ptr_Input_float %BC %34
%36 = OpLoad %float %35
%37 = OpFAdd %float %33 %36
OpStore %f %37
%38 = OpLoad %float %f
%39 = OpFOrdGreaterThan %bool %38 %float_1
OpSelectionMerge %40 None
OpBranchConditional %39 %41 %40
%41 = OpLabel
OpBranch %26
%40 = OpLabel
OpBranch %27
%27 = OpLabel
%42 = OpLoad %int %i
%43 = OpIAdd %int %42 %int_1
OpStore %i %43
OpBranch %25
%26 = OpLabel
%44 = OpLoad %float %t
OpStore %fo %44
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%47 = OpUndef %float
%main = OpFunction %void None %9
%24 = OpLabel
OpBranch %25
%25 = OpLabel
%45 = OpPhi %float %float_0 %24 %37 %27
%46 = OpPhi %int %int_0 %24 %43 %27
%48 = OpPhi %float %47 %24 %45 %27
OpLoopMerge %26 %27 None
OpBranch %28
%28 = OpLabel
%30 = OpSLessThan %bool %46 %int_4
OpBranchConditional %30 %31 %26
%31 = OpLabel
%35 = OpAccessChain %_ptr_Input_float %BC %46
%36 = OpLoad %float %35
%37 = OpFAdd %float %45 %36
%39 = OpFOrdGreaterThan %bool %37 %float_1
OpSelectionMerge %40 None
OpBranchConditional %39 %41 %40
%41 = OpLabel
OpBranch %26
%40 = OpLabel
OpBranch %27
%27 = OpLabel
%43 = OpIAdd %int %46 %int_1
OpBranch %25
%26 = OpLabel
%49 = OpPhi %float %45 %28 %37 %41
%50 = OpPhi %float %48 %28 %45 %41
OpStore %fo %50
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, IfThenElse) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float f;
  // 
  // void main()
  // {
  //     vec4 v;
  //     if (f >= 0)
  //       v = BaseColor * 0.5;
  //     else
  //       v = BaseColor + vec4(1.0,1.0,1.0,1.0);
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %f %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %f "f"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %f "f"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
%f = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%float_0_5 = OpConstant %float 0.5
%float_1 = OpConstant %float 1
%18 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %float %f
%22 = OpFOrdGreaterThanEqual %bool %21 %float_0
OpSelectionMerge %23 None
OpBranchConditional %22 %24 %25
%24 = OpLabel
%26 = OpLoad %v4float %BaseColor
%27 = OpVectorTimesScalar %v4float %26 %float_0_5
OpStore %v %27
OpBranch %23
%25 = OpLabel
%28 = OpLoad %v4float %BaseColor
%29 = OpFAdd %v4float %28 %18
OpStore %v %29
OpBranch %23
%23 = OpLabel
%30 = OpLoad %v4float %v
OpStore %gl_FragColor %30
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%20 = OpLabel
%21 = OpLoad %float %f
%22 = OpFOrdGreaterThanEqual %bool %21 %float_0
OpSelectionMerge %23 None
OpBranchConditional %22 %24 %25
%24 = OpLabel
%26 = OpLoad %v4float %BaseColor
%27 = OpVectorTimesScalar %v4float %26 %float_0_5
OpBranch %23
%25 = OpLabel
%28 = OpLoad %v4float %BaseColor
%29 = OpFAdd %v4float %28 %18
OpBranch %23
%23 = OpLabel
%31 = OpPhi %v4float %27 %24 %29 %25
OpStore %gl_FragColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, IfThen) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float f;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (f <= 0)
  //       v = v * 0.5;
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %f %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%f = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%float_0_5 = OpConstant %float 0.5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
%20 = OpLoad %float %f
%21 = OpFOrdLessThanEqual %bool %20 %float_0
OpSelectionMerge %22 None
OpBranchConditional %21 %23 %22
%23 = OpLabel
%24 = OpLoad %v4float %v
%25 = OpVectorTimesScalar %v4float %24 %float_0_5
OpStore %v %25
OpBranch %22
%22 = OpLabel
%26 = OpLoad %v4float %v
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %float %f
%21 = OpFOrdLessThanEqual %bool %20 %float_0
OpSelectionMerge %22 None
OpBranchConditional %21 %23 %22
%23 = OpLabel
%25 = OpVectorTimesScalar %v4float %19 %float_0_5
OpBranch %22
%22 = OpLabel
%27 = OpPhi %v4float %19 %18 %25 %23
OpStore %gl_FragColor %27
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, Switch) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float f;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     int i = int(f);
  //     switch (i) {
  //       case 0:
  //         v = v * 0.1;
  //         break;
  //       case 1:
  //         v = v * 0.3;
  //         break;
  //       case 2:
  //         v = v * 0.7;
  //         break;
  //       default:
  //         break;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %f %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %i "i"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_float = OpTypePointer Input %float
%f = OpVariable %_ptr_Input_float Input
%float_0_1 = OpConstant %float 0.1
%float_0_3 = OpConstant %float 0.3
%float_0_7 = OpConstant %float 0.7
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%21 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%22 = OpLoad %v4float %BaseColor
OpStore %v %22
%23 = OpLoad %float %f
%24 = OpConvertFToS %int %23
OpStore %i %24
%25 = OpLoad %int %i
OpSelectionMerge %26 None
OpSwitch %25 %27 0 %28 1 %29 2 %30
%27 = OpLabel
OpBranch %26
%28 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_1
OpStore %v %32
OpBranch %26
%29 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpVectorTimesScalar %v4float %33 %float_0_3
OpStore %v %34
OpBranch %26
%30 = OpLabel
%35 = OpLoad %v4float %v
%36 = OpVectorTimesScalar %v4float %35 %float_0_7
OpStore %v %36
OpBranch %26
%26 = OpLabel
%37 = OpLoad %v4float %v
OpStore %gl_FragColor %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%21 = OpLabel
%22 = OpLoad %v4float %BaseColor
%23 = OpLoad %float %f
%24 = OpConvertFToS %int %23
OpSelectionMerge %26 None
OpSwitch %24 %27 0 %28 1 %29 2 %30
%27 = OpLabel
OpBranch %26
%28 = OpLabel
%32 = OpVectorTimesScalar %v4float %22 %float_0_1
OpBranch %26
%29 = OpLabel
%34 = OpVectorTimesScalar %v4float %22 %float_0_3
OpBranch %26
%30 = OpLabel
%36 = OpVectorTimesScalar %v4float %22 %float_0_7
OpBranch %26
%26 = OpLabel
%38 = OpPhi %v4float %22 %27 %32 %28 %34 %29 %36 %30
OpStore %gl_FragColor %38
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

TEST_F(LocalSSAElimTest, SwitchWithFallThrough) {
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float f;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     int i = int(f);
  //     switch (i) {
  //       case 0:
  //         v = v * 0.1;
  //         break;
  //       case 1:
  //         v = v + 0.1;
  //       case 2:
  //         v = v * 0.7;
  //         break;
  //       default:
  //         break;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %f %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %i "i"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_float = OpTypePointer Input %float
%f = OpVariable %_ptr_Input_float Input
%float_0_1 = OpConstant %float 0.1
%float_0_7 = OpConstant %float 0.7
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%21 = OpLoad %v4float %BaseColor
OpStore %v %21
%22 = OpLoad %float %f
%23 = OpConvertFToS %int %22
OpStore %i %23
%24 = OpLoad %int %i
OpSelectionMerge %25 None
OpSwitch %24 %26 0 %27 1 %28 2 %29
%26 = OpLabel
OpBranch %25
%27 = OpLabel
%30 = OpLoad %v4float %v
%31 = OpVectorTimesScalar %v4float %30 %float_0_1
OpStore %v %31
OpBranch %25
%28 = OpLabel
%32 = OpLoad %v4float %v
%33 = OpCompositeConstruct %v4float %float_0_1 %float_0_1 %float_0_1 %float_0_1
%34 = OpFAdd %v4float %32 %33
OpStore %v %34 
OpBranch %29
%29 = OpLabel
%35 = OpLoad %v4float %v
%36 = OpVectorTimesScalar %v4float %35 %float_0_7
OpStore %v %36 
OpBranch %25
%25 = OpLabel
%37 = OpLoad %v4float %v
OpStore %gl_FragColor %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%20 = OpLabel
%21 = OpLoad %v4float %BaseColor
%22 = OpLoad %float %f
%23 = OpConvertFToS %int %22
OpSelectionMerge %25 None
OpSwitch %23 %26 0 %27 1 %28 2 %29
%26 = OpLabel
OpBranch %25
%27 = OpLabel
%31 = OpVectorTimesScalar %v4float %21 %float_0_1
OpBranch %25
%28 = OpLabel
%33 = OpCompositeConstruct %v4float %float_0_1 %float_0_1 %float_0_1 %float_0_1
%34 = OpFAdd %v4float %21 %33
OpBranch %29
%29 = OpLabel
%38 = OpPhi %v4float %21 %20 %34 %28
%36 = OpVectorTimesScalar %v4float %38 %float_0_7
OpBranch %25
%25 = OpLabel
%39 = OpPhi %v4float %21 %26 %31 %27 %36 %29
OpStore %gl_FragColor %39
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalMultiStoreElimPass>(
      predefs + names_before + predefs2 + before,
      predefs + names_after + predefs2 + after,
      true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    No optimization in the presence of
//      access chains
//      function calls
//      OpCopyMemory?
//      unsupported extensions
//    Others?

}  // anonymous namespace

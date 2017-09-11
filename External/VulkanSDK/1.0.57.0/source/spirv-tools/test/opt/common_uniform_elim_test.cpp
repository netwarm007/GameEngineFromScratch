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

using CommonUniformElimTest = PassTest<::testing::Test>;

TEST_F(CommonUniformElimTest, Basic1) {
  // Note: This test exemplifies the following:
  // - Common uniform (%_) load floated to nearest non-controlled block
  // - Common extract (g_F) floated to non-controlled block
  // - Non-common extract (g_F2) not floated, but common uniform load shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  // 
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  //     float g_F2;
  // } ;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (fi > 0) {
  //       v = v * g_F;
  //     }
  //     else {
  //       float f2 = g_F2 - g_F;
  //       v = v * f2;
  //     }
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %f2 "f2"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Function_float = OpTypePointer Function %float
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %30
%32 = OpLabel
%37 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%38 = OpLoad %float %37
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFSub %float %38 %40
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f2 = OpVariable %_ptr_Function_float Function
%52 = OpLoad %U_t %_
%53 = OpCompositeExtract %float %52 0
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpLoad %float %fi
%29 = OpFOrdGreaterThan %bool %28 %float_0
OpSelectionMerge %30 None
OpBranchConditional %29 %31 %32
%31 = OpLabel
%33 = OpLoad %v4float %v
%36 = OpVectorTimesScalar %v4float %33 %53
OpStore %v %36
OpBranch %30
%32 = OpLabel
%49 = OpCompositeExtract %float %52 1
%41 = OpFSub %float %49 %53
OpStore %f2 %41
%42 = OpLoad %v4float %v
%43 = OpLoad %float %f2
%44 = OpVectorTimesScalar %v4float %42 %43
OpStore %v %44
OpBranch %30
%30 = OpLabel
%45 = OpLoad %v4float %v
OpStore %gl_FragColor %45
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CommonUniformElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Basic2) {
  // Note: This test exemplifies the following:
  // - Common uniform (%_) load floated to nearest non-controlled block
  // - Common extract (g_F) floated to non-controlled block
  // - Non-common extract (g_F2) not floated, but common uniform load shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  // in float fi2;
  // 
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  //     float g_F2;
  // } ;
  // 
  // void main()
  // {
  //     float f = fi;
  //     if (f < 0)
  //       f = -f;
  //     if (fi2 > 0) {
  //       f = f * g_F;
  //     }
  //     else {
  //       f = g_F2 - g_F;
  //     }
  //     gl_FragColor = f * BaseColor;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %fi %fi2 %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %f "f"
OpName %fi "fi"
OpName %fi2 "fi2"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpMemberName %U_t 1 "g_F2"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%fi2 = OpVariable %_ptr_Input_float Input
%U_t = OpTypeStruct %float %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%26 = OpLoad %float %fi
OpStore %f %26
%27 = OpLoad %float %f
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %float %f
%32 = OpFNegate %float %31
OpStore %f %32
OpBranch %29
%29 = OpLabel
%33 = OpLoad %float %fi2
%34 = OpFOrdGreaterThan %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
%38 = OpLoad %float %f
%39 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%40 = OpLoad %float %39
%41 = OpFMul %float %38 %40
OpStore %f %41
OpBranch %35
%37 = OpLabel
%42 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%43 = OpLoad %float %42
%44 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%45 = OpLoad %float %44
%46 = OpFSub %float %43 %45
OpStore %f %46
OpBranch %35
%35 = OpLabel
%47 = OpLoad %v4float %BaseColor
%48 = OpLoad %float %f
%49 = OpVectorTimesScalar %v4float %47 %48
OpStore %gl_FragColor %49
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%26 = OpLoad %float %fi
OpStore %f %26
%27 = OpLoad %float %f
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %float %f
%32 = OpFNegate %float %31
OpStore %f %32
OpBranch %29
%29 = OpLabel
%56 = OpLoad %U_t %_
%57 = OpCompositeExtract %float %56 0
%33 = OpLoad %float %fi2
%34 = OpFOrdGreaterThan %bool %33 %float_0
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %37
%36 = OpLabel
%38 = OpLoad %float %f
%41 = OpFMul %float %38 %57
OpStore %f %41
OpBranch %35
%37 = OpLabel
%53 = OpCompositeExtract %float %56 1
%46 = OpFSub %float %53 %57
OpStore %f %46
OpBranch %35
%35 = OpLabel
%47 = OpLoad %v4float %BaseColor
%48 = OpLoad %float %f
%49 = OpVectorTimesScalar %v4float %47 %48
OpStore %gl_FragColor %49
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CommonUniformElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Basic3) {
  // Note: This test exemplifies the following:
  // - Existing common uniform (%_) load kept in place and shared
  //
  // #version 140
  // in vec4 BaseColor;
  // in float fi;
  // 
  // layout(std140) uniform U_t
  // {
  //     bool g_B;
  //     float g_F;
  // } ;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (g_B)
  //       v = v * g_F;
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor %fi
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpMemberName %U_t 1 "g_F"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
OpName %fi "fi"
OpMemberDecorate %U_t 0 Offset 0
OpMemberDecorate %U_t 1 Offset 4
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
%U_t = OpTypeStruct %uint %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%int_1 = OpConstant %int 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%28 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%29 = OpLoad %uint %28
%30 = OpINotEqual %bool %29 %uint_0
OpSelectionMerge %31 None
OpBranchConditional %30 %32 %31
%32 = OpLabel
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Uniform_float %_ %int_1
%35 = OpLoad %float %34
%36 = OpVectorTimesScalar %v4float %33 %35
OpStore %v %36
OpBranch %31
%31 = OpLabel
%37 = OpLoad %v4float %v
OpStore %gl_FragColor %37
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%26 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%27 = OpLoad %v4float %BaseColor
OpStore %v %27
%38 = OpLoad %U_t %_
%39 = OpCompositeExtract %uint %38 0
%30 = OpINotEqual %bool %39 %uint_0
OpSelectionMerge %31 None
OpBranchConditional %30 %32 %31
%32 = OpLabel
%33 = OpLoad %v4float %v
%41 = OpCompositeExtract %float %38 1
%36 = OpVectorTimesScalar %v4float %33 %41
OpStore %v %36
OpBranch %31
%31 = OpLabel
%37 = OpLoad %v4float %v
OpStore %gl_FragColor %37
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CommonUniformElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(CommonUniformElimTest, Loop) {
  // Note: This test exemplifies the following:
  // - Common extract (g_F) shared between two loops
  // #version 140
  // in vec4 BC;
  // in vec4 BC2;
  // 
  // layout(std140) uniform U_t
  // {
  //     float g_F;
  // } ;
  // 
  // void main()
  // {
  //     vec4 v = BC;
  //     for (int i = 0; i < 4; i++)
  //       v[i] = v[i] / g_F;
  //     vec4 v2 = BC2;
  //     for (int i = 0; i < 4; i++)
  //       v2[i] = v2[i] * g_F;
  //     gl_FragColor = v + v2;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BC %BC2 %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BC "BC"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_F"
OpName %_ ""
OpName %v2 "v2"
OpName %BC2 "BC2"
OpName %i_0 "i"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BC = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_4 = OpConstant %int 4
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%U_t = OpTypeStruct %float
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%int_1 = OpConstant %int 1
%BC2 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %13
%28 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%v2 = OpVariable %_ptr_Function_v4float Function
%i_0 = OpVariable %_ptr_Function_int Function
%29 = OpLoad %v4float %BC
OpStore %v %29
OpStore %i %int_0
OpBranch %30
%30 = OpLabel
OpLoopMerge %31 %32 None
OpBranch %33
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSLessThan %bool %34 %int_4
OpBranchConditional %35 %36 %31
%36 = OpLabel
%37 = OpLoad %int %i
%38 = OpLoad %int %i
%39 = OpAccessChain %_ptr_Function_float %v %38
%40 = OpLoad %float %39
%41 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%42 = OpLoad %float %41
%43 = OpFDiv %float %40 %42
%44 = OpAccessChain %_ptr_Function_float %v %37
OpStore %44 %43
OpBranch %32
%32 = OpLabel
%45 = OpLoad %int %i
%46 = OpIAdd %int %45 %int_1
OpStore %i %46
OpBranch %30
%31 = OpLabel
%47 = OpLoad %v4float %BC2
OpStore %v2 %47
OpStore %i_0 %int_0
OpBranch %48
%48 = OpLabel
OpLoopMerge %49 %50 None
OpBranch %51
%51 = OpLabel
%52 = OpLoad %int %i_0
%53 = OpSLessThan %bool %52 %int_4
OpBranchConditional %53 %54 %49
%54 = OpLabel
%55 = OpLoad %int %i_0
%56 = OpLoad %int %i_0
%57 = OpAccessChain %_ptr_Function_float %v2 %56
%58 = OpLoad %float %57
%59 = OpAccessChain %_ptr_Uniform_float %_ %int_0
%60 = OpLoad %float %59
%61 = OpFMul %float %58 %60
%62 = OpAccessChain %_ptr_Function_float %v2 %55
OpStore %62 %61
OpBranch %50
%50 = OpLabel
%63 = OpLoad %int %i_0
%64 = OpIAdd %int %63 %int_1
OpStore %i_0 %64
OpBranch %48
%49 = OpLabel
%65 = OpLoad %v4float %v
%66 = OpLoad %v4float %v2
%67 = OpFAdd %v4float %65 %66
OpStore %gl_FragColor %67
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %13
%28 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%v2 = OpVariable %_ptr_Function_v4float Function
%i_0 = OpVariable %_ptr_Function_int Function
%72 = OpLoad %U_t %_
%73 = OpCompositeExtract %float %72 0
%29 = OpLoad %v4float %BC
OpStore %v %29
OpStore %i %int_0
OpBranch %30
%30 = OpLabel
OpLoopMerge %31 %32 None
OpBranch %33
%33 = OpLabel
%34 = OpLoad %int %i
%35 = OpSLessThan %bool %34 %int_4
OpBranchConditional %35 %36 %31
%36 = OpLabel
%37 = OpLoad %int %i
%38 = OpLoad %int %i
%39 = OpAccessChain %_ptr_Function_float %v %38
%40 = OpLoad %float %39
%43 = OpFDiv %float %40 %73
%44 = OpAccessChain %_ptr_Function_float %v %37
OpStore %44 %43
OpBranch %32
%32 = OpLabel
%45 = OpLoad %int %i
%46 = OpIAdd %int %45 %int_1
OpStore %i %46
OpBranch %30
%31 = OpLabel
%47 = OpLoad %v4float %BC2
OpStore %v2 %47
OpStore %i_0 %int_0
OpBranch %48
%48 = OpLabel
OpLoopMerge %49 %50 None
OpBranch %51
%51 = OpLabel
%52 = OpLoad %int %i_0
%53 = OpSLessThan %bool %52 %int_4
OpBranchConditional %53 %54 %49
%54 = OpLabel
%55 = OpLoad %int %i_0
%56 = OpLoad %int %i_0
%57 = OpAccessChain %_ptr_Function_float %v2 %56
%58 = OpLoad %float %57
%61 = OpFMul %float %58 %73
%62 = OpAccessChain %_ptr_Function_float %v2 %55
OpStore %62 %61
OpBranch %50
%50 = OpLabel
%63 = OpLoad %int %i_0
%64 = OpIAdd %int %63 %int_1
OpStore %i_0 %64
OpBranch %48
%49 = OpLabel
%65 = OpLoad %v4float %v
%66 = OpLoad %v4float %v2
%67 = OpFAdd %v4float %65 %66
OpStore %gl_FragColor %67
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CommonUniformElimPass>(
      predefs + before, predefs + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Disqualifying cases: extensions, decorations, non-logical addressing,
//      non-structured control flow
//    Others?

}  // anonymous namespace

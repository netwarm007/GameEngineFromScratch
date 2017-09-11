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

template <typename T>
std::vector<T> concat(const std::vector<T>& a, const std::vector<T>& b) {
  std::vector<T> ret;
  std::copy(a.begin(), a.end(), back_inserter(ret));
  std::copy(b.begin(), b.end(), back_inserter(ret));
  return ret;
}

namespace {

using namespace spvtools;

using LocalSingleBlockLoadStoreElimTest = PassTest<::testing::Test>;

TEST_F(LocalSingleBlockLoadStoreElimTest, SimpleStoreLoadElim) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
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

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
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
%15 = OpLoad %v4float %v
OpStore %gl_FragColor %15
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%14 = OpLoad %v4float %BaseColor
OpStore %gl_FragColor %14
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleBlockLoadStoreElimPass>(
      predefs_before + before, predefs_after + after, true, true);
}

TEST_F(LocalSingleBlockLoadStoreElimTest, SimpleLoadLoadElim) {
  // #version 140
  //
  // in vec4 BaseColor;
  // in float fi;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (fi < 0)
  //         v = vec4(0.0);
  //     gl_FragData[0] = v;
  //     gl_FragData[1] = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragData
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %fi "fi"
OpName %gl_FragData "gl_FragData"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%16 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%_arr_v4float_uint_32 = OpTypeArray %v4float %uint_32
%_ptr_Output__arr_v4float_uint_32 = OpTypePointer Output %_arr_v4float_uint_32
%gl_FragData = OpVariable %_ptr_Output__arr_v4float_uint_32 Output
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%int_1 = OpConstant %int 1
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%25 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%26 = OpLoad %v4float %BaseColor
OpStore %v %26
%27 = OpLoad %float %fi
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpStore %v %16
OpBranch %29
%29 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_0
OpStore %32 %31
%33 = OpLoad %v4float %v
%34 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_1
OpStore %34 %33
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%25 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%26 = OpLoad %v4float %BaseColor
OpStore %v %26
%27 = OpLoad %float %fi
%28 = OpFOrdLessThan %bool %27 %float_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpStore %v %16
OpBranch %29
%29 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_0
OpStore %32 %31
%34 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_1
OpStore %34 %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleBlockLoadStoreElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(LocalSingleBlockLoadStoreElimTest,
       NoStoreElimIfInterveningAccessChainLoad) {
  //
  // Note that even though the Load to %v is eliminated, the Store to %v
  // is not eliminated due to the following access chain reference.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // flat in int Idx;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     float f = v[Idx];
  //     gl_FragColor = v/f;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Idx %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %Idx "Idx"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %Idx Flat
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_float = OpTypePointer Function %float
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%Idx = OpVariable %_ptr_Input_int Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Idx %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Idx "Idx"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %Idx Flat
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_float = OpTypePointer Function %float
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%Idx = OpVariable %_ptr_Input_int Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f = OpVariable %_ptr_Function_float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
%20 = OpLoad %int %Idx
%21 = OpAccessChain %_ptr_Function_float %v %20
%22 = OpLoad %float %21
OpStore %f %22
%23 = OpLoad %v4float %v
%24 = OpLoad %float %f
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
%26 = OpFDiv %v4float %23 %25
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
%20 = OpLoad %int %Idx
%21 = OpAccessChain %_ptr_Function_float %v %20
%22 = OpLoad %float %21
%25 = OpCompositeConstruct %v4float %22 %22 %22 %22
%26 = OpFDiv %v4float %19 %25
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleBlockLoadStoreElimPass>(
      predefs_before + before, predefs_after + after, true, true);
}

TEST_F(LocalSingleBlockLoadStoreElimTest, NoElimIfInterveningAccessChainStore) {
  // #version 140
  //
  // in vec4 BaseColor;
  // flat in int Idx;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     v[Idx] = 0;
  //     gl_FragColor = v;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Idx %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Idx "Idx"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %Idx Flat
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%Idx = OpVariable %_ptr_Input_int Input
%float_0 = OpConstant %float 0
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%18 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %v %19
%20 = OpLoad %int %Idx
%21 = OpAccessChain %_ptr_Function_float %v %20
OpStore %21 %float_0
%22 = OpLoad %v4float %v
OpStore %gl_FragColor %22
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleBlockLoadStoreElimPass>(
      assembly, assembly, false, true);
}

TEST_F(LocalSingleBlockLoadStoreElimTest, NoElimIfInterveningFunctionCall) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void foo() {
  // }
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     foo();
  //     gl_FragColor = v;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_ "foo("
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
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%14 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%15 = OpLoad %v4float %BaseColor
OpStore %v %15
%16 = OpFunctionCall %void %foo_
%17 = OpLoad %v4float %v
OpStore %gl_FragColor %17
OpReturn
OpFunctionEnd
%foo_ = OpFunction %void None %8
%18 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleBlockLoadStoreElimPass>(
      assembly, assembly, false, true);
}

TEST_F(LocalSingleBlockLoadStoreElimTest, ElimIfCopyObjectInFunction) {
  // Note: SPIR-V hand edited to insert CopyObject
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //   vec4 v1 = BaseColor;
  //   gl_FragData[0] = v1;
  //   vec4 v2 = BaseColor * 0.5;
  //   gl_FragData[1] = v2;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragData
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v1 "v1"
OpName %BaseColor "BaseColor"
OpName %gl_FragData "gl_FragData"
OpName %v2 "v2"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%_arr_v4float_uint_32 = OpTypeArray %v4float %uint_32
%_ptr_Output__arr_v4float_uint_32 = OpTypePointer Output %_arr_v4float_uint_32
%gl_FragData = OpVariable %_ptr_Output__arr_v4float_uint_32 Output
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%float_0_5 = OpConstant %float 0.5
%int_1 = OpConstant %int 1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragData
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %gl_FragData "gl_FragData"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%_arr_v4float_uint_32 = OpTypeArray %v4float %uint_32
%_ptr_Output__arr_v4float_uint_32 = OpTypePointer Output %_arr_v4float_uint_32
%gl_FragData = OpVariable %_ptr_Output__arr_v4float_uint_32 Output
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%float_0_5 = OpConstant %float 0.5
%int_1 = OpConstant %int 1
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%v1 = OpVariable %_ptr_Function_v4float Function
%v2 = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %v1 %23
%24 = OpLoad %v4float %v1
%25 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_0
OpStore %25 %24
%26 = OpLoad %v4float %BaseColor
%27 = OpVectorTimesScalar %v4float %26 %float_0_5
%28 = OpCopyObject %_ptr_Function_v4float %v2
OpStore %28 %27
%29 = OpLoad %v4float %28
%30 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_1
OpStore %30 %29
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%23 = OpLoad %v4float %BaseColor
%25 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_0
OpStore %25 %23
%26 = OpLoad %v4float %BaseColor
%27 = OpVectorTimesScalar %v4float %26 %float_0_5
%30 = OpAccessChain %_ptr_Output_v4float %gl_FragData %int_1
OpStore %30 %27
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleBlockLoadStoreElimPass>(
      predefs_before + before, predefs_after + after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Other target variable types
//    InBounds Access Chains
//    Check for correctness in the presence of function calls
//    Others?

}  // anonymous namespace

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

// Tests for unique type declaration rules validator.

#include <functional>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "spirv-tools/markv.h"
#include "test_fixture.h"
#include "unit_spirv.h"

namespace {

using spvtest::ScopedContext;

void DiagnosticsMessageHandler(spv_message_level_t level, const char*,
                               const spv_position_t& position,
                               const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: " << position.index << ": " << message << std::endl;
      break;
    default:
      break;
  }
}

// Compiles |code| to SPIR-V |words|.
void Compile(const std::string& code, std::vector<uint32_t>* words,
             uint32_t options = SPV_TEXT_TO_BINARY_OPTION_NONE,
             spv_target_env env = SPV_ENV_UNIVERSAL_1_2) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_binary spirv_binary;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinaryWithOptions(
      ctx.context, code.c_str(), code.size(), options, &spirv_binary, nullptr));

  *words = std::vector<uint32_t>(
      spirv_binary->code, spirv_binary->code + spirv_binary->wordCount);

  spvBinaryDestroy(spirv_binary);
}

// Disassembles SPIR-V |words| to |out_text|.
void Disassemble(const std::vector<uint32_t>& words,
                 std::string* out_text,
                 spv_target_env env = SPV_ENV_UNIVERSAL_1_2) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_text text = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryToText(ctx.context, words.data(),
                                         words.size(), 0, &text, nullptr));
  assert(text);

  *out_text = std::string(text->str, text->length);
  spvTextDestroy(text);
}

// Encodes SPIR-V |words| to |markv_binary|. |comments| context snippets of
// disassembly and bit sequences for debugging.
void Encode(const std::vector<uint32_t>& words,
            spv_markv_binary* markv_binary,
            std::string* comments,
            spv_target_env env = SPV_ENV_UNIVERSAL_1_2) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  std::unique_ptr<spv_markv_encoder_options_t,
      std::function<void(spv_markv_encoder_options_t*)>> options(
          spvMarkvEncoderOptionsCreate(), &spvMarkvEncoderOptionsDestroy);
  spv_text spv_text_comments;
  ASSERT_EQ(SPV_SUCCESS, spvSpirvToMarkv(ctx.context, words.data(),
                                         words.size(), options.get(),
                                         markv_binary, &spv_text_comments,
                                         nullptr));

  *comments = std::string(spv_text_comments->str, spv_text_comments->length);
  spvTextDestroy(spv_text_comments);
}

// Decodes |markv_binary| to SPIR-V |words|.
void Decode(const spv_markv_binary markv_binary,
            std::vector<uint32_t>* words,
            spv_target_env env = SPV_ENV_UNIVERSAL_1_2) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_binary spirv_binary = nullptr;
  std::unique_ptr<spv_markv_decoder_options_t,
      std::function<void(spv_markv_decoder_options_t*)>> options(
          spvMarkvDecoderOptionsCreate(), &spvMarkvDecoderOptionsDestroy);
  ASSERT_EQ(SPV_SUCCESS, spvMarkvToSpirv(ctx.context, markv_binary->data,
                                         markv_binary->length, options.get(),
                                         &spirv_binary, nullptr, nullptr));

  *words = std::vector<uint32_t>(
      spirv_binary->code, spirv_binary->code + spirv_binary->wordCount);

  spvBinaryDestroy(spirv_binary);
}

// Encodes/decodes |original|, assembles/dissasembles |original|, then compares
// the results of the two operations.
void TestEncodeDecode(const std::string& original_text) {
  std::vector<uint32_t> expected_binary;
  Compile(original_text, &expected_binary);
  ASSERT_FALSE(expected_binary.empty());

  std::string expected_text;
  Disassemble(expected_binary, &expected_text);
  ASSERT_FALSE(expected_text.empty());

  std::vector<uint32_t> binary_to_encode;
  Compile(original_text, &binary_to_encode,
          SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_FALSE(binary_to_encode.empty());

  spv_markv_binary markv_binary = nullptr;
  std::string encoder_comments;
  Encode(binary_to_encode, &markv_binary, &encoder_comments);
  ASSERT_NE(nullptr, markv_binary);

  // std::cerr << encoder_comments << std::endl;
  // std::cerr << "SPIR-V size: " << expected_binary.size() * 4 << std::endl;
  // std::cerr << "MARK-V size: " << markv_binary->length << std::endl;

  std::vector<uint32_t> decoded_binary;
  Decode(markv_binary, &decoded_binary);
  ASSERT_FALSE(decoded_binary.empty());

  EXPECT_EQ(expected_binary, decoded_binary) << encoder_comments;

  std::string decoded_text;
  Disassemble(decoded_binary, &decoded_text);
  ASSERT_FALSE(decoded_text.empty());

  EXPECT_EQ(expected_text, decoded_text) << encoder_comments;

  spvMarkvBinaryDestroy(markv_binary);
}

TEST(Markv, U32Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%u32 = OpTypeInt 32 0
%100 = OpConstant %u32 0
%200 = OpConstant %u32 1
%300 = OpConstant %u32 4294967295
)");
}

TEST(Markv, S32Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%s32 = OpTypeInt 32 1
%100 = OpConstant %s32 0
%200 = OpConstant %s32 1
%300 = OpConstant %s32 -1
%400 = OpConstant %s32 2147483647
%500 = OpConstant %s32 -2147483648
)");
}

TEST(Markv, U64Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
%u64 = OpTypeInt 64 0
%100 = OpConstant %u64 0
%200 = OpConstant %u64 1
%300 = OpConstant %u64 18446744073709551615
)");
}

TEST(Markv, S64Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
%s64 = OpTypeInt 64 1
%100 = OpConstant %s64 0
%200 = OpConstant %s64 1
%300 = OpConstant %s64 -1
%400 = OpConstant %s64 9223372036854775807
%500 = OpConstant %s64 -9223372036854775808
)");
}

TEST(Markv, U16Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpMemoryModel Logical GLSL450
%u16 = OpTypeInt 16 0
%100 = OpConstant %u16 0
%200 = OpConstant %u16 1
%300 = OpConstant %u16 65535
)");
}

TEST(Markv, S16Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpMemoryModel Logical GLSL450
%s16 = OpTypeInt 16 1
%100 = OpConstant %s16 0
%200 = OpConstant %s16 1
%300 = OpConstant %s16 -1
%400 = OpConstant %s16 32767
%500 = OpConstant %s16 -32768
)");
}

TEST(Markv, F32Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%f32 = OpTypeFloat 32
%100 = OpConstant %f32 0
%200 = OpConstant %f32 1
%300 = OpConstant %f32 0.1
%400 = OpConstant %f32 -0.1
)");
}

TEST(Markv, F64Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Float64
OpMemoryModel Logical GLSL450
%f64 = OpTypeFloat 64
%100 = OpConstant %f64 0
%200 = OpConstant %f64 1
%300 = OpConstant %f64 0.1
%400 = OpConstant %f64 -0.1
)");
}

TEST(Markv, F16Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Float16
OpMemoryModel Logical GLSL450
%f16 = OpTypeFloat 16
%100 = OpConstant %f16 0
%200 = OpConstant %f16 1
%300 = OpConstant %f16 0.1
%400 = OpConstant %f16 -0.1
)");
}

TEST(Markv, StringLiteral) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpExtension "xxx"
OpExtension "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OpExtension ""
OpMemoryModel Logical GLSL450
)");
}

TEST(Markv, WithFunction) {
  TestEncodeDecode(R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Physical32 OpenCL
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%100 = OpConstant %u32 1
%200 = OpConstant %u32 2
%main = OpFunction %void None %void_func
%entry_main = OpLabel
%300 = OpIAdd %u32 %100 %200
OpReturn
OpFunctionEnd
)");
}

TEST(Markv, ForwardDeclaredId) {
  TestEncodeDecode(R"(
OpCapability Addresses
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
)");
}

TEST(Markv, WithSwitch) {
  TestEncodeDecode(R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpCapability Int64
OpMemoryModel Physical32 OpenCL
%u64 = OpTypeInt 64 0
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%val = OpConstant %u64 1
%main = OpFunction %void None %void_func
%entry_main = OpLabel
OpSwitch %val %default 1 %case1 1000000000000 %case2
%case1 = OpLabel
OpNop
OpBranch %after_switch
%case2 = OpLabel
OpNop
OpBranch %after_switch
%default = OpLabel
OpNop
OpBranch %after_switch
%after_switch = OpLabel
OpReturn
OpFunctionEnd
)");
}

TEST(Markv, WithLoop) {
  TestEncodeDecode(R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%entry_main = OpLabel
OpLoopMerge %merge %continue DontUnroll|DependencyLength 10
OpBranch %begin_loop
%begin_loop = OpLabel
OpNop
OpBranch %continue
%continue = OpLabel
OpNop
OpBranch %begin_loop
%merge = OpLabel
OpReturn
OpFunctionEnd
)");
}

TEST(Markv, WithDecorate) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 ArrayStride 4
OpDecorate %1 Uniform
%2 = OpTypeFloat 32
%1 = OpTypeRuntimeArray %2
)");
}

TEST(Markv, WithExtInst) {
  TestEncodeDecode(R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
%opencl = OpExtInstImport "OpenCL.std"
OpMemoryModel Physical32 OpenCL
%f32 = OpTypeFloat 32
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%100 = OpConstant %f32 1.1
%main = OpFunction %void None %void_func
%entry_main = OpLabel
%200 = OpExtInst %f32 %opencl cos %100
OpReturn
OpFunctionEnd
)");
}

}  // namespace

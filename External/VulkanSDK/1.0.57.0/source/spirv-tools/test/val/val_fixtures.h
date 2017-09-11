// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Common validation fixtures for unit tests

#ifndef LIBSPIRV_TEST_VALIDATE_FIXTURES_H_
#define LIBSPIRV_TEST_VALIDATE_FIXTURES_H_

#include "unit_spirv.h"
#include "source/val/validation_state.h"

namespace spvtest {

template <typename T>
class ValidateBase : public ::testing::Test,
                     public ::testing::WithParamInterface<T> {
 public:
  ValidateBase();

  virtual void TearDown();

  // Returns the a spv_const_binary struct
  spv_const_binary get_const_binary();

  void CompileSuccessfully(std::string code,
                           spv_target_env env = SPV_ENV_UNIVERSAL_1_0);

  // Overwrites the word at index 'index' with the given word.
  // For testing purposes, it is often useful to be able to manipulate the
  // assembled binary before running the validator on it.
  // This function overwrites the word at the given index with a new word.
  void OverwriteAssembledBinary(uint32_t index, uint32_t word);

  // Performs validation on the SPIR-V code and compares the result of the
  // spvValidate function
  spv_result_t ValidateInstructions(spv_target_env env = SPV_ENV_UNIVERSAL_1_0);

  // Performs validation. Returns the status and stores validation state into
  // the vstate_ member.
  spv_result_t ValidateAndRetrieveValidationState(
      spv_target_env env = SPV_ENV_UNIVERSAL_1_0);

  std::string getDiagnosticString();
  spv_position_t getErrorPosition();
  spv_validator_options getValidatorOptions();

  spv_binary binary_;
  spv_diagnostic diagnostic_;
  spv_validator_options options_;
  std::unique_ptr<libspirv::ValidationState_t> vstate_;
};
}
#endif

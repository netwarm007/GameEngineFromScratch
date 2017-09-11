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

#include "val_fixtures.h"

#include <functional>
#include <tuple>
#include <utility>

#include "test_fixture.h"

namespace spvtest {

template <typename T>
ValidateBase<T>::ValidateBase() : binary_(), diagnostic_() {
  // Initialize to default command line options. Different tests can then
  // specialize specific options as necessary.
  options_ = spvValidatorOptionsCreate();
}

template <typename T>
spv_const_binary ValidateBase<T>::get_const_binary() {
  return spv_const_binary(binary_);
}

template <typename T>
void ValidateBase<T>::TearDown() {
  if (diagnostic_) {
    spvDiagnosticPrint(diagnostic_);
  }
  spvDiagnosticDestroy(diagnostic_);
  spvBinaryDestroy(binary_);
  spvValidatorOptionsDestroy(options_);
}

template <typename T>
void ValidateBase<T>::CompileSuccessfully(std::string code,
                                          spv_target_env env) {
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS,
            spvTextToBinary(ScopedContext(env).context, code.c_str(),
                            code.size(), &binary_, &diagnostic))
      << "ERROR: " << diagnostic->error
      << "\nSPIR-V could not be compiled into binary:\n"
      << code;
}

template <typename T>
void ValidateBase<T>::OverwriteAssembledBinary(uint32_t index, uint32_t word) {
  ASSERT_TRUE(index < binary_->wordCount)
      << "OverwriteAssembledBinary: The given index is larger than the binary "
         "word count.";
  binary_->code[index] = word;
}

template <typename T>
spv_result_t ValidateBase<T>::ValidateInstructions(spv_target_env env) {
  return spvValidateWithOptions(ScopedContext(env).context, options_,
                                get_const_binary(), &diagnostic_);
}

template <typename T>
spv_result_t ValidateBase<T>::ValidateAndRetrieveValidationState(
    spv_target_env env) {
  return spvtools::ValidateBinaryAndKeepValidationState(
      ScopedContext(env).context, options_, get_const_binary()->code,
      get_const_binary()->wordCount, &diagnostic_, &vstate_);
}

template <typename T>
std::string ValidateBase<T>::getDiagnosticString() {
  return diagnostic_ == nullptr ?
      std::string() : std::string(diagnostic_->error);
}

template <typename T>
spv_validator_options ValidateBase<T>::getValidatorOptions() {
  return options_;
}

template <typename T>
spv_position_t ValidateBase<T>::getErrorPosition() {
  return diagnostic_ == nullptr ? spv_position_t() : diagnostic_->position;
}

template class spvtest::ValidateBase<bool>;
template class spvtest::ValidateBase<int>;
template class spvtest::ValidateBase<std::string>;
template class spvtest::ValidateBase<std::pair<std::string, bool>>;
template class spvtest::ValidateBase<
    std::tuple<std::string, std::pair<std::string, std::vector<std::string>>>>;
template class spvtest::ValidateBase<
    std::tuple<int, std::tuple<std::string, std::function<spv_result_t(int)>,
                               std::function<spv_result_t(int)>>>>;
template class spvtest::ValidateBase<SpvCapability>;
template class spvtest::ValidateBase<std::pair<std::string, std::string>>;
}

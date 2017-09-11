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

#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "opt/set_spec_constant_default_value_pass.h"
#include "spirv-tools/optimizer.hpp"

#include "message.h"
#include "tools/io.h"

using namespace spvtools;

void PrintUsage(const char* program) {
  printf(
      R"(%s - Optimize a SPIR-V binary file.

USAGE: %s [options] [<input>] -o <output>

The SPIR-V binary is read from <input>. If no file is specified,
or if <input> is "-", then the binary is read from standard input.
if <output> is "-", then the optimized output is written to
standard output.

NOTE: The optimizer is a work in progress.

Options:
  --strip-debug
               Remove all debug instructions.
  --freeze-spec-const
               Freeze the values of specialization constants to their default
               values.
  --eliminate-dead-const
               Eliminate dead constants.
  --fold-spec-const-op-composite
               Fold the spec constants defined by OpSpecConstantOp or
               OpSpecConstantComposite instructions to front-end constants
               when possible.
  --set-spec-const-default-value "<spec id>:<default value> ..."
               Set the default values of the specialization constants with
               <spec id>:<default value> pairs specified in a double-quoted
               string. <spec id>:<default value> pairs must be separated by
               blank spaces, and in each pair, spec id and default value must
               be separated with colon ':' without any blank spaces in between.
               e.g.: --set-spec-const-default-value "1:100 2:400"
  --unify-const
               Remove the duplicated constants.
  --flatten-decorations
               Replace decoration groups with repeated OpDecorate and
               OpMemberDecorate instructions.
  --compact-ids
               Remap result ids to a compact range starting from %%1 and without
               any gaps.
  --inline-entry-points-exhaustive
               Exhaustively inline all function calls in entry point functions.
               Currently does not inline calls to functions with early return
               in a loop.
  --convert-local-access-chains
               Convert constant index access chain loads/stores into
               equivalent load/stores with inserts and extracts. Performed
               on function scope variables referenced only with load, store,
               and constant index access chains.
  --eliminate-common-uniform
               Perform load/load elimination for duplicate uniform values.
               Converts any constant index access chain uniform loads into
               its equivalent load and extract. Some loads will be moved
               to facilitate sharing. Performed only on entry point
               functions.
  --eliminate-local-single-block
               Perform single-block store/load and load/load elimination.
               Performed only on function scope variables in entry point
               functions.
  --eliminate-local-single-store
               Replace stores and loads of function scope variables that are
               only stored once. Performed on variables referenceed only with
               loads and stores. Performed only on entry point functions.
  --eliminate-local-multi-store
               Replace stores and loads of function scope variables that are
               stored multiple times. Performed on variables referenceed only
               with loads and stores. Performed only on entry point functions.
  --eliminate-insert-extract
               Replace extract from a sequence of inserts with the
               corresponding value. Performed only on entry point functions.
  --eliminate-dead-code-aggressive
               Delete instructions which do not contribute to a function's
               output. Performed only on entry point functions.
  --eliminate-dead-branches
               Convert conditional branches with constant condition to the
               indicated unconditional brranch. Delete all resulting dead
               code. Performed only on entry point functions.
  --merge-blocks
               Join two blocks into a single block if the second has the
               first as its only predecessor. Performed only on entry point
               functions.
  -h, --help   
               Print this help.
  --version    
               Display optimizer version information.
)",
      program, program);
}

int main(int argc, char** argv) {
  const char* in_file = nullptr;
  const char* out_file = nullptr;

  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_2;

  spvtools::Optimizer optimizer(target_env);
  optimizer.SetMessageConsumer([](spv_message_level_t level, const char* source,
                                  const spv_position_t& position,
                                  const char* message) {
    std::cerr << StringifyMessage(level, source, position, message)
              << std::endl;
  });

  for (int argi = 1; argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--version")) {
        printf("%s\n", spvSoftwareVersionDetailsString());
        return 0;
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        return 0;
      } else if (0 == strcmp(cur_arg, "-o")) {
        if (!out_file && argi + 1 < argc) {
          out_file = argv[++argi];
        } else {
          PrintUsage(argv[0]);
          return 1;
        }
      } else if (0 == strcmp(cur_arg, "--strip-debug")) {
        optimizer.RegisterPass(CreateStripDebugInfoPass());
      } else if (0 == strcmp(cur_arg, "--set-spec-const-default-value")) {
        if (++argi < argc) {
          auto spec_ids_vals =
              opt::SetSpecConstantDefaultValuePass::ParseDefaultValuesString(
                  argv[argi]);
          if (!spec_ids_vals) {
            fprintf(stderr,
                    "error: Invalid argument for "
                    "--set-spec-const-default-value: %s\n",
                    argv[argi]);
            return 1;
          }
          optimizer.RegisterPass(
              CreateSetSpecConstantDefaultValuePass(std::move(*spec_ids_vals)));
        } else {
          fprintf(
              stderr,
              "error: Expected a string of <spec id>:<default value> pairs.");
          return 1;
        }
      } else if (0 == strcmp(cur_arg, "--freeze-spec-const")) {
        optimizer.RegisterPass(CreateFreezeSpecConstantValuePass());
      } else if (0 == strcmp(cur_arg, "--inline-entry-points-exhaustive")) {
        optimizer.RegisterPass(CreateInlineExhaustivePass());
      } else if (0 == strcmp(cur_arg, "--convert-local-access-chains")) {
        optimizer.RegisterPass(CreateLocalAccessChainConvertPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-code-aggressive")) {
        optimizer.RegisterPass(CreateAggressiveDCEPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-insert-extract")) {
        optimizer.RegisterPass(CreateInsertExtractElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-local-single-block")) {
        optimizer.RegisterPass(CreateLocalSingleBlockLoadStoreElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-local-single-store")) {
        optimizer.RegisterPass(CreateLocalSingleStoreElimPass());
      } else if (0 == strcmp(cur_arg, "--merge-blocks")) {
        optimizer.RegisterPass(CreateBlockMergePass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-branches")) {
        optimizer.RegisterPass(CreateDeadBranchElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-local-multi-store")) {
        optimizer.RegisterPass(CreateLocalMultiStoreElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-common-uniform")) {
        optimizer.RegisterPass(CreateCommonUniformElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-const")) {
        optimizer.RegisterPass(CreateEliminateDeadConstantPass());
      } else if (0 == strcmp(cur_arg, "--fold-spec-const-op-composite")) {
        optimizer.RegisterPass(CreateFoldSpecConstantOpAndCompositePass());
      } else if (0 == strcmp(cur_arg, "--unify-const")) {
        optimizer.RegisterPass(CreateUnifyConstantPass());
      } else if (0 == strcmp(cur_arg, "--flatten-decorations")) {
        optimizer.RegisterPass(CreateFlattenDecorationPass());
      } else if (0 == strcmp(cur_arg, "--compact-ids")) {
        optimizer.RegisterPass(CreateCompactIdsPass());
      } else if ('\0' == cur_arg[1]) {
        // Setting a filename of "-" to indicate stdin.
        if (!in_file) {
          in_file = cur_arg;
        } else {
          fprintf(stderr, "error: More than one input file specified\n");
          return 1;
        }
      } else {
        PrintUsage(argv[0]);
        return 1;
      }
    } else {
      if (!in_file) {
        in_file = cur_arg;
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  if (out_file == nullptr) {
    fprintf(stderr, "error: -o required\n");
    return 1;
  }

  std::vector<uint32_t> binary;
  if (!ReadFile<uint32_t>(in_file, "rb", &binary)) return 1;

  // Let's do validation first.
  spv_context context = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t binary_struct = {binary.data(), binary.size()};
  spv_result_t error = spvValidate(context, &binary_struct, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
    return error;
  }
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  // By using the same vector as input and output, we save time in the case
  // that there was no change.
  bool ok = optimizer.Run(binary.data(), binary.size(), &binary);

  if (!WriteFile<uint32_t>(out_file, "wb", binary.data(), binary.size())) {
    return 1;
  }

  return ok ? 0 : 1;
}

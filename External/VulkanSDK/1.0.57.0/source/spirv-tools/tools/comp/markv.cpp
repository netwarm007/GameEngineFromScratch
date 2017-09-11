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

#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "source/spirv_target_env.h"
#include "source/table.h"
#include "spirv-tools/markv.h"
#include "tools/io.h"

namespace {

enum Task {
  kNoTask = 0,
  kEncode,
  kDecode,
};

struct ScopedContext {
  ScopedContext(spv_target_env env) : context(spvContextCreate(env)) {}
  ~ScopedContext() { spvContextDestroy(context); }
  spv_context context;
};

void print_usage(char* argv0) {
  printf(
      R"(%s - Encodes or decodes a SPIR-V binary to or from a MARK-V binary.

USAGE: %s [e|d] [options] [<filename>]

The input binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

If no output is specified then the output is printed to stdout in a human
readable format.

WIP: MARK-V codec is in early stages of development. At the moment it only
can encode and decode some SPIR-V files and only if exacly the same build of
software is used (is doesn't write or handle version numbers yet).

Tasks:
  e               Encode SPIR-V to MARK-V.
  d               Decode MARK-V to SPIR-V.

Options:
  -h, --help      Print this help.
  --comments      Write codec comments to stdout.
  --version       Display MARK-V codec version.

  -o <filename>   Set the output filename.
                  Output goes to standard output if this option is
                  not specified, or if the filename is "-".
)",
      argv0, argv0);
}

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

}  // namespace

int main(int argc, char** argv) {
  const char* input_filename = nullptr;
  const char* output_filename = nullptr;

  Task task = kNoTask;

  if (argc < 3) {
    print_usage(argv[0]);
    return 0;
  }

  const char* task_char = argv[1];
  if (0 == strcmp("e", task_char)) {
    task = kEncode;
  } else if (0 == strcmp("d", task_char)) {
    task = kDecode;
  }

  if (task == kNoTask) {
    print_usage(argv[0]);
    return 1;
  }

  bool want_comments = false;

  for (int argi = 2; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case 'o': {
          if (!output_filename && argi + 1 < argc) {
            output_filename = argv[++argi];
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case '-': {
          if (0 == strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
          } else if (0 == strcmp(argv[argi], "--comments")) {
            want_comments = true;
          } else if (0 == strcmp(argv[argi], "--version")) {
            fprintf(stderr, "error: Not implemented\n");
            return 1;
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case '\0': {
          // Setting a filename of "-" to indicate stdin.
          if (!input_filename) {
            input_filename = argv[argi];
          } else {
            fprintf(stderr, "error: More than one input file specified\n");
            return 1;
          }
        } break;
        default:
          print_usage(argv[0]);
          return 1;
      }
    } else {
      if (!input_filename) {
        input_filename = argv[argi];
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  if (task == kDecode && want_comments) {
    fprintf(stderr, "warning: Decoder comments not yet implemented\n");
    want_comments = false;
  }

  const bool write_to_stdout = output_filename == nullptr ||
      0 == strcmp(output_filename, "-");

  spv_text comments = nullptr;
  spv_text* comments_ptr = want_comments ? &comments : nullptr;

  ScopedContext ctx(SPV_ENV_UNIVERSAL_1_2);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  if (task == kEncode) {
    std::vector<uint32_t> contents;
    if (!ReadFile<uint32_t>(input_filename, "rb", &contents)) return 1;

    std::unique_ptr<spv_markv_encoder_options_t,
        std::function<void(spv_markv_encoder_options_t*)>> options(
            spvMarkvEncoderOptionsCreate(), &spvMarkvEncoderOptionsDestroy);
    spv_markv_binary markv_binary = nullptr;

    if (SPV_SUCCESS !=
        spvSpirvToMarkv(ctx.context, contents.data(), contents.size(),
                        options.get(), &markv_binary, comments_ptr, nullptr)) {
      std::cerr << "error: Failed to encode " << input_filename << " to MARK-V "
                << std::endl;
      return 1;
    }

    if (want_comments) {
      if (!WriteFile<char>(nullptr, "w", comments->str,
                           comments->length)) return 1;
    }

    if (!want_comments || !write_to_stdout) {
      if (!WriteFile<uint8_t>(output_filename, "wb", markv_binary->data,
                              markv_binary->length)) return 1;
    }
  } else if (task == kDecode) {
    std::vector<uint8_t> contents;
    if (!ReadFile<uint8_t>(input_filename, "rb", &contents)) return 1;

    std::unique_ptr<spv_markv_decoder_options_t,
        std::function<void(spv_markv_decoder_options_t*)>> options(
            spvMarkvDecoderOptionsCreate(), &spvMarkvDecoderOptionsDestroy);
    spv_binary spirv_binary = nullptr;

    if (SPV_SUCCESS !=
        spvMarkvToSpirv(ctx.context, contents.data(), contents.size(),
                        options.get(), &spirv_binary, comments_ptr, nullptr)) {
      std::cerr << "error: Failed to encode " << input_filename << " to MARK-V "
                << std::endl;
      return 1;
    }

    if (want_comments) {
      if (!WriteFile<char>(nullptr, "w", comments->str,
                           comments->length)) return 1;
    }

    if (!want_comments || !write_to_stdout) {
      if (!WriteFile<uint32_t>(output_filename, "wb", spirv_binary->code,
                              spirv_binary->wordCount)) return 1;
    }
  } else {
    assert(false && "Unknown task");
  }

  spvTextDestroy(comments);

  return 0;
}

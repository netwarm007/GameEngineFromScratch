# SPIR-V Tools

[![Build Status](https://travis-ci.org/KhronosGroup/SPIRV-Tools.svg?branch=master)](https://travis-ci.org/KhronosGroup/SPIRV-Tools)
[![Build status](https://ci.appveyor.com/api/projects/status/gpue87cesrx3pi0d/branch/master?svg=true)](https://ci.appveyor.com/project/Khronoswebmaster/spirv-tools/branch/master)

## Overview

The SPIR-V Tools project provides an API and commands for processing SPIR-V
modules.

The project includes an assembler, binary module parser, disassembler,
validator, and optimizer for SPIR-V. Except for the optimizer, all are based
on a common static library.  The library contains all of the implementation
details, and is used in the standalone tools whilst also enabling integration
into other code bases directly. The optimizer implementation resides in its
own library, which depends on the core library.

The interfaces have stabilized:
We don't anticipate making a breaking change for existing features.

See [`projects.md`](projects.md) to see how we use the
[GitHub Project
feature](https://help.github.com/articles/tracking-the-progress-of-your-work-with-projects/)
to organize planned and in-progress work.

SPIR-V is defined by the Khronos Group Inc.
See the [SPIR-V Registry][spirv-registry] for the SPIR-V specification,
headers, and XML registry.

## Verisoning SPIRV-Tools

See [`CHANGES`](CHANGES) for a high level summary of recent changes, by version.

SPIRV-Tools project version numbers are of the form `v`*year*`.`*index* and with
an optional `-dev` suffix to indicate work in progress.  For exampe, the
following versions are ordered from oldest to newest:

* `v2016.0`
* `v2016.1-dev`
* `v2016.1`
* `v2016.2-dev`
* `v2016.2`

Use the `--version` option on each command line tool to see the software
version.  An API call reports the software version as a C-style string.

## Supported features

### Assembler, binary parser, and disassembler

* Based on SPIR-V version 1.1 Rev 3
* Support for extended instruction sets:
  * GLSL std450 version 1.0 Rev 3
  * OpenCL version 1.0 Rev 2
* Support for SPIR-V 1.0 (with or without additional restrictions from Vulkan 1.0)
* Assembler only does basic syntax checking.  No cross validation of
  IDs or types is performed, except to check literal arguments to
  `OpConstant`, `OpSpecConstant`, and `OpSwitch`.

See [`syntax.md`](syntax.md) for the assembly language syntax.

### Validator

The validator checks validation rules described by the SPIR-V specification.

Khronos recommends that tools that create or transform SPIR-V modules use the
validator to ensure their outputs are valid, and that tools that consume SPIR-V
modules optionally use the validator to protect themselves from bad inputs.
This is especially encouraged for debug and development scenarios.

The validator has one-sided error: it will only return an error when it has
implemented a rule check and the module violates that rule.

The validator is incomplete.
See the [CHANGES](CHANGES) file for reports on completed work, and
the [Validator
sub-project](https://github.com/KhronosGroup/SPIRV-Tools/projects/1) for planned
and in-progress work.

*Note*: The validator checks some Universal Limits, from section 2.17 of the SPIR-V spec.
The validator will fail on a module that exceeds those minimum upper bound limits.
It is [future work](https://github.com/KhronosGroup/SPIRV-Tools/projects/1#card-1052403)
to parameterize the validator to allow larger
limits accepted by a more than minimally capable SPIR-V consumer.


### Optimizer

*Note:* The optimizer is still under development.

Currently supported optimizations:
* General
  * Strip debug info
* Specialization Constants
  * Set spec constant default value
  * Freeze spec constant
  * Fold `OpSpecConstantOp` and `OpSpecConstantComposite`
  * Unify constants
  * Eliminate dead constant
* Code Reduction (Entry Point Functions)
  * Inline all function calls exhaustively
  * Convert local access chains to inserts/extracts
  * Eliminate local load/store in single block
  * Eliminate local load/store with single store
  * Eliminate local load/store with multiple stores
  * Eliminate local extract from insert
  * Eliminate dead instructions (aggressive)
  * Eliminate dead branches
  * Merge single successor / single predecessor block pairs
  * Eliminate common uniform loads

For the latest list with detailed documentation, please refer to
[`include/spirv-tools/optimizer.hpp`](include/spirv-tools/optimizer.hpp).

### Extras

* [Utility filters](#utility-filters)
* Build target `spirv-tools-vimsyntax` generates file `spvasm.vim`.
  Copy that file into your `$HOME/.vim/syntax` directory to get SPIR-V assembly syntax
  highlighting in Vim.  This build target is not built by default.

## Source code

The SPIR-V Tools are maintained by members of the The Khronos Group Inc.,
at https://github.com/KhronosGroup/SPIRV-Tools.

Contributions via merge request are welcome. Changes should:
* Be provided under the [Apache 2.0](#license).
  You'll be prompted with a one-time "click-through" Contributor's License
  Agreement (CLA) dialog as part of submitting your pull request or
  other contribution to GitHub.
* Include tests to cover updated functionality.
* C++ code should follow the [Google C++ Style Guide][cpp-style-guide].
* Code should be formatted with `clang-format`.  Settings are defined by
  the included [.clang-format](.clang-format) file.

We intend to maintain a linear history on the GitHub `master` branch.

### Source code organization

* `example`: demo code of using SPIRV-Tools APIs
* `external/googletest`: Intended location for the
  [googletest][googletest] sources, not provided
* `include/`: API clients should add this directory to the include search path
* `external/spirv-headers`: Intended location for
  [SPIR-V headers][spirv-headers], not provided
* `include/spirv-tools/libspirv.h`: C API public interface
* `source/`: API implementation
* `test/`: Tests, using the [googletest][googletest] framework
* `tools/`: Command line executables

### Tests

The project contains a number of tests, used to drive development
and ensure correctness.  The tests are written using the
[googletest][googletest] framework.  The `googletest`
source is not provided with this project.  There are two ways to enable
tests:
* If SPIR-V Tools is configured as part of an enclosing project, then the
  enclosing project should configure `googletest` before configuring SPIR-V Tools.
* If SPIR-V Tools is configured as a standalone project, then download the
  `googletest` source into the `<spirv-dir>/external/googletest` directory before
  configuring and building the project.

*Note*: You must use a version of googletest that includes
[a fix][googletest-pull-612] for [googletest issue 610][googletest-issue-610].
The fix is included on the googletest master branch any time after 2015-11-10.
In particular, googletest must be newer than version 1.7.0.

## Build

The project uses [CMake][cmake] to generate platform-specific build
configurations. Assume that `<spirv-dir>` is the root directory of the checked
out code:

```sh
cd <spirv-dir>
git clone https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers
git clone https://github.com/google/googletest.git external/googletest # optional

mkdir build && cd build
cmake [-G <platform-generator>] <spirv-dir>
```

Once the build files have been generated, build using your preferred
development environment.

### CMake options

The following CMake options are supported:

* `SPIRV_COLOR_TERMINAL={ON|OFF}`, default `ON` - Enables color console output.
* `SPIRV_SKIP_TESTS={ON|OFF}`, default `OFF`- Build only the library and
  the command line tools.  This will prevent the tests from being built.
* `SPIRV_SKIP_EXECUTABLES={ON|OFF}`, default `OFF`- Build only the library, not
  the command line tools and tests.
* `SPIRV_USE_SANITIZER=<sanitizer>`, default is no sanitizing - On UNIX
  platforms with an appropriate version of `clang` this option enables the use
  of the sanitizers documented [here][clang-sanitizers].
  This should only be used with a debug build.
* `SPIRV_WARN_EVERYTHING={ON|OFF}`, default `OFF` - On UNIX platforms enable
  more strict warnings.  The code might not compile with this option enabled.
  For Clang, enables `-Weverything`.  For GCC, enables `-Wpedantic`.
  See [`CMakeLists.txt`](CMakeLists.txt) for details.
* `SPIRV_WERROR={ON|OFF}`, default `ON` - Forces a compilation error on any
  warnings encountered by enabling the compiler-specific compiler front-end
  option.

## Library

### Usage

The internals of the library use C++11 features, and are exposed via both a C
and C++ API.

In order to use the library from an application, the include path should point
to `<spirv-dir>/include`, which will enable the application to include the
header `<spirv-dir>/include/spirv-tools/libspirv.h{|pp}` then linking against
the static library in `<spirv-build-dir>/source/libSPIRV-Tools.a` or
`<spirv-build-dir>/source/SPIRV-Tools.lib`.
For optimization, the header file is
`<spirv-dir>/include/spirv-tools/optimizer.hpp`, and the static library is
`<spirv-build-dir>/source/libSPIRV-Tools-opt.a` or
`<spirv-build-dir>/source/SPIRV-Tools-opt.lib`.

* `SPIRV-Tools` CMake target: Creates the static library:
  * `<spirv-build-dir>/source/libSPIRV-Tools.a` on Linux and OS X.
  * `<spirv-build-dir>/source/libSPIRV-Tools.lib` on Windows.
* `SPIRV-Tools-opt` CMake target: Creates the static library:
  * `<spirv-build-dir>/source/libSPIRV-Tools-opt.a` on Linux and OS X.
  * `<spirv-build-dir>/source/libSPIRV-Tools-opt.lib` on Windows.

#### Entry points

The interfaces are still under development, and are expected to change.

There are five main entry points into the library in the C interface:

* `spvTextToBinary`: An assembler, translating text to a binary SPIR-V module.
* `spvBinaryToText`: A disassembler, translating a binary SPIR-V module to
  text.
* `spvBinaryParse`: The entry point to a binary parser API.  It issues callbacks
  for the header and each parsed instruction.  The disassembler is implemented
  as a client of `spvBinaryParse`.
* `spvValidate` implements the validator functionality. *Incomplete*
* `spvValidateBinary` implements the validator functionality. *Incomplete*

The C++ interface is comprised of two classes, `SpirvTools` and `Optimizer`,
both in the `spvtools` namespace.
* `SpirvTools` provides `Assemble`, `Disassemble`, and `Validate` methods.
* `Optimizer` provides methods for registering and running optimization passes.

## Command line tools

Command line tools, which wrap the above library functions, are provided to
assemble or disassemble shader files.  It's a convention to name SPIR-V
assembly and binary files with suffix `.spvasm` and `.spv`, respectively.

### Assembler tool

The assembler reads the assembly language text, and emits the binary form.

The standalone assembler is the exectuable called `spirv-as`, and is located in
`<spirv-build-dir>/tools/spirv-as`.  The functionality of the assembler is implemented
by the `spvTextToBinary` library function.

* `spirv-as` - the standalone assembler
  * `<spirv-dir>/tools/as`

Use option `-h` to print help.

### Disassembler tool

The disassembler reads the binary form, and emits assembly language text.

The standalone disassembler is the executable called `spirv-dis`, and is located in
`<spirv-build-dir>/tools/spirv-dis`. The functionality of the disassembler is implemented
by the `spvBinaryToText` library function.

* `spirv-dis` - the standalone disassembler
  * `<spirv-dir>/tools/dis`

Use option `-h` to print help.

The output includes syntax colouring when printing to the standard output stream,
on Linux, Windows, and OS X.

### Optimizer tool

The optimizer processes a SPIR-V binary module, applying transformations
in the specified order.

This is a work in progress, with initially only few available transformations.

* `spirv-opt` - the standalone optimizer
  * `<spirv-dir>/tools/opt`

### Validator tool

*Warning:* This functionality is under development, and is incomplete.

The standalone validator is the executable called `spirv-val`, and is located in
`<spirv-build-dir>/tools/spirv-val`. The functionality of the validator is implemented
by the `spvValidate` library function.

The validator operates on the binary form.

* `spirv-val` - the standalone validator
  * `<spirv-dir>/tools/val`

### Control flow dumper tool

The control flow dumper prints the control flow graph for a SPIR-V module as a
[GraphViz](http://www.graphviz.org/) graph.

This is experimental.

* `spirv-cfg` - the control flow graph dumper
  * `<spirv-dir>/tools/cfg`

### Utility filters

* `spirv-lesspipe.sh` - Automatically disassembles `.spv` binary files for the
  `less` program, on compatible systems.  For example, set the `LESSOPEN`
  environment variable as follows, assuming both `spirv-lesspipe.sh` and
  `spirv-dis` are on your executable search path:
  ```
   export LESSOPEN='| spirv-lesspipe.sh "%s"'
  ```
  Then you page through a disassembled module as follows:
  ```
  less foo.spv
  ```
  * The `spirv-lesspipe.sh` script will pass through any extra arguments to
    `spirv-dis`.  So, for example, you can turn off colours and friendly ID
    naming as follows:
    ```
    export LESSOPEN='| spirv-lesspipe.sh "%s" --no-color --raw-id'
    ```

* [vim-spirv](https://github.com/kbenzie/vim-spirv) - A vim plugin which
  supports automatic disassembly of `.spv` files using the `:edit` command and
  assembly using the `:write` command. The plugin also provides additional
  features which include; syntax highlighting; highlighting of all ID's matching
  the ID under the cursor; and highlighting errors where the `Instruction`
  operand of `OpExtInst` is used without an appropriate `OpExtInstImport`.

* `50spirv-tools.el` - Automatically disassembles '.spv' binary files when
  loaded into the emacs text editor, and re-assembles them when saved,
  provided any modifications to the file are valid.  This functionality
  must be explicitly requested by defining the symbol
  SPIRV_TOOLS_INSTALL_EMACS_HELPERS as follows:
  ```
  cmake -DSPIRV_TOOLS_INSTALL_EMACS_HELPERS=true ...
  ```

  In addition, this helper is only installed if the directory /etc/emacs/site-start.d
  exists, which is typically true if emacs is installed on the system.

  Note that symbol IDs are not currently preserved through a load/edit/save operation.
  This may change if the ability is added to spirv-as.


### Tests

Tests are only built when googletest is found. Use `ctest` to run all the
tests.

## Future Work
<a name="future"></a>

_See the [projects pages](https://github.com/KhronosGroup/SPIRV-Tools/projects)
for more information._

### Assembler and disassembler

* The disassembler could emit helpful annotations in comments.  For example:
  * Use variable name information from debug instructions to annotate
    key operations on variables.
  * Show control flow information by annotating `OpLabel` instructions with
    that basic block's predecessors.
* Error messages could be improved.

### Validator

This is a work in progress.

## Licence
<a name="license"></a>
Full license terms are in [LICENSE](LICENSE)
```
Copyright (c) 2015-2016 The Khronos Group Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

[spirv-registry]: https://www.khronos.org/registry/spir-v/
[spirv-headers]: https://github.com/KhronosGroup/SPIRV-Headers
[googletest]: https://github.com/google/googletest
[googletest-pull-612]: https://github.com/google/googletest/pull/612
[googletest-issue-610]: https://github.com/google/googletest/issues/610
[CMake]: https://cmake.org/
[cpp-style-guide]: https://google.github.io/styleguide/cppguide.html
[clang-sanitizers]: http://clang.llvm.org/docs/UsersManual.html#controlling-code-generation

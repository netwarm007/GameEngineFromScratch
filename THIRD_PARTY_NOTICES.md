# Third-party software notices

The project code in `LICENSE` is licensed under MIT. That license does not
replace the separate terms for third-party components listed here. Preserve
the corresponding upstream license files and notices when redistributing those
components. For Git submodules, the referenced license files are available
after `git submodule update --init --recursive`.

## Required OpenGEX and OpenDDL credits

The OpenGEX import module and OpenDDL Library are included through
`External/src/opengex`. Their source headers contain their full licenses and
require these exact attributions in binary distribution documentation or
other distribution materials:

> This software contains an OpenGEX import module based on work by Eric Lengyel

> This software contains a modified version of the OpenDDL Library by Eric Lengyel

The OpenGEX and OpenDDL source licenses require modified source files to
identify that they were modified. Retain the notices already present in those
files.

## Vendored source and submodule dependencies

| Component | Copyright / license | License notice location |
| --- | --- | --- |
| Bullet Physics | Bullet's zlib-style license; see the component license for its copyright holders and any bundled subcomponents. | `External/src/bullet/LICENSE.txt` |
| cmft | Copyright 2014-2015 Dario Manesku; BSD-style 2-clause license. | `External/src/cmft/LICENSE` |
| crossguid | Copyright (c) 2014 Graeme Hill; MIT. | `External/src/crossguid/LICENSE` |
| Emscripten SDK | Copyright (c) 2018 Emscripten authors; MIT. Build-time/WebAssembly toolchain. | `External/src/emsdk/LICENSE` |
| GNU Bison | Free Software Foundation; GPL-3.0. Build-time parser generator. | `External/src/bison/COPYING` |
| glslang | Google, Khronos Group and other contributors; BSD-3-Clause for glslang proper, with additional component notices described in its license file. | `External/src/glslang/LICENSE.txt` |
| GLFW | Copyright (c) 2002-2006 Marcus Geelnard and (c) 2006-2019 Camilla Löwy; zlib/libpng license. | `External/src/glfw/LICENSE.md` |
| ImGui | Copyright (c) 2014-2026 Omar Cornut; MIT. This also covers the copied platform and OpenGL backend sources in `Platform/` and `RHI/OpenGL/`. | `External/src/imgui/LICENSE.txt` |
| ImGui Node Editor | Copyright (c) 2019 Michał Cichoń; MIT. | `External/src/imgui-node-editor/LICENSE` |
| ISPC Texture Compressor | Copyright 2017 Intel Corporation; MIT. | `External/src/ISPCTextureCompressor/license.txt` |
| libcrc | Copyright (c) 1999-2016 Lammert Bies; MIT. | `External/src/libcrc/LICENSE` |
| libpng | PNG Reference Library License version 2; copyright notices are listed in the license file. | `External/src/libpng/LICENSE` |
| OpenAL Soft | GNU Library General Public License, version 2 (LGPL-2.0). Review the LGPL requirements when redistributing a linked OpenAL library. | `External/src/openal/COPYING` |
| OpenGEX / OpenDDL | Copyright 2014-2017 Eric Lengyel; custom permissive licenses with source modification notices and binary attribution requirements. | License headers in `External/src/opengex/OpenGEX/` and `External/src/opengex/OpenDDL/` |
| SPIRV-Cross | Khronos Group; Apache-2.0. | `External/src/spirv-cross/LICENSE` |
| tinyobjloader | Copyright (c) 2012-2019 Syoyo Fujita and contributors; MIT. The distribution also contains Mapbox earcut under ISC. | `External/src/tinyobjloader/LICENSE` |
| Vulkan-Hpp | Khronos Group; Apache-2.0. | `External/src/Vulkan-hpp/LICENSE.txt` |
| zlib | Copyright (C) 1995-2017 Jean-loup Gailly and Mark Adler; zlib license. | Notice in `External/src/zlib/zlib.h` |
| astc-encoder | ARM; Apache-2.0. | `External/src/astc-encoder/LICENSE.txt` |

The platform ISPC compiler packages have separate notices in
`External/ispc/<platform>/LICENSE.txt` and
`External/ispc/<platform>/third-party-programs.txt`.

## Other copied or generated code

- **GLAD OpenGL loader** (`External/src/glad/`): generated with glad 0.1.14a0;
  the glad source is MIT, Copyright (c) 2013-2018 David Herberth. The included
  Khronos platform header identifies its Khronos copyright and license in the
  header itself.
- **Khronos OpenGL declarations** (`RHI/OpenGL/imgui_impl_opengl3_loader.h`):
  Copyright 2013-2020 The Khronos Group Inc.; MIT. The generated `gl3w` loader
  portion identifies itself as public domain in the same file.
- **Microsoft D3D12 utility excerpt** in `RHI/D3d/D3d12Utility.hpp`: Copyright
  (c) Microsoft; MIT. The original notice is retained beside the excerpt.
- **CMake FindSDL2 module** (`cmake/FindSDL2.cmake`): BSD 3-Clause, as stated
  in the file header; see <https://cmake.org/licensing>.
- **SDL2 library**: optional external dependency discovered by
  `find_package(SDL2)`; upstream uses the zlib license, Copyright Sam Lantinga.
  SDL2 is not vendored here, so include the notice from the actual SDL2 package
  version when redistributing that library.
- **OpenGEX Blender exporter** (`scripts/OpenGex-Blender.py`): Copyright 2017
  Terathon Software LLC; Creative Commons Attribution-ShareAlike 3.0 Unported.
  The exporter records the author and license in its header.
- **LLVM scripts** (`scripts/run-clang-tidy.py` and
  `Game/Billiard/llvm.sh`): Apache-2.0 with LLVM Exceptions; notices are
  retained in the files.
- **Gradle wrapper scripts** (`Game/Billiard/gradlew` and
  `Game/Billiard/gradlew.bat`): Apache-2.0; notices are retained in the files.

CEF and the Viking Room model already have their notices in `LICENSE`.

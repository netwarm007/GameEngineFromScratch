# Game Engine From Scratch
[![CircleCI Build Status](https://circleci.com/gh/netwarm007/GameEngineFromScratch.svg?style=shield)](https://circleci.com/gh/netwarm007/GameEngineFromScratch) 
[![Build status](https://ci.appveyor.com/api/projects/status/hld88pk7py29thx5?svg=true&branch=master)](https://ci.appveyor.com/project/netwarm007/gameenginefromscratch)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/netwarm007/GameEngineFromScratch/master/LICENSE)

配合我的知乎[专栏](https://zhuanlan.zhihu.com/c_119702958)当中的系列文章《从零开始手敲次世代游戏引擎》所写的项目。

This project is written by me as the sample of My articles [Hand-made Next-Gen Game Engine From Scratch](https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352)

このプロジェクトは私の連載中の[ゼロから始める手作り次世代ゲームエンジン](
https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352)のサンプルソースである

## Platform Support Status
- Windows 10 / 11 (targeted; CircleCI uses Windows Server 2022).
- macOS (CircleCI uses Apple Silicon and Xcode 16.4; Catalina-Monterey are not
  covered by the current CI).
- Linux (CircleCI uses Ubuntu 22.04; Ubuntu 20.04 and CentOS 7 are not covered
  by the current CI).
- FreeBSD (not tested recently; build might fail).
- Android (CircleCI job disabled; not tested recently).
- WebAssembly (Emscripten; not tested recently; build might fail).

CircleCI [workflow 90cbedb9](https://app.circleci.com/workflow/90cbedb9-ad56-4538-baf9-63bdb4d63399)
passed the configured builds and tests for revision `29800d81` (Viewer scene
graph debug window) on Windows, macOS, and Linux. Separately, CircleCI
[workflow a82fc15c](https://app.circleci.com/workflow/a82fc15c-fc05-4f0f-a825-18b4bb5bedd1)
passed the configured builds and tests for revision
`f298bfc2c300eca4f5a537dec098499b7a3e052d` (Viewer screenshot capture) on the
same three platforms. These results apply only to those revisions; they do
not validate their merged changes or every OS version.

The macOS job saves Viewer output under the `viewer-screenshots` artifacts:
`viewer-render.png` and `viewer.log`. Viewer uses Metal GPU readback rather than
desktop capture, so capturing does not require a physical display. Launch with
`-screenshot <path>` and optionally `-screenshotDelay <seconds>` (default: 45).
Screenshot capture is best-effort and does not gate the build; a successful job
alone does not guarantee a PNG was produced.

The Linux job also attempts a screenshot under `viewer-screenshots`, using Xvfb
and Mesa software OpenGL rendering. After 60 seconds it captures the Viewer
window as `viewer-render.png`; `viewer.log`, `capture.log`, and `diagnostics.txt`
record startup, capture errors, and the OpenGL renderer. The capture has a
180-second timeout and is best-effort, not a build gate. Linux capture was
verified in CircleCI at revision `11ea6f88feb773734f1c75252045afa21ee158b0`;
the resulting image shows the scene and ImGui interface without the earlier
window-size clipping.

The Windows job attempts D3D12 back-buffer readback after 45 seconds and writes
`viewer-render.png` using WIC, without desktop capture. Set
`GEFS_SCREENSHOT_PATH` to an output filename to enable this one-shot capture;
Viewer closes after the attempt. Device creation falls back to WARP software
rendering when hardware device creation fails. The job retains `viewer.log`,
`viewer-error.log`, and capture errors under `viewer-screenshots`, with a
180-second timeout. Capture is best-effort. Windows capture was verified in
CircleCI at revision `f298bfc2c300eca4f5a537dec098499b7a3e052d`; the supplied
image shows the rendered scene, textures, and Chinese ImGui interface. The
writer maps only valid pixel bytes and packs rows as opaque BGRA for PNG
encoding, removing incomplete files on failure. This verifies screenshot
capture, not full rendering correctness or execution through the WARP fallback.

## Graphic API Support Status
- OpenGL
- OpenGL ES (Not tested recently)
- Metal2
- DirectX 12
- Vulkan (On going)

## Physics
- Bullet
- My Physics (on going)

## Scene Graph
- OpenGEX
- Collada (On the roadmap, not implemented yet)

## Shading Language
- HLSL, auto convert to GLSL/Metal Performance Shader

## Texture Format
- JPEG
- PNG
- TGA
- HDR
- DDS
- BMP
- PVR (BC1 - 7)
- ASTC

## High Performance / Parallel Computing
- ISPC

## Architecture
- `RHI/`: render hardware interfaces (OpenGL, Vulkan, Metal, D3d, D2d, Empty)
- `Platform/`: platform layers (Windows, Linux, Darwin, Sdl, Android, Empty)
- `Framework/`, `Physics/`, `Asset/`, `Utility/`, `Game/`, `Viewer/`, `Test/`

## Dependencies
Requires a C++20 compiler and CMake 4.0 or newer. Third-party libraries live in
`External/src` as Git submodules; Git LFS is also used by CI.

- Windows
  - Windows Platform SDK
  - Visual Studio or Clang
  - CMake
  - Vulkan SDK (used by AppVeyor)

- MacOS
  - Xcode and command line tools
  - Homebrew (as used by CI): `brew install cmake ninja flex bison ispc` (MacPorts also works)

- Linux
  - gcc/g++ or clang/clang++, CMake, Ninja, ISPC, flex, bison
  - uuid-dev libx11-dev libx11-xcb-dev libgl1-mesa-dev libnss3-dev libxss-dev libatk1.0-dev libatk-bridge2.0-dev libglib2.0-dev libpango1.0-dev libxi-dev libfontconfig1-dev libnspr4-dev libxcomposite-dev libxcursor-dev libxrender-dev libxtst-dev libxrandr-dev libgio2.0-cil-dev libdbus-1-dev libasound2-dev libcups2-dev libncurses5-dev

- Android
  - Android SDK
  - Android NDK

## Build Steps
Fetch sources first:

    git submodule sync
    git submodule update --init --recursive
    git lfs pull

### Commit acceptance

A commit is accepted only when its CircleCI pipeline completes the configured
builds and tests successfully on all enabled platforms: currently Windows,
Linux, and macOS. Local tests do not replace this requirement. Android is
currently disabled in the CircleCI workflow and is not part of this gate.

Commits that change only documentation (Markdown files and other prose that no
build, test, or script reads) are exempt: CircleCI is not run for them, and
their commit message carries `[ci skip]`. Mixed commits are not exempt. Such a
commit does not change the last verified revision.

The Windows job uses two parallel CMake build jobs and caches installed dependency
files using their revisions, build scripts, and toolchain details. MSVC may also
compile source files in parallel within a build job.
Cache misses rebuild the dependencies; the engine and tests still run on every
pipeline.

Packaging and package artifacts are limited to `master` and tagged builds;
builds, tests, and Viewer screenshot attempts also run on other branches.

### Windows
Windows scripts that use Visual Studio select the newest installed version
supported by CMake. If an existing build directory retains an older generator,
move it aside before rerunning the script to allow a fresh configuration.

    scripts\build_crossguid
    scripts\build_opengex
    scripts\build_zlib
    scripts\build_bullet
    scripts\build_cef
    scripts\build_glslangValidator
    scripts\build_spirvcross
    scripts\build_imgui
    scripts\build_libispc-texcomp
    scripts\build
### MacOS & Linux
    ./scripts/build_crossguid.sh
    ./scripts/build_opengex.sh
    ./scripts/build_bullet.sh
    ./scripts/build_cef.sh
    ./scripts/build_glslangValidator.sh
    ./scripts/build_spirvcross.sh
    ./scripts/build_imgui.sh
    ./scripts/build_libispc-texcomp.sh
    ./scripts/build-ninja.sh

CI uses plain CMake after the dependencies are built:

    cmake -S . -B build
    cmake --build build
    cmake --build build --target package

Windows CI additionally enables `-DOPENGL_RHI_DEBUG=ON -DD3D12_RHI_DEBUG=ON`.

## Cook Materials
### Windows
    scripts\cook_material
### MacOS & Linux
    ./scripts/cook_material

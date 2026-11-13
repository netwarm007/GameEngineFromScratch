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

CircleCI [workflow](https://app.circleci.com/workflow/d18dff17-1575-4c6b-b2d8-d2ece02d9230)
passed the configured builds and tests for revision `1b6da1c677aa4e763f37396d3610fa066ba14273` on Windows,
macOS, and Linux. These results apply to that revision and do not validate every
OS version.

The macOS job saves Viewer output under the `viewer-screenshots` artifacts:
`viewer-render.png` and `viewer.log`. Viewer uses Metal GPU readback rather than
desktop capture, so capturing does not require a physical display. Launch with
`-screenshot <path>` and optionally `-screenshotDelay <seconds>` (default: 45).
Screenshot capture is best-effort and does not gate the build; a successful job
alone does not guarantee a PNG was produced. Linux and Windows screenshot
capture is not implemented.

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

## Dependencies
- Windows
-- Windows Platform SDK
-- Visual Studio or Clang
-- CMake

- MacOS
-- Xcode
-- Xcode command line tools
-- MacPorts
-- CMake

- Linux
-- gcc/g++ or clang/clang++
-- uuid-dev libx11-dev libx11-xcb-dev libgl1-mesa-dev libnss3-dev libxss-dev libatk1.0-dev libatk-bridge2.0-dev libglib2.0-dev libpango1.0-dev libxi-dev libfontconfig1-dev libnspr4-dev libxcomposite-dev libxcursor-dev libxrender-dev libxtst-dev libxrandr-dev libgio2.0-cil-dev libdbus-1-dev libasound2-dev libcups2-dev libncurses5-dev

- Android
-- Android SDK
-- Android NDK

## Build Steps
### Commit acceptance

A commit is accepted only when its CircleCI pipeline completes the configured
builds and tests successfully on all enabled platforms: currently Windows,
Linux, and macOS. Local tests do not replace this requirement. Android is
currently disabled in the CircleCI workflow and is not part of this gate.

The Windows job uses two parallel CMake build jobs and caches installed dependency
files using their revisions, build scripts, and toolchain details. MSVC may also
compile source files in parallel within a build job.
Cache misses rebuild the dependencies; the engine and tests still run on every
pipeline.

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
    ./scripts/build_imgui
    ./scripts/build_libispc-texcomp
    ./scripts/build-ninja.sh

## Cook Materials
### Windows
    scripts\cook_material
### Mac)S & Linux
    ./scripts/cook_material

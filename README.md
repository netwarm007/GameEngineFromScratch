# Game Engine From Scratch 
[![CircleCI Build Status](https://circleci.com/gh/netwarm007/GameEngineFromScratch.svg?style=shield)](https://circleci.com/gh/netwarm007/GameEngineFromScratch) 
[![Build status](https://ci.appveyor.com/api/projects/status/hld88pk7py29thx5?svg=true)](https://ci.appveyor.com/project/netwarm007/gameenginefromscratch)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/netwarm007/GameEngineFromScratch/master/LICENSE)

配合我的知乎[专栏](https://zhuanlan.zhihu.com/c_119702958)当中的系列文章《从零开始手敲次世代游戏引擎》所写的项目。

This project is written by me as the sample of My articles [Hand-made Next-Gen Game Engine From Scratch](https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352)

このプロジェクトは私の連載中の[ゼロから始める手作り次世代ゲームエンジン](
https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352)のサンプルソースである

## Platform Support Status
- Windows 10
- MacOS Catalina
- Linux (Build test on Ubuntu 20.04 and CentOS 7)
- FreeBSD (Not tested recently, build might fail)
- Android (Not tested recently, build might fail)
- WebAssembly (Emscripten, not tested recently, build might fail)
- PlayStation 4 (Not continued, related source code not disclosed due to NDA)
- PlayStation Vita (Not continued, related source code not disclosed due to NDA)

## Graphic API Support Status
- OpenGL
- OpenGL ES (Not tested recently)
- Metal2
- DirectX 12 (On going)
- Vulkan (On the roadmap, not implemented yet)
- GNM (Not disclosed due to NDA)

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
- TIFF
- HDR
- DDS
- BMP

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
### Windows
    scripts/build_crossguid
    scripts/build_opengex
    scripts/build_zlib
    scripts/build_bullet
    scripts/build_cef
    scripts/build_glslangValidator
    scripts/build_spirvcross
    scripts/build
### MacOS
    export PATH=/opt/local/bin:/opt/klocal/sbin:$PATH 
    ./scripts/build_crossguid.sh
    ./scripts/build_opengex.sh
    ./scripts/build_zlib.sh
    ./scripts/build_bullet.sh
    ./scripts/build_cef.sh
    ./scripts/build-ninja.sh
### Linux
    ./scripts/build_crossguid.sh
    ./scripts/build_opengex.sh
    ./scripts/build_bullet.sh
    ./scripts/build_cef.sh
    ./scripts/build_glslangValidator.sh
    ./scripts/build_spirvcross.sh
    ./scripts/build-ninja.sh
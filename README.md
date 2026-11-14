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

CircleCI [pipeline 875](https://app.circleci.com/pipelines/github/netwarm007/GameEngineFromScratch/875)
passed the configured builds and tests for revision
`1f7935c0343e344228dab395d80aea53819d34ff` on Windows (job 4175), Linux
(job 4174), and macOS (job 4173), including the runtime lifecycle, scene
services, parser, scene-loading, animation, and headless layout regressions.
The Linux static-link failure from pipeline 873 is resolved.
Actual Viewer screenshots from all three jobs were inspected
against the lifecycle checkpoint `c8da9d98f9b269585ff6c3979c458bc28108828f`
([pipeline 872](https://app.circleci.com/pipelines/github/netwarm007/GameEngineFromScratch/872)):
the splash scene,
materials, scene graph icons, and docked panels show no visible regression.
These results apply only to the tested configurations and captured scene,
not every OS version or full rendering correctness.

Viewer's ImGui renderer backends build and update the font atlas dynamically;
startup registers fonts without manually building the atlas.

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
At revision `f6dabf7dfa9a57066ba406ca760ffa5fd8ebbe2a`, the Linux job produced
the PNG artifact, and its Viewer and capture logs contained no font-atlas
startup assertion.

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

On first use, Viewer opens with a docked layout: the scene graph on the left,
the scene preview on the right, and the debug panel across the bottom. The
panels resize with the main viewport without overlapping. Existing ImGui
layouts are preserved. Use Debug > Reset window layout to restore these
defaults, or reopen individual panels through the Debug menu.

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

Metal material textures upload supplied mip chains. Single-level materials
generate a full mip chain before rendering; BC textures are first sampled into
an uncompressed texture because Metal blit mip generation does not support BC
formats. This processing does not regenerate prefiltered environment maps or
add mipmaps to UI textures. PVR parsing records block-aligned mip pitches,
sizes, and offsets, including levels smaller than a compression block.

The macOS screenshot from pipeline 861 confirms reduced brick-floor speckling
in the splash scene compared with pipeline 857 (`724642b3`). In the visible
floor region x=[630,820), y=[510,610), the mean absolute RGB difference from
the four-neighbor average fell from 17.127 to 7.528 on a 0-255 scale (56.05%).
The same-region Linux and Windows values were 6.948 and 7.047. This is a
pixel-scale high-frequency comparison, not a general image-quality benchmark
or verification of every material and camera position.

## High Performance / Parallel Computing
- ISPC
  - Official stable compiler: [1.32.0](https://github.com/ispc/ispc/releases/tag/v1.32.0)
    (released 2026-10-13), pinned by SHA-256 in `scripts/install-ispc.cmake`.
    Windows, Linux x86-64/AArch64, and universal macOS packages are supported.
    The legacy FreeBSD binary is not covered by this upgrade.

## Architecture
- `RHI/`: render hardware interfaces (OpenGL, Vulkan, Metal, D3d, D2d, Empty)
- `Platform/`: platform layers (Windows, Linux, Darwin, Sdl, Android, Empty)
- `Framework/`, `Physics/`, `Asset/`, `Utility/`, `Game/`, `Viewer/`, `Test/`

Runtime modules are borrowed by the application and must outlive it. Register
each module once, before initialization. Initialization and updates preserve
registration order; shutdown runs in reverse order. If initialization fails,
the application rolls back only successfully initialized modules. The failing
module must release its own partially initialized resources before returning
an error. A rolled-back group can retry initialization; final shutdown clears
its registrations. Viewer and Billiard do not enter the main loop on failure.

The lifecycle regression can run without platform, graphics, asset, or external
dependencies:

    cmake -S Test/Runtime -B build-runtime
    cmake --build build-runtime --config Debug
    ctest --test-dir build-runtime -C Debug --output-on-failure

`SceneManager` takes an `IAssetReader` and an `ISceneParser` by reference;
`AnimationManager` takes an `ISceneManager` by reference. These borrowed
dependencies must outlive the managers. Application entry points select the
OpenGEX parser explicitly; scene services no longer discover their dependencies
through the application. `SceneServicesTest` exercises loading, failed reloads,
and animation without creating an application or a window. It links
`SceneServices`, `SceneGraph`, `AssetIO`, `Common`, and the small
`PlatformAssetIO` bootstrap, not the rendering or window libraries.
`AssetIO` contains shared file operations; `PlatformAssetIO` supplies native
initialization, while SDL applications retain their separate `SdlAssetIO`
bootstrap. The selected bootstrap is linked as an object rather than a static
archive, so its virtual-function implementation is not dependent on archive
scan order. Scene textures still own loading behavior; removing that remaining
resource-model coupling is a separate migration.
Failed scene reloads leave the current scene and its revision unchanged and
report the read or parse failure.

Physics backends own rigid-body bindings and retain bound geometry nodes until
deletion or cleanup. Scene nodes contain no backend handles. A published
simulation transform overrides the authored/runtime transform for rendering;
deleting a body restores normal scene transforms without changing the authored
data. Graphics consumes `GetRenderTransform()` and no longer queries physics.
Forces and explicit body-transform updates address scene nodes and return false
when no body is bound. `MyPhysicsManager` still does not simulate forces.
Scene-wide cleanup uses the backend's own bindings, not the newly published
scene, so reloads cannot strand old bodies. `PhysicsSceneIntegrationTest`
exercises both backends without constructing an application or window; their
scene discovery during `Tick()` remains a later dependency-injection step.

Billiard uses the renderer's dynamic font-atlas initialization, like Viewer.
Its opt-in desktop regression loads the real billiard scene, settles it for
60 ticks, applies the normal left-key shot, and follows 120 further ticks.
It requires cue-ball XY displacement of at least 0.1 scene units and movement
of another ball of at least 0.01 units, with finite transforms. A regression-only
overview camera keeps the table in view; normal camera and saved layouts are
unchanged. The default Windows OpenGL build can run
`BilliardGame -physicsRegression <output-directory>` to save before/after BMPs
and a `physics-passed` result. Windows CI explicitly selects the existing
D3D12 backend with `-DBILLIARD_USE_D3D12=ON`; ordinary builds default to OpenGL.
Set `GEFS_SCREENSHOT_PATH` and `GEFS_SCREENSHOT_WAIT_FOR` (the `physics-passed`
path) to capture a native PNG immediately after the check. This does not verify
Windows OpenGL on CI hosts without WGL support.
Linux uses `scripts/capture_billiard_linux.sh` inside Xvfb to capture the
Billiard window after physics success, then signals `capture-complete` and
verifies normal exit. OpenGL modules shut down before their contexts are
destroyed, including the previously missing Linux module finalization.
macOS uses native Metal PNG capture with
`-screenshot <path> -screenshotWaitFor <output-directory>/physics-passed
-screenshotDelay 0`; capture cannot complete before the physics check passes.
All three enabled CI jobs run this regression and retain Billiard screenshots
(Windows D3D12, Linux OpenGL, macOS Metal).
Third-stage acceptance requires inspecting these real Billiard images, not
Viewer's physics-free scene alone.

## Dependencies
Requires a C++20 compiler and CMake 4.0 or newer. Third-party libraries live in
`External/src` as Git submodules; Git LFS is also used by CI.

- Windows
  - Windows Platform SDK
  - Visual Studio or Clang
  - CMake and Ninja
  - Vulkan SDK (used by AppVeyor)

- MacOS
  - Xcode and command line tools
  - Homebrew (as used by CI): `brew install cmake ninja flex bison` (MacPorts also works)

- Linux
  - gcc/g++ or clang/clang++, CMake, Ninja, flex, bison
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
    scripts\build_openal
    scripts\build

To build and run the Windows DX12 Viewer with Visual Studio after preparing the
dependencies above:

    cmake -S . -B build -DOPENGL_RHI_DEBUG=ON -DD3D12_RHI_DEBUG=ON
    cmake --build build --target Viewer --config Debug --parallel 2
    cd build
    Viewer\Debug\Viewer.exe

Run from the build directory so the asset loader finds compiled shaders under
`build/Asset` and source assets in the parent directory. Re-run
`scripts\build_imgui` after updating ImGui; the Windows script uses native
`copy` commands and does not require a Unix shell. Viewer lets the ImGui
renderer backend build and upload the font atlas instead of building it before
backend initialization.
Release builds also need matching Release versions of crossguid, OpenGEX, and
OpenDDL; linking their Debug libraries causes MSVC `LNK2038` runtime and iterator
debug-level mismatches. Keep separate build directories and select the matching
libraries with `XG_LIBRARY`, `OPENGEX_LIBRARY`, and `OPENDDL_LIBRARY` when
validating both configurations locally.

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

### ISPC and geometry math regression

`scripts/build_ispc.bat` (Windows) or `scripts/build_ispc.sh` (Linux/macOS)
installs the official pinned compiler into `External/<platform>/bin`. Texture
compressor build scripts also run this installer; no ISPC/LLVM source submodule
is required. Redistribution notices are in `External/ispc`.

The focused suite builds the same tests for both C++ and ISPC without requiring
rendering, audio, or asset dependencies. Run from a compiler-enabled shell
(on Windows, an x64 Visual Studio developer prompt with Ninja on PATH):

    cmake -S Test/GeomMath -B build-geommath -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build-geommath --parallel 2
    ctest --test-dir build-geommath --output-on-failure

All three enabled CircleCI platforms run this suite in addition to their engine
builds and tests. It checks DCT/IDCT against an independent direct formula,
round trips, SIMD reduction tails, and row-vector transforms. DCT/IDCT now use
two separable passes and a shared precomputed orthonormal basis (1024 instead
of 4096 terms per block; ISPC scratch storage is 256 bytes instead of 16 KiB).
Transform coordinates retain row-vector translation in the fourth matrix row.

For a Release benchmark with runtime-generated inputs, five samples, and an
observable checksum, run each `GeomMathCppTest`/`GeomMathIspcTest` executable
from `build-geommath` with `--benchmark`. Timing is informational, not a
machine-dependent correctness threshold. These tests do not by themselves
establish full three-platform commit acceptance.

## Cook Materials
### Windows
    scripts\cook_material
### MacOS & Linux
    ./scripts/cook_material

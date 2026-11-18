# Game Engine From Scratch
[![CircleCI Build Status](https://circleci.com/gh/netwarm007/GameEngineFromScratch.svg?style=shield)](https://circleci.com/gh/netwarm007/GameEngineFromScratch) 
[![Build status](https://ci.appveyor.com/api/projects/status/hld88pk7py29thx5?svg=true&branch=master)](https://ci.appveyor.com/project/netwarm007/gameenginefromscratch)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/netwarm007/GameEngineFromScratch/master/LICENSE)

配合我的知乎[专栏](https://zhuanlan.zhihu.com/c_119702958)当中的系列文章《从零开始手敲次世代游戏引擎》所写的项目。

This project is written by me as the sample of My articles [Hand-made Next-Gen Game Engine From Scratch](https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352)

このプロジェクトは私の連載中の[ゼロから始める手作り次世代ゲームエンジン](
https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352)のサンプルソースである

## Project intent and evolution

This project grew alongside the [Zhihu series](https://zhuanlan.zhihu.com/c_119702958?group_id=934116274502500352).
Its starting plan was deliberately runtime-first and rendering-first, with
editor and host tools deferred until after the runtime. The series records an
open-ended, hands-on progression: from the initial plan
([introduction](https://zhuanlan.zhihu.com/p/28587092)), through integrating
Bullet into a scene simulation ([part 38](https://zhuanlan.zhihu.com/p/33090740)),
to later GPU-focused ray-tracing experiments
([part 101](https://zhuanlan.zhihu.com/p/612018390)).

Those articles are historical context, not a feature contract for the current
repository. Current behavior and engineering plans are defined by the source,
focused roadmaps, tests, and revision-specific verification records.

## Platform Support Status
- Windows 10 / 11 (targeted; CircleCI uses Windows Server 2022).
- macOS (CircleCI uses Apple Silicon and Xcode 16.4; Catalina-Monterey are not
  covered by the current CI).
- Linux (CircleCI uses Ubuntu 22.04; Ubuntu 20.04 and CentOS 7 are not covered
  by the current CI).
- FreeBSD (not tested recently; build might fail).
- Android (CircleCI job disabled; not tested recently).
- WebAssembly (Emscripten; not tested recently; build might fail).

Revision-specific CI results and screenshot observations are kept in
[verification records](docs/verification/), named by the full tested commit SHA.
A record applies only to its exact revision and listed configurations; it does
not verify later commits or every supported OS and rendering backend.

Viewer and Billiard's ImGui renderer backends build and update the font atlas dynamically;
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
The [modularization roadmap](docs/architecture/modularization-roadmap.md)
describes the refactoring goals, completed boundaries, proposed next checkpoints,
and acceptance process.
The [memory allocator roadmap](docs/architecture/memory-allocator-roadmap.md)
tracks the review findings and planned allocators.

Desktop graphics applications create the native window with `CreateMainWindow()`
before calling `Initialize()`. Application initialization prepares the backend
before initializing runtime modules; module-initialization retries reuse a
successfully prepared backend. `Finalize()` synchronizes GPU work, stops modules
in reverse order, destroys backend resources, then releases the native window.
Call it on initialization failure as well as normal exit. The failing initializer
still owns its partial-resource cleanup contract; this ordering does not imply
that every native allocation failure is recoverable.

On Linux, framebuffer configuration discovery remains a window-creation
prerequisite, but the GLX context is created during initialization. On macOS,
the Metal view/device is prepared before modules; Metal graphics-manager
initialization still owns its RHI setup. SDL and Android keep their existing
creation contracts.

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
scene, so reloads cannot strand old bodies. Both physics backends borrow an
`ISceneManager` supplied at construction; it must outlive the backend.
`PhysicsSceneIntegrationTest` exercises both backends and scene replacement
without constructing an application or window.

Billiard uses the renderer's dynamic font-atlas initialization, like Viewer.
Its opt-in desktop regression loads the real billiard scene, settles it for
60 ticks, applies the normal left-key shot, and follows 120 further ticks.
It requires cue-ball XY displacement of at least 0.1 scene units and movement
of another ball of at least 0.01 units, with finite transforms. Captures use the
authored scene camera, so the regression also exposes default framing changes;
saved UI layouts are unchanged. The default Windows OpenGL build can run
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
Viewer screenshots alone.

The Billiard scene places the camera at `(2.0, 0.0, 1.5)` meters, behind the
cue ball, looking at `(0.0, 0.0, 0.83)` with a 60-degree vertical field of view.
An optional `CameraNode` extension supplies a world-space target:
`Extension (applic = "MyGameEngine", type = "camera_target") {float[3] {{0.0, 0.0, 0.83}}}`.
It requires exactly three finite floats; cameras without it retain the existing
origin target. Six invisible static box cushions follow the four table edges,
with gaps for four corner pockets and two middle pockets. Separate finite
support slabs preserve the existing support height of 0.83 meters while leaving
the pocket openings unsupported. The boxes approximate the modeled pocket
contours; they do not implement scoring, ball removal, jump shots, or high-speed
continuous collision detection. `PhysicsSceneIntegrationTest` loads this actual
scene and checks camera containment, resting support, all six cushions, and
rolling falls through all six pockets before crossing the outer table bounds.

The Viewer uses the Bullet backend. `GeometryObject` accepts these
`MyGameEngine` extensions for physics:

- `collision`: `string {"plane"|"sphere"|"box"|"bvhmesh"}` plus `float {...}`
  parameters (`bvhmesh` takes none and builds a static triangle mesh shape).
- `physics_material`: `float {friction, restitution}`, both non-negative.
- `rigid_body`: `string {"dynamic"|"static"}` plus `float {mass, vx, vy, vz}`.
- `motion_constraint`: `string {"plane"}` plus an axis-aligned unit normal
  `float {x, y, z}`; the body stays in that plane (it can only rotate about
  the normal).

The Viewer's Splash sequence plays a 6.208-second text rotation ending at
90 degrees upright, then holds that pose. The ball remains hidden and neither
it nor the text has a physics body during this introduction. At the animation's
last time key, the Viewer creates the static text collision mesh and the dynamic
ball, and shows the ball. Scene reset or reload restarts the introduction.
After that handoff, Bullet alone drives the ball from left to right against the
actual upright text mesh, without motion constraints, hidden supports or scripted
trajectory corrections. The launch direction follows the text's final heading;
the speed and material values are unchanged. Its start position was tuned with
the real engine to cross the letters without falling away.
The trajectory is chaotic, so changing the text mesh, material values
or Bullet version requires re-tuning; `PhysicsSceneIntegrationTest` checks both
the staged handoff and the real static-text traversal. Runtime visibility is
checked by each rendering backend when drawing, including shadow passes, so the
ball can appear without rebuilding the scene's GPU resources.

## Dependencies
Requires a C++20 compiler and CMake 4.0 or newer. Third-party libraries live in
`External/src` as Git submodules; Git LFS is also used by CI.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party software
copyright and license notices.

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

The Windows clang-cl build script writes to `build/clang-cl`, separate from
the default build tree. The dependency scripts that refresh their CMake
configuration use separate build directories under `External/build` for
native and cross-toolchain builds. `cmake --fresh` resets CMake configuration;
use an empty build directory if you need to rule out all old build outputs.

On Linux, `scripts/coverage_linux.sh [build-dir] [report-dir]` leaves existing
report directories intact. Each run writes profiles, HTML, and a text summary
to its own `run.*` subdirectory. The latest completed LCOV export is published
as `lcov.info` in the report directory for CI.

### Scene resource boundary

Scene construction, parsing, and texture naming do not read files or start tasks.
`SceneObjectTexture` stores a name and a prepared image; changing its name clears
that image. Application entry points inject a `SceneResourceLoader` into
`SceneManager`, alongside the asset reader and scene parser. The resource loader
uses concurrent binary reads and decodes all material channels, skybox faces,
and terrain tiles before the manager publishes a replacement scene. Its asset
reader must support concurrent reads; do not mutate asset search paths during
loading.

Missing texture files still use the shared missing-texture icon and emit a
diagnostic. Unsupported formats, failed decoding, or an unavailable fallback
fail scene loading without replacing the current scene or advancing its revision.
Code that parses a scene directly or changes texture names on an existing scene
must explicitly prepare resources before rendering. `SceneGraph` and
`SceneServices` no longer link asset IO or image decoders. `SceneResources`
depends on `ImageDecoders`; other direct users of decoder headers (including
the GUI icon loader) declare that dependency too.

### Runtime lifecycle and frame traversal

`RuntimeModuleGroup` owns no modules: it tracks registration, initialization,
failure rollback, and reverse-order finalization. Its initialized-module view
is borrowed and must not be retained across registration or lifecycle changes.
`TickFrame` traverses an explicitly supplied ordered view without initializing,
finalizing, or retaining modules. Callers must supply live, initialized modules.

`BaseApplication` uses the group's current initialized-module view on each
frame, preserving the existing registration/update order. Platform event polling,
backend frame setup, simulation cadence, and rendering/presentation order are
unchanged. This checkpoint does not introduce fixed time steps or phased updates.
The standalone runtime regression verifies independent frame selection/order,
failed-initialization suppression, retry, and cleanup without a window or GPU.

### CPU camera preparation

`SceneCamera` prepares view/projection matrices and camera position from an
explicit scene snapshot, canvas dimensions, and projection convention. It links
only scene data and CPU math, not Application, RHI, ImGui, or asset loading.
`GraphicsManager` copies the result into the current frame; camera preparation
no longer looks up the scene through Application.

OpenGL retains its [-1, 1] depth range; the D3D/Metal convention retains [0, 1].
The render canvas, rather than window or framebuffer size, determines aspect.
A scene without a live camera uses the existing built-in view and perspective,
now also publishing the matching camera position instead of a stale value.
Invalid dimensions, projection parameters, and missing/unsupported camera
references are reported and skip rendering that frame without publishing
partial camera data.

### CPU lighting preparation

`SceneLighting` prepares light constants from an explicit scene snapshot and
projection convention, with no Application, RHI, ImGui, or asset loading linkage.
`GraphicsManager` supplies the same frame snapshot used for camera preparation.
Omni, spot, area, and directional lights retain their existing transforms,
attenuation, shadow projections, and shader layout. Directional shadows retain
their camera-relative target and bounded orthographic extent. Shadow-map
allocation and index assignment remain in the rendering passes.

Each successful preparation replaces the light set, so scene replacement and
light-type changes cannot retain previous per-light data. Expired nodes are
skipped. Unresolved references, invalid shadow projection parameters, and more
than `MAX_LIGHTS` live lights are reported and skip rendering that frame without
publishing partial lighting. The headless regression covers all four light
types, backend clip depths, runtime movement, scene replacement, and capacity.
RHI ownership and ImGui lifecycle remain for subsequent checkpoints.

Point-light shadow maps store world-space radial distance normalized by
`POINT_SHADOW_FAR_PLANE`; sampling uses that same scale on every backend.
OpenGL and D3D12 expand each triangle to six cube faces in a geometry shader.
Metal uses six vertex instances and a layered depth attachment instead.
`PointShadowMatrices.hpp` shares the cube-face transforms, accounting for the
backend viewport Y convention. A missing required pipeline is an explicit
error, not a fallback to an unrelated shader.

### Rendering service injection

Viewer and Billiard construct their graphics backend with explicit
`ISceneManager` and `IPipelineStateManager` references. Both services must outlive
the graphics manager; module registration still controls initialization and
frame order. Passes receive that backend instance and the injected pipeline
service directly, without looking them up through `BaseApplication`.

The default graphics constructor intentionally omits scene-driven passes for
standalone graphics tests and clients with their own draw loop. It does not
discover services registered with Application. Both modes still require an
`IApplication` pointer for platform configuration and framebuffer dimensions.

`RenderServices` contains the rendering coordinator separately from the other
managers. `RenderServicesTest` uses injected services, a recording backend, and
an `IApplication` test double without linking the Application implementation or
an RHI. It covers unloaded scenes, one snapshot per frame, scene replacement,
missing-snapshot retry, pipeline/backend routing, and graphics-only clients.
ImGui and the existing UI asset dependencies remain linked until the UI
lifecycle checkpoint; this is not yet a CPU-only rendering coordinator.

### Commit acceptance

A commit is accepted only when its CircleCI pipeline completes the configured
builds and tests successfully on all enabled platforms: currently Windows,
Linux, and macOS. Local tests do not replace this requirement. Android is
currently disabled in the CircleCI workflow and is not part of this gate.

Commits that change only documentation (Markdown files and other prose that no
build, test, or script reads) are exempt: CircleCI is not run for them, and
their commit message carries `[ci skip]`. Mixed commits are not exempt. Such a
commit does not verify a new source revision or supersede an existing record.

After all enabled platforms pass for the exact revision, add
`docs/verification/<full-tested-commit-sha>.md` with the pipeline link, each
platform's job number and result, tested backends, relevant artifact observations,
and limitations. Keep one record per tested revision; amend that record only to
correct or supplement evidence for the same revision. A merge creates a new
revision that needs its own CI result; passing results from either parent are
not transferable.

Do not replace a shared "latest verified revision", append a shared result
index, or update this README for routine acceptance results. Update README only
when durable build instructions, supported configurations, or behavior change.
This keeps parallel branches' verification records independent.

The Windows job uses two parallel CMake build jobs and caches installed dependency
files using their revisions, build scripts, and toolchain details. MSVC may also
compile source files in parallel within a build job.
Cache misses rebuild the dependencies; the engine and tests still run on every
pipeline.

Packaging and package artifacts are limited to `master` and tagged builds;
builds, tests, and Viewer screenshot attempts also run on other branches.

### CircleCI test results

The Windows, Linux, and macOS jobs generate JUnit XML with CTest's
`--output-junit junit.xml` and upload it with `store_test_results` for the job's
Tests tab. The standalone C++/ISPC math regression report is
`build-geommath/junit.xml`; the engine test report is `build/junit.xml`.
Each report is uploaded immediately after its test step, including when tests
fail. Test failures still fail the job. Uploads are not guaranteed if a job is
killed by cancellation or reaches its runtime timeout.

Results represent registered CTest tests, not individual assertions or Viewer
screenshot captures. This reporting does not enable Smarter Testing or change
which tests run.

### Windows
The engine's `windows-vs2026` CMake preset pins the Visual Studio 2026 generator
and uses the repository's `build` directory. CMake Tools, `scripts\build`, and
the commands below use the same preset and output layout. Select
`windows-vs2026` as the configure preset and `windows-vs2026-debug` or
`windows-vs2026-release` as the build preset in VS Code, then run
`CMake: configure (active preset)`. The launch configurations resolve targets
through CMake Tools and build the selected target before starting it. If `build`
was configured with another generator, move it aside before configuring this
preset. Dependency scripts that use Visual Studio still select the newest
version supported by CMake.

The `build_glslangValidator` script retains its name for compatibility and
builds the `glslang` executable.

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

    cmake --preset windows-vs2026
    cmake --build --preset windows-vs2026-debug --target Viewer --parallel 2
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
debug-level mismatches. Select matching libraries with `XG_LIBRARY`,
`OPENGEX_LIBRARY`, and `OPENDDL_LIBRARY` when validating both configurations.
Use a separate build directory when changing generators, not merely when
switching between Debug and Release.

### MacOS & Linux
    ./scripts/build_crossguid.sh
    ./scripts/build_opengex.sh
    ./scripts/build_bullet.sh
    ./scripts/build_cef.sh
    ./scripts/build_glslangValidator.sh
    ./scripts/build_spirvcross.sh
    ./scripts/build_imgui.sh
    ./scripts/build_libispc-texcomp.sh
    ./scripts/build-ninja.sh Debug

Select the `ninja-debug` configure and build presets in CMake Tools to use the
same generator and `build` directory as `scripts/build-ninja.sh`. Use the
`ninja-release` presets or pass `Release` to the script for Release builds.
Run `CMake: configure (active preset)` after preparing dependencies; launch
configurations resolve targets through CMake Tools and build before starting.
The Xcode helper uses a separate `build/xcode` directory.

Manual CMake builds use the matching presets:

    cmake --preset ninja-debug
    cmake --build --preset ninja-debug
    cmake --build --preset ninja-debug --target package

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

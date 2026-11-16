# Android and Vulkan Backend Revival Plan

## Goal

Restore a reproducible Android build, bring up the engine on Android through
Vulkan, and verify real rendering and lifecycle behavior before enabling Android
in the required CI matrix. Vulkan compilation alone is not functional
verification.

## Baseline and verified boundaries

- Planning baseline: source revision
  `c5d87e361638270a603392771dc20184ee02aaf5`; its Windows, Linux, and macOS
  CircleCI jobs passed. The revision-specific record is
  `docs/verification/c5d87e361638270a603392771dc20184ee02aaf5.md`.
- Linux CI found Vulkan and compiled/linked `VulkanRHI`, the Linux Vulkan
  application, and `VulkanRHITest`. It did not run `VulkanRHITest` or render
  through Vulkan.
- `VulkanGraphicsManager` and `VulkanPipelineStateManager` still contain
  placeholder implementations. The current build is not evidence that the
  engine's Vulkan renderer works.
- Android CI job `build_android` exists but is commented out in the workflow.
  Its 2022 toolchain uses CMake 3.23.2 although the root project requires CMake
  4.0 or newer. The Android Gradle project uses AGP 7.2.1, Gradle 7.4.2,
  compile/target SDK 30, min SDK 24, and arm64-v8a.
- Android's CMake path skips `find_package(Vulkan)`, uses a removed-era NDK
  Vulkan include path, and links EGL/GLESv3. `android_main` directly constructs
  `OpenGLESApplication`, so the Vulkan choice in
  `Platform/CrossPlatformGfxApp.hpp` is not used by Android startup.
- The shared `AndroidApplication::Initialize` implementation currently lives
  in `Platform/Android/OpenGLESApplication.cpp`, and
  `APP_CMD_TERM_WINDOW` does not release graphics resources. Extract shared
  Android initialization before making GLES optional and add explicit
  window/surface teardown.
- The Android Billiard target is a native shared library, while Viewer has no
  Android APK project. Android app packaging and the native entry point need to
  be tested together.
- OpenGLES is shared by Android and WebAssembly (`RHI/OpenGL/CMakeLists.txt`).
  Removing ES globally would also remove the current WebAssembly rendering
  path.

## Recommended backend policy

Make Vulkan the Android engine renderer after the required feature set and
device/lifecycle tests pass. Do not remove the shared OpenGLES implementation:
WebAssembly still uses it. Treat removal of Android's EGL/GLES application path
as a separate, gated cleanup after Vulkan is the supported Android path.

Keep the Android GLES fallback decision open until the target-device matrix is
agreed and measured. Android API level alone does not guarantee that a device
has a usable Vulkan driver or the extensions/features the renderer requires.
If the supported device set requires fallback, retain an explicit GLES
fallback; otherwise document the minimum Vulkan/device requirements and fail
clearly on unsupported devices.

## WebAssembly backend assessment

WebAssembly does not mandate OpenGLES. Browser WebAssembly code can call
JavaScript browser APIs, and Emscripten provides a C/C++ WebGPU port. WebGPU is
a realistic additional backend, but it is not a universally available drop-in
replacement for WebGL 2.

The MDN browser-compatibility data queried on 2026-11-16 reports:

| Browser | WebGPU availability in the queried data | Important limits |
|---|---|---|
| Chrome / Edge desktop | Chrome 113+ on ChromeOS, macOS, and Windows; Chrome 144+ on Linux | Linux note limits support to Intel Gen12+ GPUs; Edge mirrors Chromium. |
| Chrome Android | Chrome 121+ | The physical device still needs a compatible WebGPU adapter and driver. |
| Firefox desktop | Firefox 141+, partial implementation | Windows since 141; Apple silicon macOS since 145/147 depending on OS; no Linux or Intel macOS support in the data. |
| Firefox Android | Not available | No WebGPU fallback from this browser. |
| Safari / iOS Safari | Safari 26+, iOS mirrors Safari | Still subject to supported OS/device and adapter features. |

By comparison, MDN lists WebGL 2 in Chrome 56+, Firefox 51+, and Safari 15+,
with corresponding Android/iOS browser entries. WebGPU's adapter can still be
unavailable even where the API is implemented. WebGPU also requires a secure
context; production web delivery should use HTTPS (localhost is suitable for
development). Browser compatibility changes over time, so re-check the live
matrix before setting a product support policy.

Official references:

- [MDN WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [MDN WebGPU compatibility data](https://github.com/mdn/browser-compat-data/blob/bdee0e51289bc0e7aeb0763f315a3a200f23ff7e/api/GPU.json)
- [MDN WebGL 2 compatibility data](https://github.com/mdn/browser-compat-data/blob/bdee0e51289bc0e7aeb0763f315a3a200f23ff7e/api/WebGL2RenderingContext.json)
- [Emscripten: Using WebGPU](https://github.com/emscripten-core/emscripten/blob/2e9f2298d0ba76962543f2ab7192085af677ed0a/site/source/docs/porting/multimedia_and_graphics/WebGPU-support.rst)

Emscripten's current documented route is Emdawnwebgpu, a Dawn-style
`webgpu.h` C API implemented on top of the browser WebGPU JavaScript API. Its
remote port requires Emscripten 4.0.10 or newer and is enabled during both
compile and link with `--use-port=emdawnwebgpu`. Pin the Emscripten SDK and
remote-port version/hash for reproducible CI; do not make a release build
depend on an unpinned download.

The repository's current WebAssembly path is specifically WebGL 2:
`Game/Billiard/CMakeLists.txt` sets `USE_WEBGL2`, `Platform/CrossPlatformGfxApp.hpp`
selects `OpenGLESApplication`, and `RHI/OpenGL/CMakeLists.txt` builds the
OpenGLES RHI for `WA`. The WebGL shader loader expects `Shaders/WebGL/` and
disables compute shaders, while `Asset/Shaders/CMakeLists.txt` currently emits
SPIR-V and desktop GLSL 4.20, not WGSL. No WebGL/WebGL2 shader files are
tracked under `Asset/Shaders` in the examined revision. This means the present
WebAssembly path should not be assumed to be a working renderer merely because
its OpenGLES/WebGL code compiles.

WebGPU is not a browser-exposed Vulkan, Metal, or Direct3D RHI. It is a distinct
web API/RHI: the browser maps it to the native graphics stack. Reusing the
existing `VulkanRHI` is not a viable shortcut. A WebGPU RHI would need its own
device/surface and asynchronous adapter/device startup, resource and render
pipeline code, WGSL shader inputs or a proven SPIR-V-to-WGSL conversion path,
and an ImGui WebGPU renderer backend. Existing HLSL-to-SPIR-V plus
SPIRV-Cross-to-desktop-GLSL does not produce WGSL.

### Recommended WebAssembly follow-up

1. First make the existing WebGL 2 path measurable: build the Emscripten target,
   ensure all required `Shaders/WebGL` assets are generated/packaged, serve it
   from HTTPS/localhost, and inspect a browser screenshot. Keep the existing
   WebGL2 fallback because its browser coverage is wider.
2. In a separate spike, add a minimal Emscripten WebGPU triangle/clear backend
   using a pinned Emdawnwebgpu port. Verify async startup, canvas resize,
   device loss, and screenshots in current supported Chrome/Edge, Safari, and
   Firefox desktop configurations; record unsupported combinations as skips,
   not passes.
3. If the spike succeeds, compare the cost of the missing engine pieces
   (WGSL/shader pipeline, buffers/textures, passes, UI, assets) against fixing
   WebGL2 first. Implement WebGPU as a separate RHI behind the existing
   engine-facing graphics interfaces; select it at build/runtime only where
   `navigator.gpu` and a compatible adapter are present.
4. Do not remove WebGL2 until the project's browser support policy explicitly
   excludes browsers/devices that lack WebGPU and real browser tests cover the
   replacement. If broad browser reach remains a goal, keep both backends.

## Work plan

### Phase 0 — Establish a reproducible Android build

1. Select and pin a mutually compatible Android SDK, NDK, AGP, Gradle wrapper,
   and CMake 4.x combination. Preserve arm64-v8a and the current min SDK unless
   a concrete dependency requires changing them.
2. Update the Gradle external-native-build CMake version and CI image/install
   steps together; remove the obsolete standalone `cmake/android.cmake`
   settings from the active build path.
3. Replace obsolete NDK Vulkan include assumptions with the selected NDK's
   Vulkan headers/loader and the repository's pinned Vulkan-Hpp dependency.
4. Rebuild all Android external dependencies from a clean checkout and confirm
   their scripts support the selected NDK, API level, ABI, and CMake version.
5. Produce and inspect a debug APK with `assembleDebug`; verify the native
   library, manifest, launcher activity, and packaged assets are present.

**Exit criteria:** a clean CI environment can configure and build the arm64
debug APK twice without relying on untracked prebuilt libraries or local SDK
state. No Android runtime support is claimed yet.

### Phase 1 — Android Vulkan platform and lifecycle

1. Add an Android Vulkan application/surface implementation using
   `ANativeWindow` and `VK_KHR_android_surface`; enable the Android Vulkan
   platform declarations and load/create the surface through Vulkan's normal
   extension mechanism.
2. Route `android_main` through the selected graphics backend instead of
   unconditionally constructing `OpenGLESApplication`. Move shared
   `AndroidApplication::Initialize` out of the GLES translation unit so the
   Vulkan-only Android build does not depend on compiling EGL code.
3. Update Android `PlatformInterface` and the Billiard target to compile and
   link the Vulkan application and `VulkanRHI` with the NDK Vulkan loader.
4. Handle `APP_CMD_INIT_WINDOW`, `APP_CMD_TERM_WINDOW`, pause/resume, focus,
   resize, and destruction. Create/destroy the surface and swapchain in the
   correct order and recover when Android replaces the native window.
5. Add explicit initialization errors for missing loader, surface extensions,
   unsupported physical devices, or required features; do not silently fall
   back to an empty renderer.

**Exit criteria:** the APK launches on a Vulkan-capable Android arm64 device,
creates a Vulkan instance/device/surface/swapchain, presents a clear frame, and
survives background/foreground and window recreation without validation errors.

### Phase 2 — Vulkan rendering MVP

Implement the engine-facing backend in dependency order:

1. Load and validate SPIR-V shaders in `VulkanPipelineStateManager`; create
   shader modules, descriptor/pipeline layouts, and graphics pipelines.
2. Implement per-frame command recording, synchronization, acquire/submit/
   present, swapchain resize/recreation, and resource lifetime management.
3. Implement vertex/index buffer allocation and upload; replace the current
   placeholder offsets/returns in `VulkanGraphicsManager`.
4. Implement uniform/descriptor updates, texture upload/sampling, viewport and
   depth state, and a basic lit textured mesh using the project's generated
   SPIR-V assets.
5. Add a deterministic Vulkan scene test and an image/readback or screenshot
   check. Register the Vulkan test with CTest or provide a dedicated CI command;
   the current `VulkanRHITest` is built but not registered/run by CTest.

**Exit criteria:** the same basic Billiard/Viewer scene is visibly rendered by
Vulkan on Linux and Android, with an inspected artifact for each backend. A
successful build or OpenGL screenshot is not a substitute.

### Phase 3 — Feature parity and backend support matrix

Create a feature matrix against the current OpenGL/Metal/D3D12 engine path, then
implement and test the agreed Android MVP features:

- materials and all required texture formats/mips;
- skybox and cubemap/environment textures;
- directional/point-light shadow maps;
- compute passes used by the selected scenes;
- resize, device/surface loss, and resource teardown.

For each feature, add a focused test or a deterministic scene with a captured
result. Mark unsupported features explicitly rather than leaving empty methods
that report success.

**Exit criteria:** every feature required by the Android target is either
implemented and tested or explicitly excluded from the product support
statement. Vulkan errors and unsupported capabilities are observable.

### Phase 4 — Device validation and GLES retirement decision

1. Agree on a small but representative Android device/API matrix, including
   actual Vulkan-capable hardware and at least two driver families where
   available. Record Vulkan API level, extensions, formats, and limits.
2. Test install, launch, asset loading, touch/input, orientation, suspend/
   resume, repeated surface recreation, and sustained rendering. Inspect
   screenshots/readbacks for clipping, black frames, validation errors, and
   rendering regressions.
3. Decide whether Vulkan is mandatory or whether GLES fallback is required for
   the documented device set. Do not decide based on API level alone.
4. Once Vulkan is the supported Android path, remove only Android-specific
   GLES/EGL selection and code that is demonstrably unused. Keep GLES needed by
   WebAssembly unless WebAssembly receives and verifies a replacement backend.

**Exit criteria:** the supported device matrix and fallback policy are
documented; no supported device fails the agreed launch/render/lifecycle suite.

### Phase 5 — CI enablement and acceptance

1. Keep Android CI optional while phases 0–3 are under development. The Android
   job should build the APK and run applicable native/unit tests; add device
   runtime validation only on a runner that can provide a real Vulkan device
   or a proven Vulkan-capable emulator.
2. Enable `build_android` in the workflow only after its clean build and test
   steps are stable. Add Android to the enabled-platform acceptance list in
   `AGENTS.md` only when that job is a required, reliable gate.
3. For each accepted source revision, inspect relevant platform artifacts and
   add `docs/verification/<full-tested-commit-sha>.md` with job results,
   tested renderer/device matrix, visual observations, and limitations.

**Exit criteria:** the exact source revision passes every enabled platform;
Android build and runtime claims match what its CI/device tests actually
exercise.

## Main risks and dependencies

- Vulkan support varies by Android device, driver, supported formats, and
  extensions; min SDK is not a Vulkan capability guarantee.
- Native-window lifecycle and swapchain recreation are mandatory for Android,
  not optional polish.
- Shader reflection, descriptor layouts, texture formats, and compute usage
  must agree with the engine's existing HLSL-to-SPIR-V pipeline.
- CI APK compilation cannot prove GPU rendering. Runtime evidence requires a
  Vulkan-capable device or a validated emulator configuration.
- ES can be retired from Android independently, but not globally while the
  current WebAssembly path depends on it.

## Not included in this plan

- Rewriting the Vulkan backend or Android application in one change.
- Claiming Android support from a successful APK build alone.
- Removing the shared OpenGLES backend before WebAssembly has a replacement.
- Enabling Android as a required CI platform before build and runtime criteria
  are met.

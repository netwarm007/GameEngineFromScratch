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

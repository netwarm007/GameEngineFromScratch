# Vulkan RHI implementation plan

## Status and authority

This is the maintained implementation and acceptance plan for completing the
Vulkan backend. Subsequent work on this task must read this document before
editing, check the relevant checkpoint and invariants while implementing, and
compare the result against its exit criteria before reporting completion.
Update the plan when evidence requires a scope or design change; explain the
change rather than silently diverging from it.

**The upstream dependency review is complete; the point-light pause is lifted.**
Remote master was checked at
`0a20412f26ffd0165faeef89af90d50b8b762f05`, containing both the point-light
shadow fix (PR #67) and stage C ImGui lifecycle separation (PR #66).
Checkpoint 1 is implemented and locally validated, awaiting user confirmation;
checkpoints 2-5 remain unstarted. Do not advance to the next checkpoint without
the user's confirmation. Checkpoints do not require PRs; CI may be used during
implementation with the user's authorization.

The initial investigation used master at
`0f784beebd4143e83e643e398e5140d4b78182c0`. Before checkpoint 1, the local
documentation commit was rebased onto master
`0a20412f26ffd0165faeef89af90d50b8b762f05`, preserving the plan and integrating
both upstream dependencies. Recheck master and preserve local changes before
later integration work; do not silently change the tested baseline.
The historical `redesigned-fiesta` branch, inspected at
`0b19c5714b7d7762650558c82a58c1cdf4c738bf`, is reference material only:
do not merge or cherry-pick its implementation wholesale.

This document records intended work, not completed functionality or acceptance.
The governing architecture and contribution rules are in the
[README](../../README.md#architecture),
[modularization roadmap](modularization-roadmap.md), and
[repository instructions](../../AGENTS.md).

### Reviewed upstream evidence and limits

- [Point-light source and verification record](https://github.com/netwarm007/GameEngineFromScratch/blob/0a20412f26ffd0165faeef89af90d50b8b762f05/docs/verification/8a9ef21e96010b7cf2a72f46d7b03208fcd29cd8.md):
  pipeline 973 records successful Windows D3D12, Linux OpenGL, and macOS Metal
  jobs and inspected Viewer/Billiard artifacts for source
  `8a9ef21e96010b7cf2a72f46d7b03208fcd29cd8`.
- [Stage C verification record](https://github.com/netwarm007/GameEngineFromScratch/blob/0a20412f26ffd0165faeef89af90d50b8b762f05/docs/verification/1ed352fa974f2c73b0f51d6e5f9e449d3d69c72b.md):
  pipeline 968 records successful native three-platform tests and inspected
  artifacts for `1ed352fa974f2c73b0f51d6e5f9e449d3d69c72b`.

The merged production changes and these records were reviewed, not rerun here.
Neither record verifies Vulkan or transfers acceptance to the merge revision.
Point-shadow native screenshots do not exhaustively cover all faces/multiple
lights; the detailed software OpenGL depth probe was a session artifact, not
a committed CI regression. Stage C does not establish native interactive
multi-viewport teardown or SDL/Android input ownership.

## Goal and non-negotiable boundaries

Complete Vulkan as a peer of the existing D3D12 and OpenGL backends by
implementing the engine's existing rendering contracts, not by building a
separate Vulkan renderer.

- Framework retains scene snapshots, CPU camera/light preparation, and pass
  orchestration. Reuse its existing passes and pipeline names.
- GraphicsManager maps those rendering operations to the backend.
- RHI owns GPU resource operations, commands, synchronization, and submission.
- Platform owns native window/connection resources and surface integration,
  coordinating the existing initialization and reverse shutdown sequence.
- Applications continue to supply scene and pipeline services explicitly.
- GraphicsManager retains ownership of its offscreen targets; Frame borrows
  their descriptors. Do not redesign shared ownership to simplify Vulkan.
- Preserve existing default backends and behavior. Vulkan selection is explicit.
- Do not introduce a Vulkan-only scene pipeline that bypasses shared passes,
  duplicate shader algorithms, or report empty implementations as supported.
- Reuse the completed stage C ImGui ownership boundary; do not reimplement or
  expand that refactor. Do not combine this task with general engine
  modularization, a new frame scheduler, a render-graph rewrite, or unrelated fixes.

## Initial findings

On the investigated master, VulkanGraphicsManager has many empty operations,
VulkanPipelineStateManager does not actually load shaders, and VulkanRHI exposes
mostly a single-model demonstration path. Native Vulkan initialization also
differs between platforms. Completing only presentation or one textured model
does not establish engine integration.

The historical branch added separate Vulkan scene, shadow, skybox, and BRDF
shaders and a dedicated "Vulkan Scene Geometry" pipeline. These are not the
architecture to carry forward. Individual resource or synchronization fixes
may be reused only after checking their applicability to the current design.

## Shared shader source and ABI

Use the existing [HLSL sources](../../Asset/Shaders/HLSL/) and
[shader build](../../Asset/Shaders/CMakeLists.txt):

```text
Shared HLSL
  +-- DXC ------------------------> D3D12 bytecode
  +-- glslang -------------------> SPIR-V
                                     +-- Vulkan consumption
                                     +-- SPIRV-Cross --> GLSL
                                     +-- SPIRV-Cross --> Metal
```

Keep the current glslang/SPIRV-Cross toolchain. A compiler migration is not part
of this task. Vulkan may need a target-specific SPIR-V variant from the same
source; it must not require copies of PBR, shadow, skybox, or BRDF algorithms.

Before implementing pipeline layouts:

1. Inspect generated SPIR-V and define a deterministic mapping from HLSL
   `b`, `t`, `s`, and `u` registers to descriptor sets/bindings without namespace
   collisions. Validate stage compatibility and descriptor types.
2. Verify CPU/GPU constant-buffer layout, matrix major order, device alignment
   requirements, vertex locations, and shader stage interfaces.
3. Keep Vulkan-only binding or coordinate adaptations out of the intermediate
   artifacts used by OpenGL and Metal. Prefer compiler options; use narrow
   target conditions in shared source only where necessary.
4. Load the existing pipeline definitions rather than substitute Vulkan scene
   shader names. Missing files, incompatible layouts, and pipeline failures
   must produce explicit errors, not fallback success.

Use fixed layouts validated against compiler output initially; do not introduce
a general-purpose runtime reflection framework without a demonstrated need.

### Implemented shader ABI (checkpoint 1)

[VulkanShaders.cmake](../../cmake/VulkanShaders.cmake) keeps the common shader
manifest and compiles Vulkan 1.1-targeted variants into
`Asset/Shaders/Vulkan/Engine/`. These variants do not replace the legacy
`Asset/Shaders/Vulkan/*.spv` intermediates consumed by GLSL/Metal conversion or
the existing standalone Vulkan demo. Shared HLSL algorithms and legacy
conversion commands are unchanged.

The fixed mapping is recorded in
[VulkanShaderBindings.hpp](../../RHI/Vulkan/VulkanShaderBindings.hpp) and checked
against reflection of actual compiler output:

| HLSL register class | Vulkan set | Binding |
| --- | --- | --- |
| Constant buffers `bN` | 0 | N |
| Sampled images `tN` | 0 | 32 + N |
| Samplers `sN` | 0 | 64 + N |
| Storage images `uN` | 0 | 96 + N |

Current shaders fit within each reserved 32-binding range. Per-frame,
per-batch, light, and pass constants use bindings 10, 11, 12, and 13.
Debug and shadow constants alias binding 13 in separate pipelines, not in the
same shader module. The standalone simple-model demo retains its different
`b0`/`t1`/`s1` contract.

The tests compare reflected buffer sizes, all reflected CPU-buffer member
offsets, matrix stride 16, Light array stride 288, and shadow-matrix array stride
64 with the actual C++ types. They check descriptor names/bindings/image types,
PBR vertex locations, and producer/consumer stage locations/types. GPU readback
also checks non-symmetric matrix transforms and the second Light/last shadow
matrix, rather than assuming reflected matrix-major decorations establish
correct multiplication semantics.

The instanced omni variant declares vertex-layer capability; compilation alone
does not establish device support. The local glslang 11.5 compiler also emits
`SampledCubeArray` capability for compute shaders that include the shared
declarations. The compute probe explicitly requires and enables `imageCubeArray`;
device setup in checkpoint 2 must honor emitted shader capabilities, not only
the resources remaining active after optimization.

### Coordinates and integrated point-light contract

Preserve shared CPU camera and light preparation and Vulkan's `[0, 1]` depth
range. Prefer viewport/raster state for Y orientation and winding compensation;
negative-height viewports are the initial candidate, not an unverified contract.
Verify rendered orientation, shadow projection/sampling, cubemap face
directions, culling, and UI preview UVs together to avoid double inversion.

The integrated fix establishes the contract Vulkan must preserve:

- The shadow fragment shader consumes an interpolated world position, not
  raster `SV_POSITION`, and writes distance from the light divided by
  `far_plane`. Supply `POINT_SHADOW_FAR_PLANE` (currently 1000) and reconstruct
  with that same shared constant in the production sampling function.
- Reuse `BuildPointShadowMatrices` from
  `Framework/GeomMath/PointShadowMatrices.hpp`: face order is +X, -X, +Y, -Y,
  +Z, -Z, with the helper's matching up vectors. Determine its `flipY` argument
  together with Vulkan viewport/cubemap conventions through GPU tests; do not
  blindly copy either the D3D12 or Metal setting.
- Keep allocation in `PrepareSceneLights` before upload. `ShadowMapPass`
  consumes each light's published index; Vulkan must not renumber it.
- The shared geometry shader writes `cubeIndex * 6 + face` for cross-compiled
  output. A full-array attachment must retain that absolute layer convention;
  a six-layer view based at `cubeIndex * 6` requires relative face output.
  Never apply the cube offset twice.

Windows/Linux devices with geometry shader support should consume the existing
shared omni vertex/geometry/fragment stages. Geometry SPIR-V is now built by
the asset target; no Vulkan shadow shader copy is needed.

Metal's integrated path uses `shadowmap_omni_metal.vert.hlsl`: six vertex
instances, relative layer output, a six-layer depth attachment starting at
`cubeIndex * 6`, explicit triangle topology, and constants available to both
vertex and fragment stages. Its fragment shader and radial-depth algorithm
are shared with the geometry path.

For MoltenVK, reuse that HLSL stage where Vulkan vertex-layer output capabilities
permit it; native Metal support alone does not prove Vulkan support. Otherwise
use six face draws targeting single-layer views, with a narrowly scoped shared
HLSL entry/variant supplying world position and the selected face transform.
This is a stage/capability adaptation, not another lighting/depth algorithm.
The current registry selects the Metal stage using `OS_MACOS`; Vulkan pipeline
loading on macOS must handle that selection explicitly and validate its SPIR-V
capabilities instead of expecting a geometry shader or a Metal binary.

Extend committed regressions to check production shader GPU depth readback on
all six faces, translated lights, at least two independent cube indices, and
occluded/unoccluded sampling. Include a negative control for the old
screen-position depth or incorrect reconstruction scale where feasible.
Keep existing shader stage-link and shared matrix tests.

## Functional parity

Use the [pipeline registry](../../Framework/Manager/PipelineStateManager.cpp),
[graphics interface](../../Framework/Interface/IGraphicsManager.hpp), actual
application call paths, and current D3D12/OpenGL behavior as the checklist.
Distinguish existing limitations from functionality this task must implement.

| Area | Required result |
| --- | --- |
| Geometry/materials | Vertex/index buffers, draw batches, material textures, BASIC and PBR pipelines, runtime transforms and visibility. |
| Environment lighting | Skybox, prefiltered cubemap data, and BRDF LUT compute using shared shaders. |
| Shadows | Existing point, spot, directional, and area-light paths with 2D/cube arrays, preserving the integrated world-space radial-depth and layer-index contracts. |
| Render targets | Offscreen color/depth, MSAA, resolve, and final presentation. |
| Textures | Upload/update, supplied and generated mip chains as appropriate, arrays/cubes, and debug views; handle format capabilities explicitly. |
| UI | Dynamic font atlas, icons, scene preview, shadow/texture debugging, docking, saved layouts, and multi-viewports where the native input backend supports them. |
| Lifecycle | Partial initialization cleanup, scene replacement, resize, minimize/restore, and normal shutdown. |
| Compute | Existing compute pipeline contracts; do not replace the compute raytrace shader with Vulkan ray-tracing extensions or enable dormant passes merely for this task. |

Preserve material versus prefiltered-environment mip semantics. Unsupported
formats/features require an explicit diagnostic or a tested, intentional
conversion path, not missing textures or a silently disabled pass.

## Implementation responsibilities

### VulkanRHI

Implement instance/device setup, capability selection, queues, memory and
resources, upload, command pools/buffers, barriers, submission, synchronization,
swapchain management, and readback. Begin with a graphics queue that also
supports compute; do not require asynchronous compute, dedicated transfer
queues, bindless resources, or Vulkan 1.3-only features without a concrete need.

Distinguish frame slots from acquired swapchain image indices. Track completion
of submission separately from presentation semaphore reuse. Handle image state
at the required mip/layer granularity. Protect shared depth, MSAA, and shadow
resources across in-flight frames instead of assuming they are frame-private.
Use scoped synchronization rather than device-wide idle on every normal frame.

Partial initialization must release its own allocations; finalization must be
safe for the supported failure stages and repeated cleanup. Preserve the
ordering of GPU completion, module cleanup, backend destruction, and native
window destruction, including application rollback.

### ImGui ownership inherited from stage C

- Viewer owns the context through `GuiContext` and configures its CPU font
  atlas. Keep that context current and alive through application finalization.
  VulkanRHI native initialization must not create or require an ImGui context.
- BaseApplication initializes graphics resources, then `InitializeGuiPlatform`,
  then runtime modules. Reuse Windows/Cocoa/Xcb input ownership; Cocoa must bind
  the final content view after graphics setup. Window creation alone binds no
  ImGui backend.
- VulkanGraphicsManager owns the Vulkan ImGui renderer, GPU font resources,
  and renderer descriptor infrastructure. Reject missing/already-bound contexts
  without replacing borrowed state, propagate initialization failures, clean
  partial allocations, and shut down only a successfully initialized renderer.
- Wait for relevant Vulkan submissions before releasing draw passes/UI
  textures. Release icons/debug views while their descriptors, renderer, and
  device are live; then shut down the renderer and release its descriptor pool.
  Application teardown then releases GUI platform state, native graphics,
  window resources, and finally the context.
- Successful platform input state survives module rollback for retry; renderer
  state owned by successfully initialized modules does not. Cover failure,
  retry and repeated finalization without claiming full device-loss recovery
  or whole-module-graph restart.

Reuse ApplicationLifetimeTest, GuiSubPassLayoutTest, and RenderServicesTest;
add Vulkan-native renderer binding/descriptor lifetime regressions modeled on
D3d12GuiTextureTest. Windows/macOS floating viewport lifecycle needs explicit
Vulkan validation. Xcb currently disables ImGui viewports; preserve and report
that limitation rather than expand Linux input support in this task.

### VulkanPipelineStateManager and VulkanGraphicsManager

The pipeline manager loads shared shader outputs and builds graphics/compute
pipelines from existing pipeline definitions. The graphics manager implements
existing passes, scene resources, per-frame/per-batch/light binding, and UI
rendering. Do not grow a second scene traversal or pass scheduler inside RHI.

### Shared code and native integration

Expected changes outside Vulkan are limited to explicit backend selection,
linking/resource packaging, narrowly scoped shared-shader adaptations, native
surface setup and failure propagation, and UI texture adaptation if necessary.
Fix Vulkan surface ABI and initialization inconsistencies where required.

First try to satisfy the existing UI texture contract inside Vulkan. If an
interface change is necessary, document why, keep other backends compatible,
and test all affected call sites. Do not disable docking, supported
multi-viewports, texture debugging, or saved layouts to get an initial screenshot.

## Checkpoints and exit criteria

Checkpoint 0 and baseline integration are complete. Checkpoint 1 has passed
focused local validation and awaits user confirmation, not full cross-platform
acceptance. Checkpoints 2-5 remain planned and proceed in order, with focused
validation and user confirmation at each boundary.

| Checkpoint | Deliverables and exit criteria |
| --- | --- |
| 0. Review upstream dependencies | Complete: reviewed merged point-shadow/stage C changes and integrated master `0a20412f26ffd0165faeef89af90d50b8b762f05` while preserving the plan. |
| 1. Shader and ABI | Implemented; local checks passed, awaiting user confirmation. Shared shader variants, fixed bindings, C++ layout/stage-interface checks, capability inspection, and GPU ABI/BRDF readback are present. Cross-platform/native graphics acceptance is not established by this checkpoint. |
| 2. Device/resource loop | Implement initialization and selected failure cleanup, buffers/textures/upload, command submission, synchronization, presentation, resize, and minimize/restore. Add bounded automated native tests. |
| 3. Shared rendering passes | Integrate offscreen rendering, BASIC/PBR, skybox/BRDF, shadows, and MSAA in that order. Verify six-face/multi-light depth and sampling, render output, and resource correctness without a Vulkan-only pass bypass. |
| 4. UI and applications | Add explicit Vulkan selection for Viewer/Billiard without changing defaults. Follow stage C ownership; test renderer rejection/failure/rollback/retry/descriptor cleanup, fonts, icons, previews, debug views, supported multi-viewports, and application-only screenshot readback. |
| 5. Cross-platform acceptance | Complete original-backend regressions and explicit Vulkan coverage; inspect actual artifacts and record exact-revision evidence and limitations. |

At the start of each checkpoint, identify the production paths and focused
tests. At its end, review the actual diff against this plan, record any justified
deviation here, and do not claim the next checkpoint's functionality early.
Build/test only the smallest relevant targets initially, escalating as needed.

### Checkpoint 1 validation and reproducible commands

The focused suite can run without the engine's full asset/window/dependency
build, but needs CMake 4+, C++20, the pinned crossguid submodule, Vulkan headers
and loader, glslangValidator, SPIRV-Cross, spirv-val, and spirv-dis. The compute
test additionally requires a Vulkan 1.1-capable driver and the Khronos
validation layer; absence is an explicit failure, not a skipped success.

```text
cmake -S Test/RHI/Vulkan -B build-vulkan-abi -DCMAKE_BUILD_TYPE=Debug
cmake --build build-vulkan-abi --config Debug
ctest --test-dir build-vulkan-abi -C Debug --output-on-failure
```

In the full engine build, opt in with `-DVULKAN_RHI_TESTS=ON`; ordinary native
backend builds do not gain an unannounced Vulkan driver requirement merely
because a Vulkan SDK is installed. Tools may be supplied through the CMake
`GLSL_VALIDATOR`, `SPIRV_CROSS`, `SPIRV_VAL`, and `SPIRV_DIS` paths. Standalone
builds compile crossguid from its pinned source; no new dependency is introduced.

Focused Windows Debug checks passed with the installed Vulkan SDK 1.2.182
(glslang 11.5) on AMD Radeon(TM) Graphics:

- All 24 production shader stages passed SPIR-V validation, reflected ABI and
  relevant producer/consumer interface checks, including both omni stage paths.
- TEST_VulkanCompute passed actual uniform/matrix readback and production
  `integrateBRDF.comp` dispatch over a 512x512 LUT. Four distributed sample
  positions matched an independent double-precision CPU integration within
  0.001 per RG channel; repeated BA channels were also checked.
- Khronos core and synchronization validation reported no errors during the
  compute probe, including device/resource destruction.
- A negative control compiled PBR without binding shifts: TEST_VulkanShaderAbi
  failed on the old bindings; restoring the native variant passed both tests.
- The existing ShadowShaderCompatibility script passed GLSL stage linking and
  MSL generation with the installed tools.

Limits: the initial full engine configure stopped at missing prebuilt Debug
zlib, so no complete engine build, DXC regression, native Metal compilation,
Linux/macOS execution, or CircleCI acceptance is claimed here. The standalone
probe exercises the shader ABI directly, not the unfinished VulkanRHI frame or
resource implementation. Uniform descriptors use offset zero; dynamic uniform
stride and device-specific allocation/lifetime behavior remain checkpoint 2
work. Vertex-layer execution, graphics pipelines, shadow rendering, actual
Viewer/Billiard images, and portability fallback are not verified yet.
No plan deviation is needed: the headless compute harness is deliberately
independent of the device/window implementation reserved for checkpoint 2.

## Platform and acceptance strategy

Desktop scope is Windows and Linux Vulkan plus macOS Vulkan through MoltenVK.
Establish the main path on Windows/Linux, then complete portability adaptation
on macOS. Android and WebAssembly are outside this task's acceptance scope.

Keep two separate kinds of evidence:

1. **Existing-backend regression:** Windows D3D12, Linux OpenGL, and macOS Metal
   retain their configured builds, tests, and visual behavior.
2. **Vulkan verification:** explicitly selected Vulkan builds and runtime tests,
   shader compilation/SPIR-V validation, checked GPU readback, synchronization
   validation, resize/reload/shutdown, and actual Viewer/Billiard images.

Use [.circleci/config.yml](../../.circleci/config.yml) as the platform command and
dependency source of truth. Extend coverage deliberately rather than infer
Vulkan success from native-backend jobs. Software Vulkan is useful CI coverage,
but is not hardware-driver verification. Missing or unavailable platform
coverage remains unknown/blocked and must not be reported as passed.

Inspect screenshots from the actual backend and exact revision: scene content,
materials, environment lighting, shadow placement, texture orientation, and
visible/non-overlapping/unclipped UI. Reuse GuiSubPassLayoutTest for layout
regressions; headless success does not replace rendered-image inspection.
Verify Billiard's configured physics regression and images, not Viewer alone.
Capture only the application, not the shared desktop.

Before acceptance, every enabled CircleCI platform must finish its configured
build and tests successfully for the same source revision, and required visual
artifacts must be inspected. Pending, skipped, canceled, failed, or missing jobs
do not meet the gate. Report the pipeline link, full revision, platform job
results, tested backends, observations, and limitations.

Commit/push/pipeline execution requires user authorization. If a PR is requested,
complete acceptance first. After passing, write the exact-revision record under
[docs/verification](../verification/); preserve other revisions' records.
Do not use this plan or README as a rolling acceptance log. Documentation-only
commits use `[ci skip]` and do not require CI or verify a new source revision.

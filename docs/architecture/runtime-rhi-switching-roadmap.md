# Runtime RHI selection and switching roadmap

## Status and scope

This is the maintained, incremental implementation roadmap.
Engineering lessons from this effort are curated in
[engineering-lessons.md](../engineering-lessons.md), split into long-term
principles and version-dependent practical notes (the latter must be
re-verified when their dependency changes).
The user requested a path from compile-time selection to building all supported
backends for a platform and switching between them at runtime, and authorized
stepwise implementation after reviewing the draft. M0 adopts the first-release
contract below; M1's build/asset-staging foundation has passed full native
three-platform CI and artifact review. See its
[exact-revision verification](../verification/9e91a93eee1a80337165af7e40a5ea24d0f68acf.md)
for the tested scope and distribution limitations. M2.1's startup-selection
contract and M2.2's Vulkan loader isolation are implemented and verified at
their headless boundaries. M2.3's owned startup composition, M2.4 Viewer and
M2.5 Billiard runtime selection, the informative window titles, and M2.6
(Viewer + Billiard per-backend startup matrices, RHI diagnostics smoke, Vulkan
coverage migration, and retirement of the specialized Vulkan job) are
implemented and verified by full native three-platform CI. The M2 startup
selection checkpoint is complete. M3 (isolated reconstruction) is complete:
the session contract, RHI injection, host/backend boundary, owned session
composition with same-backend reconstruction, hardening, and the native
reconstruction matrix are implemented and verified by full native
three-platform CI (pipeline 1138). The M3 checkpoint is human-confirmed;
the integration merge into `feat/runtime-rhi-build` is verified separately
(pipeline 1139). M4 (cross-backend switching) is in progress under the
execution plan below. UI rollout (M5) remains a separate checkpoint
requiring human confirmation. Deliverable acceptance is the later all-roadmap +
three-platform-CI + PR gate.

The working branch was fast-forwarded to master
`175d3567bb032e5356a8b537b5c5bb5f50edc6b2` before preparing this draft.
This identifies the inspected baseline, not a new acceptance result.

Follow the [repository instructions](../../AGENTS.md),
[modularization roadmap](modularization-roadmap.md), and
[Vulkan implementation plan](vulkan-rhi-implementation-plan.md).
Reuse their existing ownership boundaries; this effort does not reopen completed
scene, input, physics, or ImGui refactors. Vulkan's outstanding acceptance work
remains governed by its own plan.

## First-release contract

- Build and package every backend in the agreed desktop platform matrix into
  one application. Build selection controls inclusion, not the active backend.
- First deliver startup selection, then switching within the same process.
  Viewer is the first integration client; Billiard follows using the same
  composition mechanism, not duplicated switching logic.
- Allow a rendering pause and, where necessary, recreation of the native
  window or content view. Seamless switching is not a first-release guarantee.
- Preserve the current CPU scene, prepared assets, camera, physics, animation,
  game state, ImGui context/font data, and saved/custom UI layout. Do not reload
  the scene or initialize the whole runtime merely to change backends.
- Pause simulation during switching without accumulating a catch-up interval.
  Continue necessary platform event processing; discard stale window events
  and reconcile held input when focus/window ownership changes.
- Recreate every backend-owned GPU resource. Never transfer native texture,
  buffer, pipeline, descriptor, draw-batch, or ImGui texture handles.
- Preserve application-specific defaults initially: Viewer uses D3D12 on
  Windows, OpenGL on Linux, and Metal on macOS; Billiard's ordinary Windows
  default is OpenGL, although CI explicitly selects D3D12.
- Explicit startup selection and an explicit switch must not silently choose
  another backend. Report unavailable backends and initialization failures.
  Initially, an unavailable default also fails explicitly; an automatic
  fallback policy is a separate decision, not an implicit factory behavior.
- If a switch fails before retirement, keep the old session. After retirement,
  attempt to reconstruct the previous backend once. Report the failed switch
  even when recovery succeeds. If recovery fails, stop rendering and exit
  orderly with both errors; do not reuse the retired session.

Dynamic library/plugin unloading, dual-device prewarming, zero-black-frame
transitions, simultaneous rendering with multiple RHIs, device-loss recovery,
Wayland integration, and new mobile/web backends are outside this first release.

## Platform and availability matrix

Distinguish compiled inclusion, runtime prerequisites, and successful
initialization. A library target or installed SDK does not establish native
application support. A preliminary probe cannot guarantee initialization.

| Platform | Proposed complete desktop build | Limits |
| --- | --- | --- |
| Windows | D3D12, OpenGL, Vulkan | Probe each backend's device, driver, and presentation requirements. |
| Linux | OpenGL, Vulkan | Use native XCB for the future switching host; M1 preserves current SDL/OpenGL startup when SDL is available. |
| macOS | Metal, Vulkan through MoltenVK | The OpenGL library target alone does not establish a supported native application path. |
| Android | Not a supported build of this effort | Unmaintained for years and disabled in CI; the existing OpenGL ES path predates the runtime architecture. Revival requires refactor/rewrite against the host/backend composition model (known limitation), not re-enabling the old build. |
| WebAssembly | Not a supported build of this effort | Unmaintained for years and disabled in CI; the existing WebGL/SDL path predates the runtime architecture. Revival requires refactor/rewrite (and a separate WebGPU decision) before any support claim (known limitation). |
| FreeBSD and other targets | Preserve existing behavior | No new runtime-switching support claim without native evidence. |

D2D and Empty remain available for their existing specialized/test uses; they
are not advertised as full scene-rendering peers. Vulkan native availability
must retain the maintenance-extension and presentation-retirement requirements
in its implementation plan. Headless Vulkan is not a fallback for native WSI.

A complete desktop build should fail configuration when required dependencies
are missing, rather than silently omit a matrix backend. An explicitly trimmed
developer build may omit backends, but must expose its compiled set and cannot
satisfy complete-build acceptance.

## Current implementation boundaries

- [RHI targets](../../RHI/CMakeLists.txt) already build multiple libraries.
  [Viewer composition](../../Viewer/main.cpp) and
  [application linking/packaging](../../Viewer/CMakeLists.txt) still select one
  backend through preprocessor conditions and CMake options.
- D3D12 graphics operations obtain the RHI through `D3d12Application` casts.
  Metal graphics initialization obtains its device/view through the window.
  Vulkan already injects its RHI into graphics/pipeline managers.
- [BaseApplication](../../Framework/Manager/BaseApplication.cpp) coordinates
  full shutdown, including platform resources.
  [RuntimeModuleGroup](../../Framework/Manager/RuntimeModuleGroup.hpp) borrows
  modules, rejects running registration, and clears registrations on finalization.
  Neither currently provides isolated render-session replacement.
- GraphicsManager owns its offscreen targets and passes; frames borrow resource
  descriptors. Scene texture objects retain prepared CPU images. Reuse these
  boundaries for reconstruction without broadening shared GPU ownership.
- ImGui context, platform integration, renderer integration, and GUI textures
  already have separate owners. Renderer shutdown must release secondary-window
  renderer resources before context/device destruction.
- OpenGL projection correction is selected at startup by `IS_OPENGL`.
  Render conventions must instead follow the active session, including camera,
  shadow, depth, and shader semantics.
- Shared HLSL already produces backend-specific shader artifacts. Package all
  required variants without copying algorithms or changing other variants' ABI.

## Target ownership and switching sequence

The application composition root retains long-lived CPU services and an owned,
replaceable render session. That session groups the backend lifecycle, graphics
and pipeline managers, passes, GPU resources, and ImGui renderer. A platform host
retains event routing and native window ownership, with explicit backend hooks
for drawable/view/window reconstruction and GUI platform rebinding.

This is an ownership boundary, not a requirement for a new universal GPU API,
service locator, configurable scheduler, or interface per resource. Use existing
interfaces and narrowly scoped factories. Static linking and explicit factory
registration are sufficient initially.

Proposed state sequence:

```text
Running -> Preflight -> Quiescing -> Retiring -> Creating -> Restoring
        -> FirstFrame -> Running

Preflight failure -> Running with old session and reported error
Failure after retirement -> RecoverPrevious -> Running with reported error
Recovery failure -> OrderlyExit with both errors
```

Process one request at a complete frame boundary, after main and secondary
viewport submission. Reject overlapping requests with an explicit result;
selecting the already active backend is a no-op.

Wait for all relevant render/compute/presentation work before retirement.
Release UI textures/views, scene allocations, offscreen resources, renderer,
and pipelines while their required device/context is alive. Unbind GUI platform
callbacks before destroying/replacing native windows or views. Then release
backend-native resources, including partial initialization state.

Create the target drawable/window integration, backend, GUI platform binding,
pipelines, renderer/fonts, and scene GPU resources in dependency order. Activate
its render conventions before preparing camera/light data. Publish success only
after the restored scene has completed a rendered/presented frame; a minimized
or zero-size window must not produce a false success or a busy wait.

Use a render-session generation independent of scene revision to invalidate
borrowed GPU references and UI caches. A switch reconstructs the same Scene
without pretending it is a newly published CPU scene. Persist UI interaction
state separately from GPU-owned pass instances where reconstruction would
otherwise reset it.

Window recreation is an intentional compatibility option: GLX visual selection
precedes window creation, WGL pixel format has window-lifetime constraints, and
Metal/Vulkan use different Cocoa view integrations. Preserve logical window
geometry, DPI-aware layout, and floating-panel placement, not native handles.

## Implementation checkpoints

Only the top-level milestones (M0–M5) are **checkpoints**. Each checkpoint
depends on the preceding one and, after a self-check, requires human
confirmation before proceeding to the next checkpoint; confirming one
checkpoint does not authorize the remaining checkpoints. The sub-increments
within a checkpoint (M#.#, such as M2.1–M2.6) are **work steps**, not
checkpoints: after completing one, self-check that its goal is achieved (CI may
be submitted to verify) and continue without a human confirmation stop, until
the parent checkpoint is complete. **Acceptance (验收) is deliverable
acceptance**, distinct from per-revision CI verification: it normally requires
all roadmap tasks to be complete, the three-platform CI to pass, and a pull
request to be created. Per-revision full-CI runs are verification evidence
(recorded under `docs/verification/`), not acceptance.

| Checkpoint | Deliverables | Exit criteria |
| --- | --- | --- |
| M0: agree contracts | Confirm matrix, defaults, Linux host choice, pause/state/window policies, availability/error model, and ownership order. Identify scoped backend failure paths to harden. | Reviewable contracts and testable acceptance cases; no implementation claims. |
| M1: complete build and package foundations | Link the agreed backend set; stage all shader/material/font/scene assets for build-tree and installed/bundle execution. Separate inclusion options from selection. | Both application build graphs and packaging helpers include the required backend libraries/assets; existing default launch remains unchanged. Final executable availability follows at M2 and native distribution rollout at M5. |
| M2: startup selection | Add explicit factories, availability diagnostics, `--rhi` and `--list-rhi`; remove conflicting entry-point aliases/macros and concrete application casts as required. | The same executable launches each matrix backend without rebuilding; missing support/assets and initialization errors fail explicitly. |
| M3: isolated reconstruction | Separate persistent runtime services from the render session and native backend hooks; maintain safe scheduling/borrowed lifetimes. Harden partial cleanup required by reconstruction. | Same-backend destruction/recreation preserves CPU state; repeated cleanup and failure recovery are covered without whole-runtime restart. |
| M4: cross-backend switching | Implement frame-boundary state machine, resource generation/invalidation, render-convention updates, window/view rebinding, and one-shot previous-backend recovery. | Native A/B/A transitions restore the scene; failed switches preserve or reconstruct the previous backend, or exit explicitly. |
| M5: UI and application rollout | Viewer selection/status/error UI; preserve docking/floating state; reuse composition in Billiard; integrate complete acceptance matrix. | All supported directed transitions have native evidence; installed applications, interactions, and backend-specific visuals are checked. |

M2 may construct a selected concrete application for startup. M3 must remove
that application-type dependency from the live switching path; replacing the
application and invalidating every module's borrowed app pointer is not the
target design. Preserve current frame participant order when introducing the
session boundary instead of reopening frame-scheduler design.

### M2 execution plan

Implement and verify these work steps separately. The first work step
establishes the shared contract; it does not advertise runtime selection in
applications before their factories and loader isolation are ready. Work steps
are self-confirmed by their full-CI acceptance; only the parent M2 checkpoint
requires human confirmation before moving to M3.

| Work step | Work | Verification boundary |
| --- | --- | --- |
| M2.1: selection contract | Canonical backend identifiers, central `--rhi`/`--list-rhi` parsing, availability diagnostics, and explicit/default selection without fallback. | Headless tests of production parsing/resolution; no native startup or backend retention claim. |
| M2.2: Vulkan loader isolation | Resolve Vulkan entry points through a controlled late-loaded loader, including platform calls and ImGui dispatch. | Missing Vulkan runtime does not prevent non-Vulkan startup; loader/dispatch failures are explicit. Preserve strict native WSI requirements. |
| M2.3: owned startup composition | Add typed, explicit backend factories and owned application/graphics/pipeline composition. | Factories retain each included implementation and clean up failed construction without conflicting configuration aliases. |
| M2.4: Viewer integration | Consume RHI arguments before application arguments, construct the selected backend, derive render conventions from it, and use the Linux XCB host. | Same Viewer executable starts each platform backend; scene/screenshot arguments and defaults remain valid. |
| M2.5: Billiard integration | Reuse the composition and argument contract; preserve its application-specific default and dispatch capture through the active backend. | Same Billiard executable starts each backend; physics/capture regressions do not invoke OpenGL for another backend. |
| M2.6: native acceptance and CI consolidation | Run the startup matrix and integrate existing Vulkan coverage into platform jobs, replacing separate Vulkan application builds. | Full exact-revision platform acceptance and inspected artifacts; retire the specialized job only after equivalent coverage is verified. |

M2.1's [RhiStartup](../../Framework/Manager/RhiStartup.hpp) production library is
implemented and covered by the existing Runtime suite's `fast`
[startup-contract regression](../../Test/Runtime/RhiStartupTest.cpp). It neither
probes devices nor creates applications; Viewer and Billiard still use their
existing startup paths. This is not completed M2 or native acceptance.

M2.2's [VulkanLoader](../../RHI/Vulkan/VulkanLoader.hpp) late-loads the Vulkan
runtime and resolves entry points for Vulkan-Hpp dynamic dispatch, platform WSI
surface creation, and ImGui dispatch under a single active dispatch context.
A missing or incomplete runtime fails explicitly and does not prevent
non-Vulkan startup; the `fast`
[loader regression](../../Test/Runtime/VulkanLoaderTest.cpp) exercises absent
and incomplete runtimes and exported WSI trampolines without a driver. Full
native CI preserves the existing Vulkan WSI coverage through the loader.
Windows native Vulkan startup evidence remains M2.6 scope. See the
[exact-revision verification](../verification/809aacb59e1ea0a4229228b51d872636ca20536c.md).

M2.3's [RuntimeComposition](../../Runtime/RhiComposition.hpp) library adds
typed, explicit per-backend factories and owned application/graphics/pipeline
composition. The central dispatch references every included factory so static
archives retain each implementation; backends that are not compiled fail
explicitly without fallback. Construction is window/GPU-free and `Initialize()`
remains the application's step; the `fast`
[composition regression](../../Test/Runtime/RhiCompositionTest.cpp) covers
repeated construction/destruction, absent and invalid backends,
failed-construction cleanup, and availability probing with Vulkan loader
evidence. It also exposed and fixed indeterminate D3D12 constant-buffer array
members that crashed uninitialized destruction on Windows Debug. Viewer and
Billiard still use their existing startup paths; runtime selection is not yet
advertised. See the
[exact-revision verification](../verification/b8e0158b605571ad8aab775b3f85efbcf4be4408.md).

M2.4 wires Viewer to the runtime selection contract: `--rhi`/`--rhi=` and
`--list-rhi` are consumed before application arguments, the backend resolves
against probed availability with Viewer defaults (Windows D3D12, Linux OpenGL,
macOS Metal), and the owned composition constructs the selected application
and managers. Render conventions follow the active backend; borrowed argument
strings and the filtered pointer array outlive the application. Legacy trimmed
builds and the opt-in Vulkan application build keep their compile-time paths.
Linux OpenGL selection retains the SDL host (see the deviation in Known
limitations); Vulkan selection uses the XCB host through a partial
`PlatformVulkanXcb` library that avoids clashing with the SDL OpenGL
application and asset implementation. Default startup through the runtime
path is verified natively on all platforms. See the
[exact-revision verification](../verification/da41b5ce53f612784f0f6c15f67a79bede96e776.md).

M2.5 reuses the startup contract and owned composition for Billiard with its
application-specific defaults (Windows OpenGL, Linux OpenGL, macOS Metal);
CI selects D3D12 explicitly through `--rhi d3d12`. Capture and quit dispatch
follow the active backend rather than build macros: Windows OpenGL keeps the
`glReadPixels` BMP capture, D3D12/Metal capture paths retire the application
themselves, Vulkan checks pending screenshots, and Linux completion stays
driven by the external `capture-complete` marker. Linux Billiard moves from
the XCB OpenGL host to the composition's retained SDL host (full ImGui
platform integration; the external X11 window capture is host-agnostic).
Legacy trimmed builds and the opt-in Vulkan application build keep their
compile-time paths.

Runtime-selection builds title their windows
`<App> - <Platform>/<Backend> (<Host>)` from the composition's platform and
host labels, so the active combination is visible in the title bar. Linux
capture scripts resolve the window id by name through `xwininfo -root -tree`
instead of exact-title import. Legacy builds keep their existing titles.

M2.6 adds a full-scope Viewer startup matrix: every CI-feasible
platform/backend combination launches from the same executable and its
capture is compared against the platform baseline by
[compare_captures.py](../../scripts/compare_captures.py) (pure-Python PNG/PPM
decoding, blank-frame and dimension guards, calibrated differing-pixel and
high-delta bounds). Linux covers OpenGL and Vulkan (lavapipe); macOS covers
Metal and Vulkan (MoltenVK); Windows covers D3D12, OpenGL and Vulkan through
Mesa llvmpipe/lavapipe deployed from mesa-dist-win. All Windows backends use
GEFS_SCREENSHOT_PATH self-capture: the Win32 OpenGL host gained a glReadPixels
front-buffer readback that writes an 8-bit RGB PNG through zlib and retires
the app (the elevated CI executor has no enumerable interactive desktop, so
GDI window capture is not viable), matching the D3D12/Vulkan paths. The
Windows Vulkan run uses the system loader (the SDK installer refreshes
System32) and discovers lavapipe through an ICD registered in
HKLM\SOFTWARE\Khronos\Vulkan\Drivers: the executor runs elevated, and the
Vulkan loader ignores VK_ICD_FILENAMES/VK_DRIVER_FILES under elevated
permissions, reading only the registry. An opt-in GEFS_VULKAN_LIBRARY
override can point the loader at an alternate runtime when needed.
Cross-backend comparisons use a wider differing-fraction bound (0.40) with
the high-delta and mean-delta guards unchanged, calibrated against measured
software-rasterizer variation. M2.6 also adds the Billiard per-backend startup
matrix (physics regression + backend-correct capture per combination) and the
`--list-rhi`/error-path native smoke on all platforms, migrates the Vulkan
regression coverage into `build_linux` via the integrated `VULKAN_RHI_TESTS`
entry point (a verified superset of the specialized job's tests), and retires
the specialized `verify_vulkan_shader_abi` job; macOS keeps its in-job MoltenVK
verification. See the stage-1
[verification](../verification/169f9c3d8f5e5b4f7fa469efffd9468a36fb4846.md) and the
[complete-M2.6 verification](../verification/6cf2c7d08853a81f4380611035e582525f828bcc.md).

The canonical startup names are `d3d12`, `opengl`, `metal`, and `vulkan`.
Accept `--rhi <name>` and `--rhi=<name>`; `--list-rhi` takes no value.
Reject unknown/empty names and repeated `--rhi` options rather than letting
argument order choose a backend. Respect `--` as the end of RHI option parsing
and preserve the remaining application arguments, including their order and
executable name. Client integration must keep the borrowed argument strings and
filtered pointer array alive for the application's lifetime.

Availability is supplied by composition/probes, not inferred from a linked
library or SDK. Distinguish not compiled, runtime prerequisites unavailable, and
ready to attempt initialization. The last state is not successful device/window
initialization. Resolve the requested backend, or the caller's application
default when omitted; a missing/unavailable choice returns a diagnostic and
never selects another ready backend.

### CI consolidation strategy

The accepted strategy is to consolidate Vulkan coverage during M2, without
waiting for reconstruction or hot switching. Keep the three platform jobs
initially; splitting every platform into a serial build/test pair is not itself
a latency optimization.

- Prepare Linux environment parity early in M2. The standard job currently uses
  Ubuntu 22.04 while the specialized Vulkan job uses Ubuntu 24.04 with lavapipe,
  validation layers, SPIR-V tools, and native XCB/Xvfb coverage. Preserve the
  required header, driver, and WSI capabilities when aligning environments.
- Use the existing `VULKAN_RHI_TESTS` integrated test entry point. Preserve ABI,
  compute, resources, pipelines, graphics, GUI, swapchain lifecycle, native
  presentation, validation checks, and application captures, not just ABI tests.
  macOS already exercises MoltenVK inside its platform job.
- Once startup factories are integrated, run each backend from the same
  application build instead of configuring a second Vulkan application tree.
  Compare the migrated test set and inspect exact-revision artifacts before
  removing `verify_vulkan_shader_abi`. Its existing coverage is Linux-specific;
  migration does not establish missing Windows native backend evidence.
- Revisit selective test fan-out as M3/M4 expand reconstruction and transition
  cases. Split only when measured test/retry benefits outweigh repeated setup
  and workspace transfer. Bind artifacts to revision, platform, configuration,
  and toolchain; retain runtime libraries, assets, shader tools, source-based
  fixtures, and generated CTest path requirements.
- Linux coverage may be isolated separately, but its instrumented build cannot
  reuse ordinary binaries. Its report-generation script continues after CTest
  failure, so a coverage report is not a substitute for the required test gate.
- Preserve normal-push full acceptance and API-triggered quick iteration.
  Every required platform build/test job remains part of acceptance after any
  topology change; quick/fast success is never acceptance.

### M1 implementation boundary

`BUILD_ALL_DESKTOP_RHIS` defaults to `ON` for Windows, Linux, and macOS.
Configuration requires Vulkan rather than silently omitting it. Viewer and
Billiard link the complete platform library set and depend on shared material
baking and CPU asset staging. Windows stages/installs HLSL, OpenGL, and Vulkan
shader variants; Linux stages/installs OpenGL and Vulkan variants. macOS builds
the existing Metal library even when the initial backend is Vulkan, and embeds
Vulkan shaders and prepared assets into both bundles.

Existing startup-selection options retain their meaning until M2.
`BUILD_ALL_DESKTOP_RHIS=OFF` preserves the legacy selected-backend build path;
it is not a complete desktop build. No runtime selector or switching menu has
been added. Normal static-library linking does not force unused backend objects
into the executable: M2 factories must create references to their implementations
before executable-level backend availability can be claimed.

`TEST_DesktopApplications` is a `fast` build/packaging regression using small
substitute libraries and a baker, exercising the production CMake helpers.
It links both fixture clients against every platform backend, checks staged,
installed, and macOS-bundled assets, and repeats the build after an asset edit.
This tests the build graph/copy paths, not real shader compilation, compression,
native device initialization, or rendering.

M1's original "one distributable contains all backend code/resources" exit
criterion is refined to the build/package foundation: unused static archive
members cannot establish executable availability before M2 references them.
Native project installer generation also remains a distinct M5 rollout check.
This is an explicit sequencing refinement, not a claim that either check has
already passed. The full M1 source revision passed pipeline 1072 on all enabled
platforms; default native and opt-in Vulkan application artifacts were inspected.

### M3 execution plan

This breakdown governs the confirmed M3 checkpoint. Work steps follow the M2
rule: self-check plus CI verification, no human stop between steps; only the
completed M3 checkpoint requires confirmation before M4. Cross-backend
transitions remain M4 and UI rollout remains M5; this plan claims neither.

Constraints carried from the target ownership section:

- The composition root retains long-lived CPU services (asset loading, scene,
  physics, animation, game logic, input), the platform host, the ImGui
  context/font data, and the engine memory system. The replaceable render
  session owns the backend RHI lifecycle, graphics/pipeline managers, passes,
  GPU resources, and the ImGui renderer backend.
- No application-type dependency in the live path: reconstruction must not
  replace the application object or invalidate every module's borrowed app
  pointer. Concrete-application casts (D3D12 graphics operations) and
  window-handle casts (Metal device/view) are replaced by injected RHI
  references and explicit drawable hooks.
- Frame participant order is preserved (`TickFrame` unchanged); the frame
  scheduler is not redesigned.
- A render-session generation independent of scene revision invalidates
  borrowed GPU references and UI caches. Reconstruction reuses the same Scene
  without publishing a new CPU revision; UI interaction state persists in the
  GUI context while GPU-owned pass instances are reconstructed.

| Work step | Work | Verification boundary |
| --- | --- | --- |
| M3.1: session contract and module lifecycle | Define the persistent-root/render-session ownership contract; split module registration into persistent and session-scoped groups so the session group supports repeated initialize/finalize without touching persistent modules; move engine memory-system ownership to the persistent root; introduce the session generation counter. | Headless `fast` tests with stub modules: repeated session initialize/finalize preserves persistent module state, failure rollback, generation monotonicity. No native claim. |
| M3.2: RHI injection | Inject the backend RHI into graphics/pipeline managers: D3D12 removes the concrete-application `dynamic_cast` sites (Vulkan already injects); Metal obtains device/view through session drawable hooks instead of window-handle casts; narrow the graphics manager's application borrow to window/configuration services. | Headless composition/runtime suites; the native startup matrix stays unchanged. |
| M3.3: platform host and backend hooks | Extract persistent window ownership and event routing from the concrete applications (Windows, Cocoa, Linux SDL/XCB) behind a common host interface; explicit backend hooks for drawable/view creation, resize, wait-idle, GUI platform bind/unbind, and teardown including partial initialization state; audit static platform/ImGui callbacks so none survives session replacement; apply per-backend window-recreation policy where native constraints require it (WGL pixel format, GLX visual), preserving logical geometry, DPI, and panel placement. | Per-platform native startup matrix stays green; hook and partial-cleanup tests where headless-feasible. Linux keeps the SDL (OpenGL) and XCB (Vulkan) hosts; XCB ImGui parity remains the tracked follow-on, not reopened here. |
| M3.4: owned session composition and same-backend reconstruction | Factories produce replaceable render sessions on the persistent host instead of concrete applications; same-backend destroy/recreate API with a generation bump; Viewer and Billiard consume the new composition; startup behavior, defaults, and window titles unchanged. | Headless reconstruction tests assert CPU state preservation (scene revision, camera, physics, ImGui layout); native startup matrix. |
| M3.5: reconstruction hardening and failure recovery | Injected partial-initialization failures (graphics, GUI renderer, drawable) roll back the session without touching persistent services; repeated cleanup is idempotent; generation-based invalidation of GPU caches, UI pass state, queued events, and held input for same-backend reconstruction (full switch behavior remains M4). | `fast` regression covers repeated cleanup and failure recovery without whole-runtime restart; native failure-path smoke where CI can drive it. |
| M3.6: native acceptance and CI consolidation | Same-backend destroy/recreate matrix on every CI-feasible platform×backend combination with before/after capture comparison and repeated reconstruction cycles; failure-recovery smoke; exact-revision verification records; resolve the M3-owned known-limitation rows. | Full exact-revision three-platform CI; M3 checkpoint self-check complete, then human confirmation before M4. |

Decision points to settle during implementation and record here:

- Whether same-backend reconstruction keeps the native window (D3D12, Metal,
  Vulkan on an existing drawable) or recreates it (WGL/GLX constraints);
  logical geometry, DPI-aware layout, and floating-panel placement are
  preserved either way, never native handles.
- The input manager stays persistent; held-input reconciliation and stale
  event discard on window ownership change are M4 scope — M3 only provides
  the hooks.
- `WaitForGraphicsIdle` before rollback remains mandatory for every session
  teardown path, including partial initialization.

Exit-criteria mapping: same-backend destruction/recreation preserves CPU
state → M3.4/M3.6; repeated cleanup and failure recovery without
whole-runtime restart → M3.5/M3.6.

M3.1's [RenderSession](../../Framework/Manager/RenderSession.hpp) contract adds
the replaceable session slot: borrowed session modules register per cycle in
frame order, `Activate` initializes forward with reverse rollback (owner GPU
drain via the rollback hook), and `Retire` finalizes in reverse and clears
registrations. The session generation is independent of scene revision and
changes on every successful activation and on retirement of an active session,
so borrowed GPU references and UI caches invalidate immediately — including
during the gap before recreation; failed activation and repeated retirement
do not publish identities. [RuntimeRoot](../../Framework/Manager/RuntimeRoot.hpp)
is the persistent composition root: long-lived CPU services in their own
module group, one session slot, and the engine allocator stack owned at the
root so session replacement never releases allocators persistent services
use. Frame participant order stays caller-composed (`TickFrame` unchanged).
The `fast` [session regression](../../Test/Runtime/RenderSessionTest.cpp)
covers repeated cycles with persistent services untouched, failure rollback
and retry, generation semantics, interleaved frame order across replacement,
and allocator-stack ownership across session cycles and root destruction.
Applications keep their existing startup paths; native wiring is M3.4. See
the [exact-revision verification](../verification/252ac2abb32ab8db651adf4cdcce57a308043cbb.md).

M3.2 injects the backend into the graphics managers, removing concrete
application types from the render path. `D3d12GraphicsManager` now takes
`D3d12RHI&` plus an optional lazy `MemorySystem` provider at construction (the
allocator stack is created after composition construction); every
`dynamic_cast<D3d12Application*>` site is gone, and the only remaining
application borrow is the `IApplication` window/configuration surface.
`Metal2GraphicsManager` takes a lazy `MetalView` drawable provider instead of
casting the application's window handle; `CocoaMetalApplication::GetMetalView`
exposes the installed drawable with a type check and returns nullptr before
graphics resources exist, so a missing drawable fails explicitly. Vulkan
already injected its RHI; OpenGL has no application casts. The runtime
factories and the legacy compile-time Viewer/Billiard paths wire the
providers, and the D3D12 GUI-texture regression constructs the manager with
the injected RHI. Startup behavior is unchanged; the native matrix verifies
the injection. See the
[M3 exact-revision verification](../verification/2fb4c0e73ff67a60fab0569557b804c42f64e2e6.md).

M3.3 formalizes the persistent host/backend boundary without changing startup
behavior. [IPlatformHost](../../Framework/Interface/IPlatformHost.hpp) declares
GUI platform bind/unbind and `ProcessEvents` (platform event drain separated
from module frame updates); the four desktop hosts (`WindowsApplication`,
`XcbApplication`, `SdlApplication`, `CocoaApplication`) implement it, and
their `Tick` is now `ProcessEvents` plus the module frame.
[IRenderBackend](../../Framework/Interface/IRenderBackend.hpp) declares the
backend-native lifecycle (graphics initialize / wait-idle / finalize, safe
after partial initialization); every desktop backend application implements
it. `MetalView` no longer borrows the application: it receives the persistent
input manager and MSAA sample count directly, removing the last
`dynamic_cast<BaseApplication*>` in the view layer. Hardening: the SDL OpenGL
backend verifies window/context at graphics initialization, binds/unbinds the
ImGui SDL platform through the GUI hooks (shutdown previously ran after
`BaseApplication::Finalize`, unconditionally), deletes the GL context at
graphics finalize (previously leaked), and aborts window creation on failure
instead of continuing with a null window; the WGL and GLX backends gain
explicit `WaitForGraphicsIdle` (`glFinish` on the current context). Statics
audit: the ImGui Win32/OSX/SDL platform backends keep their state per ImGui
context and the Win32 `WndProc` routes through per-window `GWLP_USERDATA`, so
no global state survives session replacement. The SDL host keeps the GL
context coupled to the window lifetime; XCB ImGui parity remains the tracked
follow-on. See the
[M3 exact-revision verification](../verification/2fb4c0e73ff67a60fab0569557b804c42f64e2e6.md).

M3.4 makes the session boundary native. The composition layer splits each
backend factory into a persistent host and a session-module factory
(`CreateRhiSessionModules`); `ReconstructRhiSession` drives a same-backend
cycle at a complete frame boundary — retire the session modules and native
backend state, recreate the window where the backend requires it (OpenGL's
WGL/GLX/SDL window-lifetime constraints; other backends rebuild on the
existing drawable), recreate the backend and GUI platform binding, then
compose, register, and activate fresh pipeline/graphics modules. The host
application object persists across reconstruction, so every module's
borrowed application pointer remains valid; persistent CPU services and the
root's allocator stack are never touched. Viewer and Billiard orchestrate
through `RuntimeRoot` (persistent group + session slot + caller-composed
frame plan preserving the historical interleaved order); the legacy
compile-time paths keep the application lifecycle. `--rhi-reconstruct N`
runs N same-backend cycles after the first presented frame for verification
runs. `IPlatformHost::ShutdownPlatform` carries one-time subsystem teardown
(`SDL_Quit`) that the root path would otherwise bypass. The composition
regression covers window/GPU-free session recomposition and the window
policy flag; native reconstruction evidence is M3.6. See the
[M3 exact-revision verification](../verification/2fb4c0e73ff67a60fab0569557b804c42f64e2e6.md).

M3.5 hardens reconstruction and covers failure recovery. Audit: the D3D12
RHI's `DestroyAll` releases every native object and nulls its handles, and
re-initialization recreates them on the persistent host; the Vulkan RHI's
`destroyAll`/`initializeDevice` pair is guarded and null-safe for re-entry;
graphics-manager finalization drains the GPU and shuts down the ImGui
renderer before the backend's native teardown. Scene texture objects retain
only prepared CPU images, and GPU upload tracking (`m_nSceneRevision`) plus
material/geometry caches are per-manager state, so a fresh session re-uploads
the same Scene without publishing a new CPU revision; the session generation
is the explicit epoch for borrowed GPU references and UI caches (cross-backend
invalidation is M4). The input manager stays persistent; window recreation
under the OpenGL policy discards stale native event queues, while full
held-input reconciliation remains M4. The `fast` session regression now also
covers mid-cycle graphics failure (no session published, persistent services
untouched, generation semantics), GUI-platform failure rollback of the
recreated graphics state, retry/recovery without whole-runtime restart, and
idempotent finalization after a failed cycle. See the
[M3 exact-revision verification](../verification/2fb4c0e73ff67a60fab0569557b804c42f64e2e6.md).

M3.6 adds the native same-backend reconstruction matrix: on every CI-feasible
platform/backend combination, Viewer and Billiard run two destroy/recreate
cycles (`--rhi-reconstruct 2`) after the first presented frame. Viewer
captures must still match the startup baseline (the CPU scene is preserved
while every GPU resource is rebuilt); Billiard must pass the physics
regression and capture, proving simulation state survives reconstruction.
Each run asserts the cycle-2 log marker so the matrix cannot pass vacuously.
OpenGL recreates the native window (WGL/GLX/SDL window-lifetime constraints);
D3D12, Metal, and Vulkan rebuild on the existing drawable. Failure recovery
is covered by the `fast` headless session regression (production root
primitives with injected graphics/GUI-platform failures, rollback, retry,
and idempotent finalization) plus the unchanged native error-path smoke;
native partial-init injection is not claimed. See the
[M3 exact-revision verification](../verification/2fb4c0e73ff67a60fab0569557b804c42f64e2e6.md).

### M4 execution plan

M4 is human-confirmed (M3 checkpoint accepted). Work steps follow the same
rule: self-check plus CI verification, no human stop between steps; only the
completed M4 checkpoint requires confirmation before M5. The switch trigger
UI remains M5; M4 verifies through a diagnostic CLI switch sequence.

Constraints carried from the target switching sequence:

- The host application object persists across switches; every module's
  borrowed application pointer remains valid. Backend-native state moves out
  of the concrete application classes into session-owned backend objects
  implementing `IRenderBackend` against the persistent host; the concrete
  application classes remain only for the legacy compile-time paths.
- One request at a complete frame boundary (after main and secondary
  viewport submission); overlapping requests get an explicit result;
  selecting the active backend is a no-op; unavailable targets fail
  explicitly without fallback.
- Sequence: Running → Preflight → Quiescing → Retiring → Creating →
  Restoring → FirstFrame → Running. Preflight failure keeps the old session.
  Failure after retirement attempts one previous-backend reconstruction;
  the failed switch is reported even when recovery succeeds; recovery
  failure exits orderly with both errors. The retired session is never
  reused.
- Success publishes only after the restored scene completes a
  rendered/presented frame; minimized/zero-size windows neither produce
  false success nor busy-wait. Simulation pauses during the switch without
  accumulating catch-up; platform event processing continues; stale window
  events are discarded and held input reconciled on ownership changes.
- Render conventions follow the active session (OpenGL projection correction
  updates camera/projection before the first restored frame). The session
  generation invalidates borrowed GPU references and UI caches; the same
  Scene is reconstructed without a new CPU revision; UI interaction state
  persists in the GUI context.

| Work step | Work | Verification boundary |
| --- | --- | --- |
| M4.1: backend lifecycle detachment | Move backend-native state (D3D12/Vulkan RHI, WGL/GLX/SDL contexts, Metal view creation, backend-specific GUI platform binding) out of the concrete applications into session-owned backend objects implementing `IRenderBackend` against the persistent host; runtime factories produce host + backend + session modules; root orchestrates the backend object directly. | Headless contract tests; native startup matrix and same-backend reconstruction matrix stay green (no behavior change). |
| M4.2: switch state machine and request contract | Frame-boundary switch processing with the full state sequence, explicit request results, same-backend no-op, preflight (compiled inclusion + runtime availability), generation bump, render-convention update; diagnostic `--rhi-switch` sequence entry for verification runs. | Headless state-machine tests (ordering, overlap rejection, no-op, preflight failure keeps old session); native A/B switch smoke. |
| M4.3: window/view rebinding, pause, and input reconciliation | Cross-backend window policy (recreate when either side's native constraints require it; preserve logical geometry, DPI-aware layout, floating-panel placement), Metal/Vulkan Cocoa view integration, simulation pause without catch-up, stale-event discard, held-input reconciliation, FirstFrame gate including minimized/zero-size windows. | Headless pause/gate tests where feasible; native A/B/A with state preservation. |
| M4.4: failure recovery | Preflight failure keeps the old session; post-retirement failure triggers one previous-backend reconstruction with the failed switch reported; recovery failure exits orderly with both errors; retired sessions never reused. | Headless injected-failure coverage on the production state machine; native failure smoke where CI can drive it. |
| M4.5: native A/B/A acceptance and CI consolidation | Directed switch matrix over the CI-feasible pairs (Windows d3d12/opengl/vulkan, Linux opengl/vulkan, macOS metal/vulkan): A/B/A restores the scene (capture comparison) and physics state; failure-path evidence; exact-revision verification records; resolve M4-owned limitation rows. | Full exact-revision three-platform CI; M4 checkpoint self-check complete, then human confirmation before M5. |

Decision points to settle during implementation and record here:

- Cross-backend window policy per pair: WGL/GLX/SDL sides force recreation;
  Metal↔Vulkan on Cocoa may replace the content view/layer without window
  recreation — choose per evidence, preserving geometry/DPI/panels either
  way.
- Backend-specific GUI platform binding (SDL `InitForOpenGL`, Win32
  `InitForOpenGL`, Darwin Vulkan surface callback) moves with the backend
  object; the host exposes the generic bind/unbind surface.
- The diagnostic switch sequence syntax and frame pacing are verification
  scaffolding, not the M5 UI contract.

M4.1 detaches backend lifecycle from the concrete applications:
session-owned backend objects (`D3d12Backend`, `Win32OpenGLBackend`,
`SDLOpenGLBackend`, `MetalBackend`, and a platform-neutral `VulkanBackend`
with injected surface factory, GUI-bind hook, framebuffer query, and
drawable preparation) implement `IRenderBackend` against the persistent
host. The concrete applications remain for the legacy compile-time paths
as host + owned backend. Runtime factories compose plain hosts with
separate backend objects; the dual-mode `LinuxHost` (SDL window for OpenGL,
XCB window for Vulkan) keeps the application object persistent across
Linux cross-backend switches. `IPlatformHost` gains `SetActiveBackend`
(native resize routing) and `OnBackendSwitch` (windowing-mode selection);
`IRenderBackend` gains window creation, GUI bind/unbind,
`OnSessionActivated` (renderer-dependent platform setup such as WGL
multi-viewport callbacks), resize notification, and the window-recreation
policy. Linux CI now installs SDL2 so the documented SDL host and the
dual-mode host are CI-verified; trimmed builds without SDL2 keep the plain
XCB pieces.

M4.2 adds the frame-boundary switch state machine
(`RhiSwitchController`): one request at a time after main and secondary
viewport submission; same-backend selection is a no-op; overlapping
requests are rejected explicitly; preflight (compiled inclusion + runtime
availability) keeps the old session on failure. A switch retires the
session and native state (window recreation when either side requires it),
creates the target backend on the persistent host, updates render
conventions before restoring, composes and activates fresh session
modules, and publishes success only after a rendered/presented frame at a
usable framebuffer size (FirstFrame gate; minimized/zero-size windows wait
without false success or busy-loop). Simulation pauses for the duration
through a render-only frame plan; no catch-up interval accumulates.
`CreateRhiBackendObject` dispatches per-platform backend objects;
`BaseApplication::SetRenderConvention` updates the host configuration.
Viewer and Billiard accept `--rhi-switch backend@frame,...` verification
schedules. The headless `RhiSwitchTest` covers the request contract, A/B/A
cycles with generation/convention updates, the zero-size gate, preflight
retention, post-retirement recovery, recovery failure, and session-failure
recovery.

M4.3 preserves logical window geometry across reconstruction and switching
on every host (Win32 outer rect, SDL position/size, XCB geometry, Cocoa
frame); native handles are never preserved. Retiring windows flush queued
platform messages/events (Win32 message drain, `SDL_FlushEvents`, including
mode switches away from SDL) so stale input never reaches the recreated
window; held input is reconciled by re-initializing the persistent input
manager when session modules change. ImGui docking/floating layout persists
in the GUI context; the FirstFrame gate handles minimized windows.

M4.4 failure recovery is implemented in the controller: preflight failure
keeps the old session; failure after retirement attempts one
previous-backend reconstruction and reports the failed switch even when
recovery succeeds; recovery failure requests an orderly exit with both
errors; the retired session is never reused. Headless injected-failure
tests cover all paths on the production controller. Native evidence:
Windows preflight rejection with an unavailable Vulkan loader override;
Linux/macOS post-retirement recovery with no usable Vulkan ICD.

M4.5 adds the native cross-backend switch matrix: A/B/A directed switches
for Viewer and Billiard on every CI-feasible pair (Windows
d3d12/opengl/vulkan, Linux SDL-OpenGL↔XCB-Vulkan both directions, macOS
Metal↔Vulkan both directions). Final captures must match the startup
baseline (scene restored on the original backend), Billiard physics
regression must pass, and the switch-complete marker is asserted so the
matrix cannot pass vacuously.

M4 acceptance is verified at
[4ea0cf6a](../verification/4ea0cf6a0132100a77c15396e40c07e069d1790d.md):
full three-platform CI green (Windows three-job split, Linux, macOS), all
CI-feasible A/B/A pairs restore the scene and physics, failure/recovery
smokes pass, and the explicit-failure contract is additionally confirmed on
a real AMD Windows driver. The M4 checkpoint self-check is
complete; it awaits human confirmation before M5.

### M5 execution plan

M5 is gated on M4 human confirmation. Work steps follow the same rule:
self-check plus CI verification, no human stop between steps; only the
completed M5 checkpoint requires confirmation, which culminates in deliverable
acceptance (the pull request).

Maintainer decisions (2026-11-23) shaping M5:

- **UI keeps its current form and style.** The requirement is *no regression*
  of the existing Viewer UI and docking, not a redesign. Backend selection is a
  minimal addition in the existing menu pattern (the `调试`/`帮助` main menu bar
  in `GuiSubPass.cpp`), not a new panel paradigm.
- **Switch requests execute immediately.** A modal progress indicator blocks
  user input for the duration of the switch (safer than concurrent interaction),
  showing state-machine progress and the success/failure/recovery result.
- **Billiard needs no selection panel.** The ImGui infrastructure is retained
  (for a future in-game debug interface) but hidden by default.
- **Installed-distribution verification is best-effort.** The project is
  incomplete; verify as much as the CI environment allows rather than treating
  installed-app validation as a hard gate.
- **Retina/mixed-DPI acceptance carries over existing evidence.** No dedicated
  real machine is available currently; reference the Vulkan-plan evidence rather
  than building a new rig.

Constraints carried from M4 and the Viewer UI verification guidance:

- The selection UI drives the same `RhiSwitchController` (frame boundary, one
  request at a time, explicit results, no fallback); it replaces the diagnostic
  `--rhi-switch` CLI as the user-facing entry, while the CLI remains for
  verification runs. Unavailable targets are disabled with a reason from the
  availability probe; the active backend is shown (window title already carries
  `<Platform>/<Backend> (<Host>)`).
- Existing docking/floating layout and GUI interaction state persist across
  switches: the dock tree is not rebuilt, saved layouts are preserved
  (`FindWindowSettingsByID`), defaults initialize only when no saved layout
  exists, and floating panels reposition by desktop coordinates (work position
  and size, not fixed coordinates or framebuffer pixels) after any window
  recreation.
- The modal progress indicator reflects the FirstFrame gate; it neither
  busy-waits nor false-succeeds on minimized/zero-size windows, and it releases
  input only after the restored scene presents a frame (or the switch fails and
  recovery/exit completes).

| Work step | Work | Verification boundary |
| --- | --- | --- |
| M5.1: minimal selection and status UI | Add backend selection to the existing Viewer menu pattern: list probe-available backends, mark the active one, disable unavailable targets with a reason; show current backend/host/platform status. Update the window title to reflect the active backend after each switch (it is currently static at startup). No redesign of existing panels; existing UI/docking must not regress. | Headless layout regression (`GuiSubPassLayoutTest`) stays green; native selection→switch smoke; title reflects the active backend post-switch; existing panels/docking verified unchanged on real screenshots. |
| M5.2: immediate switch, modal progress, state preservation | Wire selection to `RhiSwitchController` for immediate execution; a modal progress indicator blocks input during the switch and reports state progress, success, failure, and recovery; preserve docking/floating layout and GUI interaction state across window recreation; failed switches show an explicit error (no fallback). | Native A/B/A with layout/interaction preservation; failure-path UI evidence; modal gate respects minimized/zero-size windows. |
| M5.3: Billiard composition reuse | Refactor Billiard to the same runtime composition and switch controller, removing duplicated orchestration; retain ImGui infrastructure but hidden by default (no selection panel); preserve Billiard physics-regression behavior. Precursor to the post-M5 Core entry extraction. | Billiard physics regression and switch matrix stay green; no visible panel by default. |
| M5.4: backend-specific visual and layout acceptance | Per-backend panel visibility/non-overlap/viewport containment after switches; multi-viewport placement and floating input; Retina/mixed-DPI carried over from the Vulkan plan (no new rig). Headless layout regression plus real screenshot inspection per platform/backend. | `GuiSubPassLayoutTest` covers docking/non-overlap/containment/resize/Retina/saved-layout; real Viewer screenshots inspected per backend; Vulkan-plan DPI evidence referenced. |
| M5.5: acceptance matrix and best-effort installed verification | Integrate the complete directed-transition acceptance matrix with native evidence; generate native installers and verify installed applications launch/switch as far as the CI environment allows (best-effort, not a hard gate); exact-revision three-platform CI; verification records; resolve M5-owned limitation rows. | Full exact-revision three-platform CI; installed-app checks recorded with their limits; M5 checkpoint self-check complete, then human confirmation → deliverable acceptance (PR). |

Decision points settled (maintainer, 2026-11-23):

- Selection UI form: minimal addition in the existing menu pattern; no
  regression of current UI is the binding requirement.
- Switch UX: immediate execution with a modal progress indicator blocking input
  during the switch.
- Billiard scope: composition reuse only; ImGui retained but hidden by default.
- Installed verification: best-effort given the incomplete project and unknown
  CI installed-app capability; not a hard acceptance gate.
- Retina/DPI: carry over existing Vulkan-plan evidence; no new real-machine rig.

M5.3 (Billiard composition reuse) is the natural precursor to the post-M5 Core
entry extraction; the System/Core architecture adjustments themselves remain
after M5 acceptance (see Post-M5 architecture adjustments).

## Verification and acceptance

- Add focused production-path headless tests for selection, ordering, rejected
  requests, partial initialization, reconstruction/recovery failure, state
  retention, generations, and repeated lifecycles. Mark window/GPU-free tests
  `fast`. Demonstrate regressions fail without their implementation.
- Native tests must launch each backend from the same complete application and
  exercise every directed pair: six on Windows, two each on Linux and macOS.
  Include A/B/A loops, same-backend reconstruction, empty/populated scenes,
  resize/minimize, MSAA, shadows, skybox, textures, and GUI debug views.
- Check no stale GPU/UI handles and no cumulative engine-owned live-resource
  growth across repeated cycles. Distinguish driver caching from owned leaks.
- Inspect captures before and after switching and after recovery. Verify scene
  readiness, shadow/coordinate correctness, visible unclipped panels, and
  preserved layout; a successful screenshot command is not visual acceptance.
  Native keyboard/focus/floating-window behavior needs interactive evidence.
- Use the current `.circleci/config.yml` commands and exact-revision full
  Windows/Linux/macOS acceptance. Extend its backend runs as checkpoints need
  them. If a runner cannot exercise a matrix backend, record the gap and obtain
  revision-matched native evidence; compilation or another backend's passing
  job cannot substitute for that transition.
- Before PR creation, satisfy the repository acceptance gate and write the
  revision-specific verification record. Documentation-only changes follow the
  existing exemption; this draft does not verify new source behavior.

## Known limitations and follow-ups

| Limitation or open decision | Owning checkpoint |
| --- | --- |
| Linux runtime selection retains the SDL OpenGL host (no GUI regression): the XCB host has no ImGui platform input backend and disables multi-viewports. XCB ImGui parity is a prerequisite for the agreed XCB-only host; Vulkan selection uses the existing XCB path with its tracked GUI limits. | M2.4 deviation; XCB parity follow-on |
| Static archives can discard unused backends until factories reference them; complete executable inclusion is not claimed by library linking alone. | M2 |
| Viewer (M2.4) and Billiard (M2.5) application CLI wiring is landed and natively verified; the startup contract, Vulkan loader isolation, backend factories, and availability probes are in place. Remaining coverage migration and switching milestones gate wider advertisement. | M2.6-M5 |
| Linux Vulkan environment parity is aligned (Ubuntu 24.04, lavapipe, validation layers); the Windows native D3D12/OpenGL/Vulkan startup matrix is established; Vulkan coverage is migrated into `build_linux` (`VULKAN_RHI_TESTS`) and the specialized job retired after verified equivalence. | M2.6 (resolved) |
| Native installers: the CI `Package` step builds the Windows NSIS installer (`GEFS-0.1.0-win64.exe`) on master/tag (not on feature branches); real macOS bundles are exercised. Installed-app launch/switch verification is best-effort (not a hard gate) given the incomplete project and CI installed-app capability; fixture installation checks do not replace installed native distribution validation. | M5 (best-effort recorded) |
| Vulkan loader isolation is implemented and headless-tested; native evidence that non-Vulkan startup ignores an absent Vulkan runtime lands with application CLI integration. | M2.4 |
| Material baking uses the native Utility executable; complete desktop cross-compilation needs an explicit host-tool path before it can be supported. | Follow-on build scope |
| Backend partial-initialization and static platform/ImGui callback audit is complete (M3.3): ImGui platform backends are per-context, Win32 routing is per-window, and SDL/WGL/GLX cleanup paths are hardened. Repeatable same-backend session replacement is implemented and natively verified (M3.4–M3.6, pipeline 1138). | M3 (resolved) |
| Same-backend reconstruction invalidates GPU caches and UI pass state through fresh session objects and the session generation (M3.5). Cross-backend invalidation and held-input/queued-event reconciliation are implemented (M4.2 generation bump + render-convention update; M4.3 stale-event discard + input re-initialization) and natively verified across the A/B/A switch matrix. | M4 (resolved) |
| Minimized windows and quit requests during switching: the FirstFrame gate waits without busy-looping on minimized/zero-size windows (one check per frame boundary), and failure/recovery exits orderly via ExitRequested (M4.3/M4.4). A user-cancel of an in-flight switch is not exposed (the M5 modal progress indicator blocks input during the switch instead). | M4 (resolved; cancel UX deferred to M5 modal) |
| Vulkan Retina/mixed-DPI and floating-input acceptance remains open in the Vulkan plan; this roadmap does not resolve or duplicate its evidence. | M5 depends on the Vulkan plan |
| Complete native switching coverage is established by the cross-backend switch matrix CI runs (Windows `windows_matrix` directed pairs + recovery, Linux/macOS matrices in the build jobs) plus real-machine evidence; the earlier single-backend-only gap is resolved (M5.2–M5.4). | M5 (resolved) |
| Vulkan availability preflight now enumerates the native presentation WSI gate (instance `VK_KHR_get_surface_capabilities2` + `VK_KHR/EXT_surface_maintenance1`; device `VK_KHR/EXT_swapchain_maintenance1` paired with the matching surface extension and the `swapchainMaintenance1` feature) through a temporary instance, and marks Vulkan unavailable with a named reason on drivers lacking it (e.g. the AMD Windows driver 26.11.2). The earlier loader-only probe reported "ready" on such drivers, and the failure modal's light-gray dim was misread as a washed-out Vulkan frame; the modal dim is now dark and the failure dialog states the display recovers when closed. Headless fixture regression: `VulkanWsiProbeTest`. | M5 follow-up (resolved) |
| Windows OpenGL on AMD hardware presents a gray window (startup AND recovery): the GL back buffer holds the full scene+GUI (app-internal capture, vision-confirmed) but the visible window stays gray and SwapBuffers intermittently blocks on vsync. Exhaustive real-machine diagnosis ruled out pixel format/double-buffer/context/DC/FB binding/single-window/style/multi-viewport; CS_OWNDC, frame-start and pre-swap wglMakeCurrent, window realize (SW_SHOWNORMAL+UpdateWindow+SetForegroundWindow) and DwmFlush were ineffective. Converges on the WGL drawable/DWM redirection-surface establishment (GLX vs WGL); needs RenderDoc/APITrace engine-level debugging. The Windows D3D12 path is verified correct, and CI llvmpipe software OpenGL renders/presents correctly (matrix green). | M5 known limitation; engine-level follow-up |
| WSLg real-desktop forward runtime switch (OpenGL→Vulkan) hangs in the present/present-fence path after window rebuild (lavapipe maintenance1); headless Xvfb and reverse switch are green. WSLg XWayland WSI environment characteristic, not an engine defect; real-hardware verification pending. The recovery shadow-map crash once seen on WSLg was the shared no-current-GL-context engine bug (fixed by SDL frame-start context re-bind), not WSLg-specific. | Known limitation; WSLg WSI |
| Android and WebAssembly are unmaintained (years), disabled, and not verified by CI. Their compile-time paths predate the runtime architecture; reviving them requires refactor/rewrite against the host/backend composition model, not re-enabling the old builds. | Known limitation; follow-on rewrite after M5 |
| Other desktop targets, macOS OpenGL, plugins, seamless switching, and device-loss recovery are deferred, not supported by implication. | Follow-on scope after M5 |
| Legacy desktop application classes (host + `IRenderBackend` sibling inheritance) are retained through M4/M5 as the compile-time path and comparison baseline. Candidate post-acceptance simplification (maintainer decision recorded 2026-11-23): retire them once runtime switching is fully verified with no regression — trimmed builds are then served by the runtime factory with a reduced backend set, and validation-layer capture coverage migrates to runtime builds. This removes the sibling-override and delegation-recursion maintenance surface. Formal retirement is a roadmap amendment after acceptance, not part of M4/M5 scope. | Candidate after M5 acceptance |

Keep open limitations here with their owning checkpoint and resolve/update them
in the same implementation change. Per-revision evidence belongs in verification
records, not a rolling result index.

## Post-M5 architecture adjustments (maintainer decisions, 2026-11-23)

Decided in discussion after M4; implementation is gated on M5 acceptance.
Each phase verifies through the full three-platform CI matrix before the next.

### Service layer (Framework/System/) and two-tier module contract

The frame pipeline is not a universal module contract. Split IRuntimeModule:

- `IRuntimeModule` keeps the common lifecycle only: `Initialize`/`Finalize`
  (and the borrowed application pointer). No `Tick`.
- New `IFrameModule : _inherits_ IRuntimeModule` adds `Tick() = 0` — the
  contract for genuinely frame-driven modules: rendering, animation solving,
  physics simulation, game logic (`IGameLogic` rebases onto it), debug draw,
  and audio when implemented.
- The frame plan takes `IFrameModule` only. Services leave the plan; their
  current Ticks are all no-ops, so removal is behavior-preserving (the CI
  matrix is the regression net).

New `Framework/System/` parallel to `Manager/`, holding application-lifetime,
demand/event-driven services:

- `System/Memory/`: MemorySystem, MemoryManager, and the allocator stack
  (Frame/Ring/Slab/Buddy/Linear/SizeClass/Routing/ThreadCache/Tracking
  allocators, PmrAdapter, AllocatorUtils). Allocators belong with the memory
  service, not Algorism (which holds scene/geometry algorithms).
- `System/Input/`: InputManager (event-driven).
- `System/IO/`: AssetLoader (synchronous IO service; parsers stay in
  Framework/Parser as codecs).

Stay in `Manager/` by semantic role, not empty-Tick mechanics:

- SceneManager: the scene graph is the game world's structural description;
  scene change carries the passage of time and is Tick-bound. It may gain a
  real Tick (scene animation/interpolation) later.
- PhysicsManager: driven by in-game time; the empty base-class Tick is
  acceptable (the Bullet subclass ticks).
- PipelineStateManager: session-level service (rebuilt on switch), base
  lifecycle contract only.

The classification criterion is the semantic role (time-driven simulation /
rendering vs demand-driven service) and lifetime (app vs session), not whether
a Tick body happens to be empty today.

### Engine entry (Runtime/ renamed Core/) and app-as-consumer

The application shell (bootstrap, main loop, switch orchestration, capture
contract, System ownership) moves out of the apps into the engine. The
existing `Runtime/` composition layer is extended and renamed `Core/`;
Framework stays the subsystem library, Platform stays the hosts.

Target app content: an `IGameLogic` implementation, an AppDescriptor (title,
default scene, configuration, asset search paths, provided frame modules),
and the app's own assets. Engine-common assets are already staged separately
(M1).

Phased for stability (maintainer: 分步实现，稳妥为上):

1. Extract the shared orchestration duplicated in Viewer/Billiard main into a
   Core engine runner parameterized by the AppDescriptor. Apps keep a ~10-line
   main forwarding to `EngineMain(argc, argv, descriptor)`. Executables and
   behavior unchanged; the existing matrices verify equivalence.
2. Optional later polish: the engine entry library provides `main()` itself
   (app exports a logic factory symbol); Core absorbs platform entry quirks
   (WinMain/SDL_main/emscripten).

The frame-plan ownership question dissolves under the two-tier contract: Core
assembles the canonical plan from the frame modules the app provides; the app
does not specify the plan. Implementation must verify that the Viewer and
Billiard frame-module subsequences agree (their historical differences were
service modules with no-op Ticks).

### Sequencing relative to other follow-ups

These adjustments and the legacy desktop application-class retirement
candidate (Known limitations) are separate post-acceptance efforts; none is
part of M4/M5 scope. The retirement candidate removes the sibling-inheritance
maintenance surface that the System/Core split does not itself address.

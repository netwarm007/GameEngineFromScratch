# Engine modularization roadmap

## Purpose and scope

Incrementally separate runtime orchestration, scene data, CPU preparation,
physics, rendering, platform resources, and UI so that dependencies and resource
ownership are explicit and modules can be tested independently.

This is a maintained engineering plan, not a rewrite proposal, CI dashboard,
or authorization to implement every future stage. Confirm the next checkpoint's
scope before implementation. Prefer one independently testable boundary per
checkpoint, preserving existing application behavior.

## Target boundaries

| Area | Responsibility and dependency rule |
| --- | --- |
| Application composition | Select implementations, supply borrowed services, and coordinate lifecycle; do not make Application a service locator. |
| Runtime lifecycle | Track initialized modules, reverse cleanup, and failure rollback independently of frame scheduling or a native window. |
| Scene and animation | Own scene data and revisions; consume explicit parser, reader, resource-preparation, and scene interfaces. |
| Resource preparation | Read and decode resources before publication; scene objects retain prepared data rather than performing IO. |
| Physics | Own backend bindings and publish simulation transforms; scene nodes do not own backend handles. |
| CPU rendering preparation | Produce camera and lighting data without creating a graphics device. |
| Rendering coordination | Consume supplied scene/pipeline services and coherent scene snapshots. |
| RHI and platform | Distinguish GPU resources from native window/connection resources and make their lifetime dependencies explicit. |
| UI | Eventually distinguish context, native input integration, and renderer integration; preserve saved layouts. |

These are responsibility boundaries, not a requirement to add a new interface
or library for every row. Reuse existing abstractions and extract only what a
concrete dependency or test requires.

## Completed checkpoints

The following work is present in this branch. Historical evidence links describe
their named source revisions, not every subsequent commit or merge.

### Foundations already in the upstream baseline

- **Runtime lifecycle and scene services:** ordered initialization, reverse
  cleanup, rollback of successfully initialized modules, retry, and explicit
  scene/parser/asset/animation dependencies. Scene reload failures preserve the
  published scene. See the [combined lifecycle/scene/physics record](../verification/3e92bb82da4a76d11f6670ea2ce88c02e386ccaa.md).
- **Physics ownership and injection:** backends own body bindings, rendering
  consumes scene transforms, and physics receives its scene dependency
  explicitly. See the [physics dependency record](../verification/eead082050c90039ec0d8dd0e82490495845964f.md).
- **Scene texture/resource boundary:** scene objects no longer read files or
  decode images. An injected resource loader prepares materials, skyboxes, and
  terrain before scene publication; preparation failure preserves the previous
  scene and revision. This boundary is completed, not an outstanding task.
  Streaming and cancellation were not implemented. See the
  [resource boundary record](../verification/919fb2e9ec781fba2da359304ab12c752b7d587d.md).
- **Frame scheduling:** frame traversal is separate from module lifecycle and
  can select its own participants/order. See the
  [scheduling record](../verification/807c66cf6062b1be029fc8d9ebf11291eb1e0d1b.md).
- **CPU camera and lighting preparation:** extracted from GraphicsManager into
  independently testable production components while preserving backend
  coordinate/projection conventions. See the
  [camera record](../verification/2ed4630042ecb4951f1c7a9b8abf05b11ece0700.md)
  and [lighting record](../verification/fc86b2f8d3b20043611289b8da336cec45453046.md).
- **Rendering service injection:** Viewer and Billiard supply scene and pipeline
  dependencies explicitly. Recording-backend tests cover snapshots, revisions,
  replacement, and graphics-only clients. See the
  [rendering services record](../verification/0ebcff1802e79c36bb3414a12532fbef8b211da9.md).

### Follow-on foundations integrated upstream

The following checkpoints were originally accepted on this topic branch and
have since been integrated into master through PR #61. The stage A/B work below
builds on those upstream foundations; it is not a second implementation of them.

- **Offscreen ownership:** GraphicsManager owns per-frame color and shared
  depth/MSAA targets; Frame borrows descriptors. Resize, scene cleanup, and
  shutdown release each owned resource once and invalidate borrowed references.
  See the [ownership record](../verification/128d0a7654526c96363147044f2155defd5c37c2.md).
- **Desktop shutdown:** GPU synchronization precedes reverse module cleanup,
  backend destruction, and native window teardown. Fixes include synchronization
  between GLX and XCB destruction and D3D12 waiting before pipeline release.
  See the [shutdown record](../verification/517577bc857feb2e681d8fca665f66c05c5e093e.md).
- **Initialization rollback synchronization:** BaseApplication supplies a
  pre-rollback GPU synchronization callback to RuntimeModuleGroup. Failure
  errors, standalone module-group behavior, and retry registrations remain
  intact. Tests model initialization GPU work at every failing module position.
  The combined source, rebased onto master containing the separately developed
  PR #59/#60 algorithm fixes, has its own
  [verification record](../verification/9830f21b52cc6ed67f24849f47a75788cd645bc4.md).

The earlier acceptance records remain unchanged after rebasing. Algorithm,
coverage, and unrelated rendering improvements brought in from master are
upstream dependencies, not additional deliverables of this refactoring series.

## Follow-on stages

### A. Separate native window creation from RHI initialization

**Completed and accepted on the topic branch.** See the
[initialization boundary verification](../verification/f3ab9862160edae8d180d6d6ff81c5f4f2209f05.md)
for exact-revision evidence and limitations. This is not a claim of integration
into master.

First trace creation and failure paths on Windows, Linux, and macOS. Define the
smallest explicit sequence of native window creation, backend initialization,
and module initialization, matching the existing reverse shutdown boundary.

Deliverables:

- Name the owner and lifetime of the window, connection/context, device,
  swapchain, and borrowed handles.
- Separate orchestration responsibilities without rewriting backend device
  setup, changing rendering defaults, or introducing a general service container.
- Make failure reporting and the point at which cleanup becomes required
  explicit; preserve existing entry-point behavior or document a necessary
  intentional change before implementing it.
- Test ordering and cleanup after each newly exposed initialization boundary.
  Preserve the standalone clients exercised by existing tests.

Exit criteria: initialization and shutdown responsibilities are reviewable
without following nested platform-specific Finalize calls; focused regressions
and native three-platform acceptance pass.

Do not combine this stage with ImGui ownership, a new frame scheduler, or a
swapchain/resize redesign.

The implementation introduces an application-owned backend initialization hook
before the module walk. Successful backend setup is retained across module
rollback/retry, and finalization resets that state. A backend-hook error is
reported and returned without starting modules; the caller still finalizes the
application. Desktop graphics clients must create the native window first.

| Resource | Current owner and creation boundary |
| --- | --- |
| Windows native window and platform DC | WindowsApplication creates them in CreateMainWindow and releases them after backend cleanup. |
| D3D12 device, queues, swapchain, and sync objects | D3d12Application owns its RHI; its backend hook performs the existing setup sequence before modules. RHI callbacks borrow the application/window. |
| Windows OpenGL context | OpenGLApplication creates the temporary WGL bootstrap and final context in its backend hook, retaining native window creation in WindowsApplication. |
| Linux native window and connections | XcbApplication owns them. OpenGL framebuffer configuration discovery remains before native window creation; context/GLX drawable setup moves to the backend hook. |
| Linux GLX context/drawable | OpenGLApplication releases them and synchronizes their destruction before XcbApplication releases the native window and connections. |
| macOS window and Metal content view | CocoaApplication owns the window; CocoaMetalApplication installs the Metal view/device during its backend hook. The window retains the content view. Metal RHI setup/cleanup remains in Metal2GraphicsManager. |
| Vulkan instance, surface, device, and swapchain | VulkanApplication retains its RHI and performs the existing setup in its backend hook. On macOS, the Metal-view hook runs first because surface creation borrows its layer. |

Recording-application tests cover hook ordering/errors and retry. Native D3D12
and GLX tests check that creating the window alone does not create the
device/context, while module initialization sees a live backend. Backend-native
setup assertions, exceptions, and partial-allocation failure behavior are not
redesigned; D2D/Vulkan runtime support is not established by migrating hooks.

### B. Harden native partial initialization and cleanup

**Completed and accepted on the topic branch for the selected failure stages.**
See the [combined native failure verification](../verification/b95495ed37f5d2864a0dc56a7721a0c063c1c22b.md)
and the earlier [GLX failure verification](../verification/a9c7cdba1cc4af16f800c76248a7364db38b470b.md).
The [master integration verification](../verification/41eb3105302140bc231ba174c3c695ae60548602.md)
covers the combined revision after merging the upstream lifecycle, Splash, and
shadow changes.
This establishes the scoped exit criteria below, not exhaustive driver failure
handling, device-loss recovery, or integration into master.

Depends on the boundaries established in stage A.

The first scoped change rejects GLX drawable, context, and context-binding
failures before module startup. It releases partial GLX resources while the
native window/display remain alive, restores the previous X error handler,
and preserves the existing legacy-context fallback. Native Xvfb tests replace
GLAD entry points temporarily to fail specific calls after real resource
creation, including a queued asynchronous X error. Repeat finalization must
not destroy a resource twice. Window creation also preserves the Display that
queried the GLX configuration so legacy context fallback uses matching handles.
This is not exhaustive driver allocation-failure coverage or generalized X
display/framebuffer-selection failure handling. D3D12 and Metal have their own
selected failure coverage below rather than identical driver-level guarantees.

The other completed stages are D3D12 initialization HRESULT propagation and
partial fence-event allocation, plus Metal initialization-buffer, sampler,
queue and command-buffer allocation. D3D12 setup errors must release partial
backend resources before returning without starting modules. Event handles are
invalidated after closing, and registered resource cleanup runs once. Native
tests also submit a gated command list to prove GPU completion precedes module
rollback. Metal's failing initializer releases its own partial allocations;
successful initialization owns those same resources until explicit cleanup.
Repeated cleanup, D3D12 reinitialization after explicit platform/backend
cleanup, and same-RHI Metal retry are covered without changing ImGui context
ownership (stage C). Metal module-graph restart and failure after native GPU
submission were not independently injected. These stages do not claim
device-loss recovery, exhaustive driver fault injection, or D2D/Vulkan runtime
guarantees.

- Inspect actual backend failure paths before selecting failure-injection seams.
- Cover partial native resource creation, a single cleanup attempt, and repeat
  cleanup where the backend contract supports it.
- Distinguish application rollback from a failing module's obligation to release
  its own partial resources. A pre-rollback callback is not a substitute for
  safe cleanup inside the failing initializer.
- Exercise failure after submitted GPU work where feasible; modeled pending work
  alone must not be reported as native GPU failure-injection verification.
- Keep error codes/diagnostics explicit and preserve usable state for retry only
  where retry is deliberately supported.

Exit criteria: the selected failure stages have deterministic tests, no
double-release or premature-release behavior, and successful native application
initialization/rendering/exit remains verified on the enabled CI platforms.

### C. Separate ImGui lifecycle responsibilities

**Completed and accepted for the selected ownership boundary.** See the
[stage C verification](../verification/1ed352fa974f2c73b0f51d6e5f9e449d3d69c72b.md)
for exact-revision three-platform results and actual artifact observations.

Depends on stable platform/backend lifetime contracts from stages A and B.

- Identify ownership of the ImGui context, platform input backend, renderer
  backend, font resources, and UI texture bindings.
- Extract only the coupling that prevents independent testing or correct cleanup.
- Preserve docking, user layouts, reset behavior, desktop-coordinate placement,
  and framebuffer scaling.

The first boundary uses the following ownership contracts:

| Resource | Owner and lifetime |
| --- | --- |
| Viewer ImGui context and CPU font atlas | `GuiContext` owns the context; Viewer configures flags, style and fonts. The atlas owns font data. The context outlives explicit application shutdown, including initialization failure cleanup. A second owner must not replace an active context. |
| Enabled native desktop input backends | WindowsApplication, CocoaApplication and XcbApplication initialize GUI platform state after graphics/view setup and before modules. Window creation alone does not bind ImGui. Cocoa binds the final content view, not the view replaced by Metal setup. |
| Renderer backend and GPU font resources | D3d12GraphicsManager, OpenGLGraphicsManager and Metal2GraphicsManager own their ImGui renderer backend. Renderer initialization checks its result and rejects an already bound context. Only an initialized renderer is shut down. Metal RHI native allocation initialization no longer requires an ImGui context. |
| UI texture bindings | GuiSubPass owns its scene icons and debug texture views; graphics managers release draw passes while the renderer, texture descriptor pool and native device are still live. The D3D12 descriptor pool is reset after renderer shutdown. |

Shutdown order is the application's graphics-idle hook, reverse module cleanup
(including UI textures and renderer shutdown), GUI platform shutdown, native
graphics cleanup, native window cleanup, then context destruction. The Metal
graphics manager also waits for its retained render/compute submissions before
releasing UI resources. The context must
remain current throughout the application lifecycle. A failing initializer
cleans up its own partial state; it is not passed to its shutdown hook.
Successful GUI platform state survives module rollback for retry, but is
released once on explicit application finalization.

ApplicationLifetimeTest covers ordering, GUI platform failure, renderer/module
rollback and retry. D3d12GuiTextureTest exercises native platform/renderer
binding, borrowed-backend rejection, descriptor reuse and repeated cleanup.
GuiSubPassLayoutTest additionally exercises context ownership while retaining
its docking, viewport-origin, framebuffer-scale, resize and saved-layout cases.
MetalInitializationTest exercises native allocation failure/retry without a
context; its native macOS execution is covered by the stage C verification.

This boundary does not migrate SDL/Android platform input ownership, redesign
UI assets, or establish whole-module-graph restart or device-loss recovery.
Existing unsupported runtime configurations remain outside acceptance.
The new source is not covered by the stage A/B verification records.

Exit criteria: UI initialization/cleanup ordering and failure paths are tested;
GuiSubPassLayoutTest passes; actual backend screenshots confirm visible,
contained panels. Headless checks do not replace rendering inspection.

### D. Reassess remaining coupling

**D1 accepted: retire GPU scene resources on scene replacement.**

Implementation, regression controls and exact-revision three-platform native
acceptance are complete. See the
[D1 verification record](../verification/4ee8fee385f277506ad7ed45a07949e9da22cfbc.md)
for tested backends, scene-switch screenshots and limitations.

Read-only assessment found a concrete lifetime boundary before the remaining
service-location and UI-asset coupling: Metal appends geometry buffers and draw
batches while caching materials by name, but has no scene-specific teardown.
The common teardown does not clear these borrowed batches or material entries.
D3D12 already implements explicit scene retirement; OpenGL clears geometry but
does not consistently register scene textures for release.

Problem: replacing A with B (including identical material names with different
content), returning to A, or selecting an empty scene must not retain old
geometry, material bindings or scene-owned GPU allocations.

Scope and invariants:

- Wait for relevant GPU submissions before retiring a scene. Clear borrowed
  draw batches, material entries and frame texture bindings.
- Release scene-owned material, environment, BRDF and shadow textures once.
  Keep framebuffer ownership separate, and do not adopt UI icons or UI-owned
  texture views into the scene texture collection.
- Add Metal geometry-buffer retirement, retaining its initialized device,
  queues, per-frame constants, renderer backend and shared pipeline objects.
- Preserve existing shader/coordinate semantics, single-scene appearance,
  docking and saved layouts. Do not introduce a general resource framework,
  streaming, cancellation, hot reload or device-loss recovery.

Implementation and verification sequence:

1. Extend RenderServicesTest with registered scene textures, real geometry-node
   references, repeated material names, A/B/A and empty-scene transitions.
   Demonstrate failure before changing the common teardown.
2. Fix the common scene retirement contract and the dependent Metal/OpenGL
   ownership paths; preserve D3D12's existing explicit cleanup.
3. Add native Metal buffer-retirement/reuse coverage and exercise actual scene
   replacement, not merely application reinitialization. Verify empty/repeated
   cleanup and keep GPU/backend resources usable afterward.
4. Run focused local tests, then exact-revision Windows/Linux/macOS CI.
   Inspect native scene-switch screenshots and existing Viewer/Billiard
   artifacts. Headless checks and single-scene captures do not by themselves
   establish native scene-switch correctness.
5. Record results and remaining limitations under the tested full revision;
   mark this checkpoint accepted only after its stated gates pass.

The opt-in Viewer argument `-sceneCycle 1` loads splash then billiard, while
`-sceneCycle 2` returns to splash in the same process. CI captures both final
states and requires the corresponding `SCENE_CYCLE_READY` log marker. Without
this argument the ordinary Viewer load path is unchanged. CI sets
`GEFS_SCREENSHOT_WAIT_FOR` to a readiness marker written only after the requested
scene has rendered multiple frames; capture waits for it instead of relying on
a fixed delay that can expire during scene uploads. Empty-scene and
same-name material retirement are checked by RenderServicesTest; native Metal
buffer release and subsequent reuse are checked by MetalInitializationTest.

### D2. Explicit input-to-game-logic dependency

**Accepted.** See the
[D2 verification record](../verification/0dfac0f10cfb8bc0a389475c0c23b09a8e597e5d.md)
for exact-revision three-platform results and inspected native artifacts.

InputManager takes a required borrowed `IGameLogic&` at
construction instead of downcasting Application and looking up game logic
for every event. The injected logic must outlive InputManager. Viewer and
Billiard construct logic before input and provide that dependency directly;
there are no current InputManager construction sites without game logic.
Platform event routing, ImGui input handling and callback ordering remain
unchanged. No event bus, configurable bindings or new input abstraction is added.

Preserve per-event arrow-down callbacks, first-press callbacks until release,
independent held-key state, the `u` button mapping, reserved `d`/`r` no-ops,
mouse button no-ops and signed drag deltas/analog IDs. Initialize/finalize clear
held keys without synthesizing game callbacks, so a fresh lifecycle cannot
inherit a missing key-up from a previous lifecycle.

Verification sequence:

1. Add an application-free InputManagerTest covering repeated/simultaneous
   arrows, release/repress ordering, buttons, drags and repeated lifecycles.
   Demonstrate the lifecycle case fails without state reset.
2. Compile both composition roots and run focused local tests.
3. Verify the exact source revision with Windows/Linux/macOS CI, inspect native
   Viewer/scene-cycle/Billiard screenshots and physics results, then write its
   revision-specific acceptance record. Synthetic input tests do not establish
   native keyboard/mouse interaction or every ImGui capture configuration.

#### Known limitations and follow-ups (D2)

Input semantics are tested directly without Application, not by injecting OS
events. Native focus and every ImGui capture configuration remain unverified
interactively; screenshots and physics checks do not substitute for those paths.

### D3. Explicit GUI icon image loading

**Accepted.** See the
[D3 verification record](../verification/1be28dfa8af835b753f4097546d71743dd38ac68.md)
for exact-revision three-platform results and inspected native icon artifacts.

OverlayPass supplies GuiSubPass with an image-loading callable.
The production LoadGuiIconImage function performs existing AssetLoader file
lookup and PNG decoding; GuiSubPass has no direct asset-loader/parser dependency.
The callable is required, owned by the GUI pass and returns an owned decoded
Image. A provider returning no pixel data must report the load failure; the
production provider preserves missing-file/decode diagnostics.

GuiSubPass retains lazy one-shot loading, icon filenames/order, fallback node
labels, upload-error reporting and ownership of successfully created GPU icon
textures. Those textures remain separate from scene allocations and debug
texture views, and are released while the backend is still available.
No new asset framework or application-service refactor is included.

Verification: inject decoded/missing images into GuiSubPassLayoutTest; cover
failed GPU creation, no per-frame/per-scene reload, unused-pass destruction,
partial-success cleanup and fresh GUI lifecycles. Preserve all layout checks.
Run focused local builds/tests, then exact-revision three-platform CI and
inspect actual Viewer/scene-cycle/Billiard artifacts before recording acceptance.

#### Known limitations and follow-ups (D3)

Injected failure tests cover the GUI ownership boundary, not exhaustive malformed
PNG parsing or native driver allocation failures. Production asset resolution
and visible icons are verified by the native artifact path. Broader
Viewer/Billiard service lookups remain a separate assessment.
DrawSubPass still links ImageDecoders for the production adapter; full target
isolation and native rendering of every scene-node icon kind are not claimed.

### D4a. Explicit Viewer scene dependency

**Accepted.** ViewerLogic takes a required borrowed `ISceneManager&` instead
of downcasting Application to locate it on initialization and each input event.
Viewer constructs the scene manager before logic so the service outlives its
consumer. Application remains responsible for command-line arguments and quit
requests through IApplication; initialization without it is explicitly rejected.
Billiard dependencies are deferred to D4b.

Preserve default/custom scene loading, option handling, A/B/A scene-cycle timing,
readiness markers, load-error propagation and camera movement/rotation semantics.
Do not introduce a dependency-injection framework or change platform input.

Verification: ViewerLogicTest uses an ISceneManager fake and a plain
IApplication (not BaseApplication), with no native window/backend. Cover
default/custom loading and errors, invalid cycle options, phase order, fresh
lifecycles, quit-on-load-failure, directional camera/target movement, drag
rotation, scene replacement and absent scenes/cameras. Then build Viewer and
run focused tests, exact-revision three-platform CI and native artifact checks.

Source `a96a0d7e41fc7964f48b3bd952e6dd0f88a4b589` passed CircleCI pipeline
1012 on Windows (job 4600, 60 tests), Linux (4599, 57 tests) and macOS
(4598, 57 tests). Twelve native screenshots, phase markers/loading order and
Billiard physics evidence were inspected. See the
[revision-specific verification record](../verification/a96a0d7e41fc7964f48b3bd952e6dd0f88a4b589.md).

#### Known limitations and follow-ups (D4a)

Synthetic callbacks do not establish native OS event delivery, all ImGui focus
paths or every camera orientation. Scene-cycle marker I/O remains covered by
native capture gates, not the isolated test. D4b will separately assess the
Billiard scene/physics service lookups.

### D4b. Explicit Billiard scene and physics dependencies

**Accepted.** BilliardGameLogic takes required borrowed `ISceneManager&`
and `IPhysicsManager&` references instead of downcasting Application to locate
services. The composition root constructs both services before logic, which
must not outlive either. The inactive legacy game configuration now supplies
an explicit logic factory with the same dependencies, not a global default
construction.

Preserve the billiard scene path and load result, resolve the current
`pbb_cue` node for each held-left callback and apply the existing (-100, 0, 0)
force. Missing/expired nodes and nodes without physics bodies remain valid
no-action conditions. Right-drag rotation retains its signs and per-axis
window-size scaling. Application still supplies configuration through
IApplication and must be attached before camera callbacks; loading and cue
force routing no longer require Application. No input, physics integration,
rendering or Application framework rewrite is included.

Verification: BilliardGameLogicTest injects fake scene/physics services and a
plain IApplication, without a native window/backend. Cover loading/error
propagation, exact force/node routing, missing/expired cues, same-name node
replacement, no-body results, drag filtering/scaling, camera replacement and
missing scenes/cameras. Compile both Billiard entry/configuration paths and
run focused tests, then exact-revision three-platform CI with native artifact
and physics-threshold inspection.

Source `32db35fc56942ea085ea18d419d2ee1fb816076f` passed CircleCI pipeline
1020 on Windows (job 4628, 61 tests), Linux (4627, 58 tests) and macOS
(4629, 58 tests). Twelve native screenshots, phase markers/loading order and
Billiard physics thresholds were inspected. See the
[revision-specific verification record](../verification/32db35fc56942ea085ea18d419d2ee1fb816076f.md).

#### Known limitations and follow-ups (D4b)

Synthetic input does not establish native OS delivery, exhaustive ImGui focus
paths or every camera orientation. Fake physics establishes request routing,
not numerical integration; existing native Billiard physics gates cover that
separately. The legacy configuration factory is compile-checked but is not
wired into the current native executable. Configuration access remains an
Application dependency with its existing valid-dimensions precondition.

## Working and acceptance rules

For each checkpoint:

1. Confirm scope, exclusions, ownership invariants, and compatibility behavior.
   Read the affected production paths and existing tests before adding helpers.
2. Add a few focused behavioral regressions. For a bug fix, demonstrate that
   the regression fails without the fix where feasible.
3. Run the smallest relevant local builds/tests. Keep independent modules free
   of native window/RHI dependencies where the boundary permits.
4. With authorization, push the source and verify its exact revision on every
   enabled CircleCI platform using the commands in
   [the CI configuration](../../.circleci/config.yml). Currently these are
   Windows, Linux, and macOS; Android is disabled.
5. Inspect actual required Viewer/Billiard artifacts, physics thresholds, and
   process exits. Report backend and failure-injection limitations explicitly.
   Pending, skipped, canceled, or absent jobs are not acceptance.
6. Preserve evidence in a revision-specific file under
   [verification](../verification/). Rebased and merged revisions need their own
   evidence; parent results do not transfer.
7. An intermediate checkpoint need not create a PR. If a PR is requested, satisfy
   the repository acceptance gate first. Preserve unrelated work and coordinate
   shared blockers rather than duplicating another session's fix.

[Repository instructions](../../AGENTS.md) remain authoritative. Prose-only
changes use the documentation exemption and `[ci skip]` when committed; they
do not establish new source acceptance.

## Maintaining this plan

- Update this document when a boundary is completed, scope changes, or a concrete
  dependency changes the proposed order. Do not append per-run CI status here.
- Keep historical acceptance in revision-specific records, not a rolling
  "latest verified revision" table or a README result index.
- Keep the README entry as a stable discovery link.
- Use an issue for independently scheduled work if useful, and add an ADR only
  for a significant architectural choice with alternatives and consequences.
  Neither is required for every small checkpoint.
- No fixed delivery dates are implied. Stage A covers initialization
  orchestration; stage B covers the selected native initialization/cleanup
  failures, not exhaustive driver recovery or ImGui ownership. Stage C covers
  the selected GUI ownership boundary; D1 covers scene-resource retirement.
  Later D checkpoints require their own scope and evidence.

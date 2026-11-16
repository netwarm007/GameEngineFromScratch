# Vulkan checkpoint 2: strict presentation retirement

Tested source revision: `77cf9ebcfcb40cd3c3de857f8b44be8920eafc8a`.

[CircleCI pipeline 993](https://app.circleci.com/pipelines/github/netwarm007/GameEngineFromScratch/993),
branch `agents/vulkan-rhi-implementation-plan`.
Every job's exact source revision and successful outcome were checked.

| Platform and backend | Job | Configured build and tests | Application artifacts |
| --- | --- | --- | --- |
| Windows D3D12 | 4537 | Passed | Viewer and Billiard captures inspected; physics-passed artifact present |
| Linux OpenGL | 4534 | Passed | Viewer and Billiard captures inspected; physics-passed artifact present |
| macOS Metal | 4536 | Passed | Viewer and Billiard captures inspected; physics-passed artifact present |
| Linux software Vulkan | 4535 | Passed; all four focused CTest tests passed | Shader ABI, compute, production resources, and strict native Xcb presentation |

Checkpoint 2 implementation and reference-path validation are complete, awaiting
user confirmation before checkpoint 3. This is not acceptance of the complete
Vulkan backend or of untested native Vulkan platforms.

## Vulkan evidence

The retained llvmpipe CTest log reports passing `TEST_VulkanShaderAbi`,
`TEST_VulkanCompute`, `TEST_VulkanResources`, and `TEST_VulkanFrameLoop`, with no
validation-error/VUID markers.

- Shared-HLSL shader ABI/reflection and GPU compute readback remain passing.
- Production-RHI buffer/uniform readback, alignment, texture arrays and
  cube arrays, mip generation/preservation, padded uploads, selected subresource
  updates, invalid-input rollback, and repeated destruction remain passing.
- The native regression reports 20 frame submissions, resize, deferred
  minimize/restore, presentation-fence retirement, aborted recording, and
  repeated shutdown. Its explicit success markers were checked in addition to
  the CTest outcome.
- Native initialization requires swapchain-maintenance1 presentation fences,
  matching instance dependencies, and the device feature. The implementation
  waits for pending presentation fences before retiring the swapchain and its
  synchronization resources; device-idle alone is not used as proof.
- Shutdown drains the acquisition semaphore for an acquired but unsubmitted
  frame without executing its abandoned command recording.

The native test presents cleared swapchain images. It does not render shared
engine scene passes or prove that every submission reached the display.

## Inspected captures

All six exact-revision native-backend Viewer/Billiard PNGs were opened.
Scene content, environment, docked scene tree, and docked debug panel are
visible within the application viewport. No newly observed panel overlap,
clipping, or off-viewport placement was found. Windows retains its existing
hover-magnifier overlay. Viewer shows the splash scene with colored shadows;
Billiard shows the table, balls, and cues on all three native backends.

Metal's shadow-preview thumbnails remain white, as in previous records; these
captures do not verify those debug pipelines. The captures use
D3D12/OpenGL/Metal, not Vulkan Viewer/Billiard rendering.

## Limitations

- Linux Vulkan uses lavapipe/llvmpipe under Xcb/Xvfb, not a hardware GPU.
- Windows production-RHI headless resource tests passed locally on AMD, but
  native Windows Vulkan WSI remains unverified. The installed 1.2.182 SDK now
  fails explicitly for missing maintenance headers; the earlier native
  surface-capabilities query failure is not established as resolved.
- The native Metal job does not establish MoltenVK/Cocoa Vulkan compilation
  or execution.
- Task-local official 1.3.275 and 1.4.368 headers compile the EXT and promoted
  KHR branches respectively. These compile checks do not independently verify
  native KHR-only runtime support. Newer headers mixed with the old local
  validation layer are not counted as a passing compute configuration.
- Scene descriptor integration, shared per-frame/per-batch shader consumption,
  Vulkan graphics passes and point-shadow faces/layers, MSAA rendering,
  application integration, and ImGui remain later checkpoints.
- Compressed transfers and every declared native image format are not
  exhaustively exercised by the uncompressed resource fixtures.
- Disabled Android is not covered.

This record verifies only the tested source revision, not its documentation-only
record commit or a subsequent merge. Earlier revision records remain unchanged.

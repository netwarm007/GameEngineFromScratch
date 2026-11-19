# Narrow dependency backports

## ImGui Vulkan acquire/layout and viewport rollback lifecycle

`imgui-vulkan-acquire-layout.patch` targets the locked ImGui revision
`b916186e66aa3c02b72c6ca17455516512ed8049`. It backports the backend-only removal
from upstream [4cb4b17cb5db3d9c581c4c5d539802303e7e9777](https://github.com/ocornut/imgui/commit/4cb4b17cb5db3d9c581c4c5d539802303e7e9777)
without importing its 59 intervening commits or changing the submodule.

The old helper submitted transitions for every new swapchain image before any
image was acquired. Creation/recreation now only allocates resources. The
traditional render pass already discards through `initialLayout=UNDEFINED`;
its submission waits on the acquired image's semaphore. The old pinned dynamic
rendering path also needs a **local compatibility change**, separate from the
upstream removal: its initial barrier changes to `UNDEFINED`. That path
always uses `loadOp=CLEAR`, so discarding on each acquired frame is intentional
and does not rely on a previous presentation layout. The post-render transition
to `PRESENT_SRC_KHR` and strict engine presentation-fence retirement are unchanged.

Secondary viewport creation assigns cleanup ownership before invoking the platform
surface callback and unwinds partially created surfaces, swapchains, and frame
resources if a callback or Vulkan allocation fails. Partial-frame destruction
checks the handles actually allocated, and failed swapchain replacement retires
the previous handle rather than leaking it. The standalone
`TEST_VulkanSwapchainLifecycle` injects surface, swapchain, and image-view
creation failures and verifies that viewport data and created resources are
released.

[The shared CMake preparation function](../../cmake/ImGuiVulkanBackend.cmake)
verifies newline-normalized input and patched-output content hashes and applies
the patch to a build-directory copy
for both production and standalone Vulkan tests. The submodule stays clean,
`scripts/build_imgui.sh` remains safe to rerun, and clean checkouts need only the
normal pinned submodule initialization. There is no network fetch at configure
time. Each configure starts from the original source, so patches never stack.
A changed input, failed/skipped patch, or unexpected output is an explicit
configure error; review and retire/rebase this patch when intentionally updating
ImGui.

`TEST_VulkanSwapchainLifecycle` compiles this same selected backend with a typed
test-only function dispatch. It exercises resource creation/resize, actual
out-of-date and suboptimal acquire/rebuild callbacks, acquire-semaphore
submission waits, and transactional viewport creation failure cleanup without
a GPU. When SDK headers expose dynamic rendering, it
also executes the clear/discard and presentation-layout barriers. Older headers
report that branch unavailable. The unchanged helper fails the same test by
recording GPU work before acquisition. This does not validate real Vulkan
synchronization, presentation fences, WSI, or platform input; the native frame
loop, real applications, and exact-revision platform acceptance remain required.

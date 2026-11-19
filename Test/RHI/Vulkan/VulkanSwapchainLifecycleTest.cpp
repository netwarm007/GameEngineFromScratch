// Compile the selected backend in this test so its private renderer callbacks
// and typed loader table can be exercised without exposing a production seam.
#include MYGE_IMGUI_VULKAN_BACKEND_SOURCE

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
uintptr_t identity = 1;
unsigned generations = 0;
unsigned retired = 0;
VkSwapchainKHR current = VK_NULL_HANDLE;
VkResult acquireResult = VK_SUCCESS;
bool acquired = false;
VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
unsigned submissions = 0;
unsigned transitions = 0;
unsigned renderPasses = 0;
unsigned destroyedSurfaces = 0;
unsigned destroyedImageViews = 0;
bool failSwapchainCreate = false;
int failImageViewAfter = -1;
bool failSurfaceCreate = false;
std::array<VkImage, 3> currentImages{};

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Handle>
Handle NewHandle() {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(identity++);
    } else {
        return static_cast<Handle>(identity++);
    }
}

int CreateSurface(ImGuiViewport*, ImU64, const void*, ImU64* output) {
    *output = NewHandle<ImU64>();
    return failSurfaceCreate ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
}

void VKAPI_PTR CheckVkResult(VkResult result) {
    if (result != VK_SUCCESS) throw std::runtime_error("injected Vulkan result");
}

void VKAPI_PTR DestroySurface(VkInstance, VkSurfaceKHR surface, const VkAllocationCallbacks*) {
    if (surface) ++destroyedSurfaces;
}

VkResult VKAPI_PTR SurfaceSupport(VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32* supported) {
    *supported = VK_TRUE;
    return VK_SUCCESS;
}

VkResult VKAPI_PTR SurfaceFormats(VkPhysicalDevice, VkSurfaceKHR, uint32_t* count,
                                  VkSurfaceFormatKHR* formats) {
    if (!formats) {
        *count = 1;
    } else {
        formats[0] = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        *count = 1;
    }
    return VK_SUCCESS;
}

VkResult VKAPI_PTR SurfacePresentModes(VkPhysicalDevice, VkSurfaceKHR, uint32_t* count,
                                       VkPresentModeKHR* modes) {
    if (!modes) {
        *count = 1;
    } else {
        modes[0] = VK_PRESENT_MODE_FIFO_KHR;
        *count = 1;
    }
    return VK_SUCCESS;
}

VkResult VKAPI_PTR CreateImageView(VkDevice, const VkImageViewCreateInfo*,
                                   const VkAllocationCallbacks*, VkImageView* output) {
    if (failImageViewAfter == 0) return VK_ERROR_INITIALIZATION_FAILED;
    if (failImageViewAfter > 0) --failImageViewAfter;
    *output = NewHandle<VkImageView>();
    return VK_SUCCESS;
}

void VKAPI_PTR DestroyImageView(VkDevice, VkImageView view, const VkAllocationCallbacks*) {
    if (view) ++destroyedImageViews;
}

template <typename Info, typename Handle>
VkResult VKAPI_PTR Create(VkDevice, const Info*, const VkAllocationCallbacks*, Handle* output) {
    *output = NewHandle<Handle>();
    return VK_SUCCESS;
}

template <typename Handle>
void VKAPI_PTR Destroy(VkDevice, Handle, const VkAllocationCallbacks*) {}

VkResult VKAPI_PTR WaitIdle(VkDevice) { return VK_SUCCESS; }

VkResult VKAPI_PTR Capabilities(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* output) {
    *output = {};
    output->minImageCount = 2;
    output->maxImageCount = 3;
    output->currentExtent = {(std::numeric_limits<uint32_t>::max)(),
                             (std::numeric_limits<uint32_t>::max)()};
    output->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    output->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    return VK_SUCCESS;
}

VkResult VKAPI_PTR CreateSwapchain(VkDevice, const VkSwapchainCreateInfoKHR* info,
                                   const VkAllocationCallbacks*, VkSwapchainKHR* output) {
    Require(info->oldSwapchain == current, "Rebuild lost its previous swapchain identity");
    if (failSwapchainCreate) return VK_ERROR_INITIALIZATION_FAILED;
    current = *output = NewHandle<VkSwapchainKHR>();
    ++generations;
    return VK_SUCCESS;
}

void VKAPI_PTR DestroySwapchain(VkDevice, VkSwapchainKHR value, const VkAllocationCallbacks*) {
    if (value) ++retired;
    if (value == current) current = VK_NULL_HANDLE;
}

VkResult VKAPI_PTR Images(VkDevice, VkSwapchainKHR swapchain, uint32_t* count, VkImage* images) {
    Require(swapchain == current, "Image enumeration used a retired swapchain");
    if (images) {
        Require(*count >= 3, "Insufficient swapchain image capacity");
        for (unsigned index = 0; index < currentImages.size(); ++index) {
            currentImages[index] = images[index] = NewHandle<VkImage>();
        }
    }
    *count = 3;
    return VK_SUCCESS;
}

VkResult VKAPI_PTR AllocateCommands(VkDevice, const VkCommandBufferAllocateInfo* info,
                                    VkCommandBuffer* output) {
    for (uint32_t index = 0; index < info->commandBufferCount; ++index) {
        output[index] = NewHandle<VkCommandBuffer>();
    }
    return VK_SUCCESS;
}

void VKAPI_PTR FreeCommands(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) {}

VkResult VKAPI_PTR BeginRecording(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    Require(acquired, "Swapchain preparation attempted GPU work before image acquisition");
    return VK_SUCCESS;
}

VkResult VKAPI_PTR Acquire(VkDevice, VkSwapchainKHR swapchain, uint64_t, VkSemaphore semaphore,
                           VkFence, uint32_t* index) {
    Require(swapchain == current, "Acquisition used a retired swapchain");
    acquired = acquireResult == VK_SUCCESS || acquireResult == VK_SUBOPTIMAL_KHR;
    acquireSemaphore = semaphore;
    *index = 0;
    return acquireResult;
}

VkResult VKAPI_PTR Wait(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) {
    Require(acquired, "Renderer waited for work before acquisition");
    return VK_SUCCESS;
}

VkResult VKAPI_PTR ResetPool(VkDevice, VkCommandPool, VkCommandPoolResetFlags) { return VK_SUCCESS; }
VkResult VKAPI_PTR ResetFences(VkDevice, uint32_t, const VkFence*) { return VK_SUCCESS; }
VkResult VKAPI_PTR EndRecording(VkCommandBuffer) { return VK_SUCCESS; }

VkResult VKAPI_PTR Submit(VkQueue, uint32_t count, const VkSubmitInfo* infos, VkFence) {
    Require(acquired && count == 1 && infos->waitSemaphoreCount == 1 &&
                infos->pWaitSemaphores[0] == acquireSemaphore &&
                (infos->pWaitDstStageMask[0] & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            "Render submission did not wait for the acquired image");
    ++submissions;
    acquired = false;
    return VK_SUCCESS;
}

void VKAPI_PTR BeginPass(VkCommandBuffer, const VkRenderPassBeginInfo*, VkSubpassContents) {
    Require(acquired, "Render pass began before image acquisition");
    ++renderPasses;
}
void VKAPI_PTR EndPass(VkCommandBuffer) {}

void VKAPI_PTR Barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                       VkDependencyFlags, uint32_t, const VkMemoryBarrier*,
                       uint32_t, const VkBufferMemoryBarrier*,
                       uint32_t count, const VkImageMemoryBarrier* barriers) {
    Require(acquired && count == 1, "Image transition preceded acquisition");
    const auto& barrier = *barriers;
    Require(barrier.image == currentImages[0], "Barrier used an image other than the acquired image");
    if (transitions % 2 == 0) {
        Require(barrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                    barrier.newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                "Dynamic first-use barrier relied on an uninitialized presentation layout");
    } else {
        Require(barrier.oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
                    barrier.newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                "Dynamic rendering did not restore presentation layout");
    }
    ++transitions;
}

#ifdef IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
void VKAPI_PTR BeginRendering(VkCommandBuffer, const VkRenderingInfoKHR* info) {
    Require(acquired && info->colorAttachmentCount == 1 &&
                info->pColorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR,
            "Dynamic rendering no longer discards acquired image contents");
}
void VKAPI_PTR EndRendering(VkCommandBuffer) {}
#endif

template <typename Function>
struct Reject;
template <typename Result, typename... Args>
struct Reject<Result (VKAPI_PTR *)(Args...)> {
    static Result VKAPI_PTR Call(Args...) {
        throw std::runtime_error("Unexpected Vulkan entry point in swapchain lifecycle test");
    }
};

PFN_vkVoidFunction Load(const char* name, void*) {
#define FUNCTION(symbol, implementation) \
    if (std::strcmp(name, #symbol) == 0) return reinterpret_cast<PFN_vkVoidFunction>(implementation)
    FUNCTION(vkDeviceWaitIdle, WaitIdle);
    FUNCTION(vkDestroySurfaceKHR, DestroySurface);
    FUNCTION(vkGetPhysicalDeviceSurfaceSupportKHR, SurfaceSupport);
    FUNCTION(vkGetPhysicalDeviceSurfaceFormatsKHR, SurfaceFormats);
    FUNCTION(vkGetPhysicalDeviceSurfacePresentModesKHR, SurfacePresentModes);
    FUNCTION(vkGetPhysicalDeviceSurfaceCapabilitiesKHR, Capabilities);
    FUNCTION(vkCreateSwapchainKHR, CreateSwapchain);
    FUNCTION(vkDestroySwapchainKHR, DestroySwapchain);
    FUNCTION(vkGetSwapchainImagesKHR, Images);
    FUNCTION(vkAllocateCommandBuffers, AllocateCommands);
    FUNCTION(vkFreeCommandBuffers, FreeCommands);
    FUNCTION(vkBeginCommandBuffer, BeginRecording);
    FUNCTION(vkAcquireNextImageKHR, Acquire);
    FUNCTION(vkWaitForFences, Wait);
    FUNCTION(vkResetCommandPool, ResetPool);
    FUNCTION(vkResetFences, ResetFences);
    FUNCTION(vkEndCommandBuffer, EndRecording);
    FUNCTION(vkQueueSubmit, Submit);
    FUNCTION(vkCmdBeginRenderPass, BeginPass);
    FUNCTION(vkCmdEndRenderPass, EndPass);
    FUNCTION(vkCmdPipelineBarrier, Barrier);
#undef FUNCTION
#define OBJECT(name, type) \
    if (std::strcmp(name, "vkCreate" #type) == 0) \
        return reinterpret_cast<PFN_vkVoidFunction>(&Create<Vk##type##CreateInfo, Vk##type>); \
    if (std::strcmp(name, "vkDestroy" #type) == 0) \
        return reinterpret_cast<PFN_vkVoidFunction>(&Destroy<Vk##type>)
    OBJECT(name, RenderPass);
    OBJECT(name, Framebuffer);
    OBJECT(name, CommandPool);
    OBJECT(name, Fence);
    OBJECT(name, Semaphore);
#undef OBJECT
    if (std::strcmp(name, "vkCreateImageView") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(CreateImageView);
    if (std::strcmp(name, "vkDestroyImageView") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(DestroyImageView);
#define REJECT(symbol) \
    if (std::strcmp(name, #symbol) == 0) \
        return reinterpret_cast<PFN_vkVoidFunction>(&Reject<PFN_##symbol>::Call);
    IMGUI_VULKAN_FUNC_MAP(REJECT)
#undef REJECT
    return nullptr;
}

void CheckViewportCreationRollback() {
    ImGui_ImplVulkan_Data backend;
    backend.Functions = ImGuiImplVulkanFuncs;
    auto& init = backend.VulkanInitInfo;
    init.Instance = NewHandle<VkInstance>();
    init.PhysicalDevice = NewHandle<VkPhysicalDevice>();
    init.Device = NewHandle<VkDevice>();
    init.QueueFamily = 0;
    init.MinImageCount = 2;
    init.PipelineInfoForViewports.SwapChainImageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    init.CheckVkResultFn = CheckVkResult;
    ImGui::GetIO().BackendRendererUserData = &backend;
    ImGui::GetPlatformIO().Platform_CreateVkSurface = CreateSurface;

    ImGuiViewport viewport;
    viewport.Size = {32, 24};
    const auto expectFailure = [&](const char* description) {
        bool failed = false;
        try {
            ImGui_ImplVulkan_CreateWindow(&viewport);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Require(failed && !viewport.RendererUserData, description);
    };

    auto surfacesBefore = destroyedSurfaces;
    failSurfaceCreate = true;
    expectFailure("Surface callback failure retained a secondary viewport");
    failSurfaceCreate = false;
    Require(destroyedSurfaces == surfacesBefore + 1,
            "Surface callback failure leaked its partially created surface");

    surfacesBefore = destroyedSurfaces;
    failSwapchainCreate = true;
    expectFailure("Swapchain creation failure retained a secondary viewport");
    failSwapchainCreate = false;
    Require(!current && destroyedSurfaces == surfacesBefore + 1,
            "Swapchain creation failure leaked its viewport surface or swapchain");

    surfacesBefore = destroyedSurfaces;
    const auto retiredBefore = retired;
    const auto viewsBefore = destroyedImageViews;
    failImageViewAfter = 1;
    expectFailure("Partial swapchain allocation failure retained a secondary viewport");
    failImageViewAfter = -1;
    Require(!current && retired == retiredBefore + 1 &&
                destroyedSurfaces == surfacesBefore + 1 &&
                destroyedImageViews == viewsBefore + 1,
            "Partial swapchain allocation failure leaked viewport resources");

    ImGui::GetIO().BackendRendererUserData = nullptr;
    ImGui::GetPlatformIO().Platform_CreateVkSurface = nullptr;
}

void Check(bool dynamicRendering) {
    const auto instance = NewHandle<VkInstance>();
    const auto physical = NewHandle<VkPhysicalDevice>();
    const auto device = NewHandle<VkDevice>();
    ImGui_ImplVulkan_Data backend;
    backend.Functions = ImGuiImplVulkanFuncs;
    auto& init = backend.VulkanInitInfo;
    init.Instance = instance;
    init.PhysicalDevice = physical;
    init.Device = device;
    init.MinImageCount = 2;
    init.UseDynamicRendering = dynamicRendering;
    init.PipelineInfoForViewports.SwapChainImageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ImGui::GetIO().BackendRendererUserData = &backend;
    ImGui_ImplVulkan_ViewportData data;
    auto& window = data.Window;
    window.Surface = NewHandle<VkSurfaceKHR>();
    window.SurfaceFormat = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    window.PresentMode = VK_PRESENT_MODE_FIFO_KHR;
    window.UseDynamicRendering = dynamicRendering;
    ImDrawData drawData;
    ImGuiViewport viewport;
    viewport.RendererUserData = &data;
    viewport.DrawData = &drawData;
    struct ResetBorrowedState {
        ImGuiViewport& viewport;
        ~ResetBorrowedState() {
            viewport.RendererUserData = nullptr;
            ImGui::GetIO().BackendRendererUserData = nullptr;
        }
    } reset{viewport};
    const auto before = generations;
    for (const auto size : {VkExtent2D{160, 160}, VkExtent2D{256, 128}, VkExtent2D{128, 256}}) {
        ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical, device, &window,
            0, nullptr, size.width, size.height, 2, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        Require(window.Swapchain == current && window.ImageCount == 3 &&
                    window.Width == static_cast<int>(size.width) &&
                    window.Height == static_cast<int>(size.height),
                "Unacquired swapchain preparation did not finish its resource lifecycle");
        viewport.Size = {static_cast<float>(size.width), static_cast<float>(size.height)};
        acquireResult = VK_SUCCESS;
        const auto beforeFrame = submissions;
        ImGui_ImplVulkan_RenderWindow(&viewport, nullptr);
        Require(submissions == beforeFrame + 1,
                "First use of a created/resized swapchain did not acquire and submit");
    }
    viewport.Size = {128, 256};
    const auto beforeSubmit = submissions;
    acquireResult = VK_ERROR_OUT_OF_DATE_KHR;
    ImGui_ImplVulkan_RenderWindow(&viewport, nullptr);
    Require(data.SwapChainNeedRebuild && submissions == beforeSubmit,
            "Out-of-date acquisition did not defer rendering");
    acquireResult = VK_SUBOPTIMAL_KHR;
    ImGui_ImplVulkan_RenderWindow(&viewport, nullptr);
    Require(!data.SwapChainNeedRebuild && data.SwapChainSuboptimal &&
                generations == before + 4 && submissions == beforeSubmit + 1,
            "Out-of-date rebuild did not acquire and submit safely");
    acquireResult = VK_SUCCESS;
    ImGui_ImplVulkan_RenderWindow(&viewport, nullptr);
    Require(!data.SwapChainSuboptimal && generations == before + 5 &&
                submissions == beforeSubmit + 2,
            "Suboptimal acquisition did not rebuild and retry safely");
    ImGui_ImplVulkanH_DestroyWindow(instance, device, &window, nullptr);
    Require(generations == before + 5 && retired == generations && !current,
            "Create/resize/recreate did not retire every swapchain");
    ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical, device, &window,
        0, nullptr, 128, 256, 2, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    ImGui_ImplVulkan_RenderWindow(&viewport, nullptr);
    Require(generations == before + 6 && submissions == beforeSubmit + 3,
            "Destroyed secondary swapchain could not be recreated and first-used safely");
    ImGui_ImplVulkanH_DestroyWindow(instance, device, &window, nullptr);
    Require(retired == generations && !current, "Recreated secondary swapchain was not retired");
}
}  // namespace

int main() {
    try {
        Require(ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_1, Load),
                "Unable to initialize the test-only Vulkan function dispatch");
        ImGui::CreateContext();
        CheckViewportCreationRollback();
        Check(false);
#ifdef IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
        ImGuiImplVulkanFuncs_vkCmdBeginRenderingKHR = BeginRendering;
        ImGuiImplVulkanFuncs_vkCmdEndRenderingKHR = EndRendering;
        Check(true);
        Require(transitions == 12, "Dynamic first-use/rebuild paths did not exercise both layout transitions");
#else
        std::cout << "Dynamic rendering execution unavailable in these SDK headers\n";
#endif
        Require(renderPasses == 6, "Traditional first-use/rebuild paths did not execute render passes");
        ImGui::DestroyContext();
        std::cout << "Swapchain creation/resize, out-of-date and suboptimal rebuilds passed: "
                     "no pre-acquire GPU work; render submissions wait on acquisition\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan swapchain lifecycle regression failed: " << error.what() << '\n';
        return 1;
    }
}

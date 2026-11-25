#include "VulkanGuiRenderer.hpp"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "backends/imgui_impl_vulkan.h"

using namespace My;

namespace {
std::map<ImGuiContext*, VulkanGuiRenderer*> renderers;

void CheckResult(VkResult result) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error("ImGui Vulkan operation failed: " +
                                 vk::to_string(static_cast<vk::Result>(result)));
    }
}
}  // namespace

VulkanGuiRenderer::~VulkanGuiRenderer() {
    if (!IsInitialized()) return;
    try {
        Finalize();
    } catch (const std::exception& error) {
        std::cerr << "Vulkan GUI shutdown failed: " << error.what() << '\n';
        std::terminate();
    }
}

void VulkanGuiRenderer::CheckContext() const {
    if (!m_context || ImGui::GetCurrentContext() != m_context ||
        ImGui::GetIO().BackendRendererUserData != m_backend) {
        throw std::logic_error("Vulkan GUI requires its original live, current ImGui context and renderer");
    }
}

void VulkanGuiRenderer::Initialize(vk::RenderPass renderPass, uint32_t imageCount,
                                   const VkAllocationCallbacks* allocator) {
    if (IsInitialized() || !ImGui::GetCurrentContext() || !m_rhi.GetDevice() ||
        !renderPass || imageCount < 2) {
        throw std::logic_error("Vulkan GUI initialization requires an idle renderer, context, device, and render pass");
    }
    auto& io = ImGui::GetIO();
    if (io.BackendRendererUserData || ImGui::GetMainViewport()->RendererUserData) {
        throw std::logic_error("Vulkan GUI cannot replace a borrowed ImGui renderer");
    }
    if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) &&
        (!m_rhi.HasPresentationFences() || !ImGui::GetPlatformIO().Platform_CreateVkSurface)) {
        throw std::runtime_error("Vulkan GUI secondary viewports require maintenance-aware presentation and a platform surface callback");
    }
    const std::array sizes{
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 4096),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 2)};
    auto pool = m_rhi.GetDevice().createDescriptorPoolUnique(
        {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 4098, sizes});
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_1;
    info.Instance = m_rhi.GetInstance();
    info.PhysicalDevice = m_rhi.GetPhysicalDevice();
    info.Device = m_rhi.GetDevice();
    info.QueueFamily = m_rhi.GetGraphicsQueueFamily();
    info.Queue = m_rhi.GetGraphicsQueue();
    info.DescriptorPool = *pool;
    info.Allocator = allocator;
    info.MinImageCount = 2;
    info.ImageCount = imageCount;
    info.PipelineInfoMain.RenderPass = renderPass;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineInfoForViewports.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineInfoForViewports.RenderPass = renderPass;
    info.CheckVkResultFn = CheckResult;
    if (!ImGui_ImplVulkan_LoadFunctions(info.ApiVersion, [](const char* name, void* owner) -> PFN_vkVoidFunction {
        if (std::strcmp(name, "vkQueuePresentKHR") == 0) {
            return reinterpret_cast<PFN_vkVoidFunction>(&VulkanGuiRenderer::Present);
        }
        const auto& loader = GetVulkanLoader();
        const auto function = loader.GetInstanceFunction(static_cast<VulkanRHI*>(owner)->GetInstance(), name);
        if (function) return function;
        // The helper loads WSI entry points even for offscreen use. Loader
        // trampolines are available without enabling or invoking those extensions.
#define WSI_TRAMPOLINE(functionName) \
        if (std::strcmp(name, #functionName) == 0) return loader.GetExportedFunction(name)
        WSI_TRAMPOLINE(vkAcquireNextImageKHR);
        WSI_TRAMPOLINE(vkCreateSwapchainKHR);
        WSI_TRAMPOLINE(vkDestroySwapchainKHR);
        WSI_TRAMPOLINE(vkDestroySurfaceKHR);
        WSI_TRAMPOLINE(vkGetSwapchainImagesKHR);
        WSI_TRAMPOLINE(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
        WSI_TRAMPOLINE(vkGetPhysicalDeviceSurfaceFormatsKHR);
        WSI_TRAMPOLINE(vkGetPhysicalDeviceSurfacePresentModesKHR);
        WSI_TRAMPOLINE(vkGetPhysicalDeviceSurfaceSupportKHR);
#undef WSI_TRAMPOLINE
        return nullptr;
    }, &m_rhi)) {
        throw std::runtime_error("Vulkan GUI function loading failed");
    }
    renderers.emplace(ImGui::GetCurrentContext(), this);
    try {
        if (!ImGui_ImplVulkan_Init(&info)) {
            throw std::runtime_error("Vulkan ImGui backend initialization failed");
        }
    } catch (...) {
        if (io.BackendRendererUserData) ImGui_ImplVulkan_Shutdown();
        renderers.erase(ImGui::GetCurrentContext());
        throw;
    }
    auto& platform = ImGui::GetPlatformIO();
    m_createViewport = platform.Renderer_CreateWindow;
    m_destroyViewport = platform.Renderer_DestroyWindow;
    m_renderViewport = platform.Renderer_RenderWindow;
    m_getFramebufferScale = platform.Platform_GetWindowFramebufferScale;
    m_setViewportSize = platform.Renderer_SetWindowSize;
    platform.Renderer_CreateWindow = CreateViewport;
    platform.Renderer_DestroyWindow = DestroyViewport;
    platform.Renderer_RenderWindow = RenderViewport;
    if (m_getFramebufferScale) platform.Platform_GetWindowFramebufferScale = GetWindowFramebufferScale;
    if (m_setViewportSize) platform.Renderer_SetWindowSize = SetWindowSize;
    m_context = ImGui::GetCurrentContext();
    m_backend = io.BackendRendererUserData;
    m_pool = std::move(pool);
}

void VulkanGuiRenderer::Finalize() {
    if (!IsInitialized()) return;
    CheckContext();
    m_rhi.WaitGPU();
    auto& platform = ImGui::GetPlatformIO();
    if (platform.Renderer_CreateWindow == CreateViewport) platform.Renderer_CreateWindow = m_createViewport;
    if (platform.Renderer_DestroyWindow == DestroyViewport) platform.Renderer_DestroyWindow = m_destroyViewport;
    if (platform.Renderer_RenderWindow == RenderViewport) platform.Renderer_RenderWindow = m_renderViewport;
    if (platform.Platform_GetWindowFramebufferScale == GetWindowFramebufferScale) {
        platform.Platform_GetWindowFramebufferScale = m_getFramebufferScale;
    }
    if (platform.Renderer_SetWindowSize == SetWindowSize) platform.Renderer_SetWindowSize = m_setViewportSize;
    m_viewportScales.clear();
    for (const auto& [identity, texture] : m_textures) {
        if (texture.descriptor) ImGui_ImplVulkan_RemoveTexture(texture.descriptor);
    }
    m_textures.clear();
    ImGui_ImplVulkan_Shutdown();
    renderers.erase(m_context);
    m_pool.reset();
    m_context = nullptr;
    m_backend = nullptr;
}

void VulkanGuiRenderer::NewFrame() {
    CheckContext();
    ImGui_ImplVulkan_NewFrame();
}

void VulkanGuiRenderer::PrepareTextures() {
    CheckContext();
    for (auto* texture : ImGui::GetPlatformIO().Textures) {
        if (texture->Status != ImTextureStatus_OK) ImGui_ImplVulkan_UpdateTexture(texture);
    }
}

void VulkanGuiRenderer::RegisterTexture(ImTextureID identity, vk::ImageView view, vk::ImageLayout layout,
                                        vk::Format format) {
    CheckContext();
    if (!identity || !view || m_textures.contains(identity) ||
        (layout != vk::ImageLayout::eShaderReadOnlyOptimal &&
         layout != vk::ImageLayout::eDepthStencilReadOnlyOptimal)) {
        throw std::invalid_argument("Invalid or duplicate Vulkan GUI texture registration");
    }
    const auto features = m_rhi.GetPhysicalDevice().getFormatProperties(format).optimalTilingFeatures;
    if (!(features & vk::FormatFeatureFlagBits::eSampledImage)) {
        throw std::invalid_argument("Vulkan GUI texture format is not sampleable");
    }
    const bool nearest = !(features & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
    m_textures.emplace(identity, GuiTexture{view, layout, format, VK_NULL_HANDLE, nearest});
}

void VulkanGuiRenderer::ReleaseTexture(ImTextureID identity) {
    CheckContext();
    const auto found = m_textures.find(identity);
    if (found == m_textures.end()) throw std::invalid_argument("Unknown Vulkan GUI texture identity");
    if (found->second.descriptor) {
        m_rhi.WaitGPU();
        ImGui_ImplVulkan_RemoveTexture(found->second.descriptor);
    }
    m_textures.erase(found);
}

void VulkanGuiRenderer::Render(ImDrawData& drawData, vk::CommandBuffer command) {
    CheckContext();
    if (!command || !drawData.Valid || drawData.OwnerViewport != ImGui::GetMainViewport()) {
        throw std::invalid_argument("Vulkan GUI rendering requires completed main-viewport data and a command buffer");
    }
    WithTextures(drawData, [&] { ImGui_ImplVulkan_RenderDrawData(&drawData, command); });
}

void VulkanGuiRenderer::WithTextures(ImDrawData& drawData, const std::function<void()>& render) {
    struct Remapping {
        std::vector<std::pair<ImDrawList*, ImVector<ImDrawCmd>>> values;
        ~Remapping() {
            for (auto& [list, original] : values) list->CmdBuffer.swap(original);
        }
    } remapping;
    auto sampler = ImGui::GetPlatformIO().DrawCallback_SetSamplerLinear;
    for (auto* list : drawData.CmdLists) {
        ImVector<ImDrawCmd> translated;
        for (auto draw : list->CmdBuffer) {
            if (draw.UserCallback) {
                if (draw.UserCallback == ImGui::GetPlatformIO().DrawCallback_SetSamplerNearest ||
                    draw.UserCallback == ImGui::GetPlatformIO().DrawCallback_SetSamplerLinear) {
                    sampler = draw.UserCallback;
                } else if (draw.UserCallback == ImDrawCallback_ResetRenderState ||
                           draw.UserCallback == ImGui::GetPlatformIO().DrawCallback_ResetRenderState) {
                    sampler = ImGui::GetPlatformIO().DrawCallback_SetSamplerLinear;
                }
            }
            bool nearest = false;
            if (!draw.UserCallback && draw.ElemCount && !draw.TexRef._TexData) {
                const auto found = m_textures.find(draw.GetTexID());
                if (found == m_textures.end()) {
                    throw std::invalid_argument("Vulkan GUI draw references an unknown or released texture");
                }
                if (!found->second.descriptor) {
                    found->second.descriptor = ImGui_ImplVulkan_AddTexture(
                        found->second.view, static_cast<VkImageLayout>(found->second.layout));
                    if (!found->second.descriptor) {
                        throw std::runtime_error("Vulkan GUI texture descriptor allocation returned null");
                    }
                }
                draw.TexRef = ImTextureRef(reinterpret_cast<ImTextureID>(found->second.descriptor));
                nearest = found->second.nearest;
            }
            if (nearest) {
                ImDrawCmd select;
                select.UserCallback = ImGui::GetPlatformIO().DrawCallback_SetSamplerNearest;
                translated.push_back(select);
            }
            translated.push_back(draw);
            if (nearest) {
                ImDrawCmd restore;
                restore.UserCallback = sampler;
                translated.push_back(restore);
            }
        }
        remapping.values.emplace_back(list, ImVector<ImDrawCmd>{});
        remapping.values.back().second.swap(list->CmdBuffer);
        list->CmdBuffer.swap(translated);
    }
    render();
}

VulkanGuiRenderer& VulkanGuiRenderer::Current() {
    const auto found = renderers.find(ImGui::GetCurrentContext());
    if (found == renderers.end()) throw std::logic_error("No Vulkan GUI renderer owns the current viewport");
    found->second->CheckContext();
    return *found->second;
}

VkResult VKAPI_PTR VulkanGuiRenderer::Present(VkQueue queue, const VkPresentInfoKHR* info) {
    auto& owner = Current();
    if (!info || vk::Queue(queue) != owner.m_rhi.GetGraphicsQueue()) {
        throw std::invalid_argument("Vulkan GUI secondary presentation used an unexpected queue");
    }
    return static_cast<VkResult>(owner.m_rhi.PresentAndWait(queue, *info));
}

void VulkanGuiRenderer::CreateViewport(ImGuiViewport* viewport) {
    auto& owner = Current();
    if (!owner.m_rhi.HasPresentationFences() || !viewport || viewport->RendererUserData ||
        !ImGui::GetPlatformIO().Platform_CreateVkSurface) {
        throw std::logic_error("Vulkan secondary viewport requires strict WSI support and an unbound platform window");
    }
    try {
        owner.m_createViewport(viewport);
        if (!viewport->RendererUserData) throw std::runtime_error("Vulkan viewport surface/queue support is unavailable");
    } catch (...) {
        if (viewport->RendererUserData) owner.m_destroyViewport(viewport);
        owner.m_viewportScales.erase(viewport);
        throw;
    }
}

void VulkanGuiRenderer::DestroyViewport(ImGuiViewport* viewport) {
    auto& owner = Current();
    if (!viewport) throw std::invalid_argument("Vulkan secondary viewport is null");
    try {
        owner.m_destroyViewport(viewport);
    } catch (...) {
        owner.m_viewportScales.erase(viewport);
        throw;
    }
    owner.m_viewportScales.erase(viewport);
}

ImVec2 VulkanGuiRenderer::GetWindowFramebufferScale(ImGuiViewport* viewport) {
    auto& owner = Current();
    if (!viewport || !owner.m_getFramebufferScale) {
        throw std::logic_error("Vulkan viewport framebuffer scale callback is unavailable");
    }
    const auto scale = owner.m_getFramebufferScale(viewport);
    if (!(scale.x > 0) || !(scale.y > 0)) {
        throw std::runtime_error("Vulkan viewport framebuffer scale must be positive");
    }
    const auto [found, inserted] = owner.m_viewportScales.try_emplace(viewport, scale);
    const bool changed = found->second.x != scale.x || found->second.y != scale.y;
    if (viewport->RendererUserData && owner.m_setViewportSize &&
        ((inserted && (scale.x != 1 || scale.y != 1)) || changed)) {
        owner.m_setViewportSize(viewport, {viewport->Size.x * scale.x, viewport->Size.y * scale.y});
    }
    found->second = scale;
    return scale;
}

void VulkanGuiRenderer::SetWindowSize(ImGuiViewport* viewport, ImVec2 size) {
    auto& owner = Current();
    if (!viewport || !owner.m_setViewportSize) {
        throw std::logic_error("Vulkan viewport resize callback is unavailable");
    }
    const auto scale = viewport->FramebufferScale;
    owner.m_setViewportSize(viewport, {size.x * scale.x, size.y * scale.y});
}

void VulkanGuiRenderer::RenderViewport(ImGuiViewport* viewport, void* argument) {
    auto& owner = Current();
    if (!viewport || !viewport->DrawData || !viewport->DrawData->Valid) {
        throw std::invalid_argument("Vulkan secondary viewport has no completed draw data");
    }
    owner.WithTextures(*viewport->DrawData, [&] { owner.m_renderViewport(viewport, argument); });
}

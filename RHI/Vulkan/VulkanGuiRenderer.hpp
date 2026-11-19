#pragma once

#include "VulkanRHI.hpp"
#include "imgui.h"
#include <functional>

namespace My {
class VulkanGuiRenderer {
   public:
    explicit VulkanGuiRenderer(VulkanRHI& rhi) : m_rhi(rhi) {}
    ~VulkanGuiRenderer();
    VulkanGuiRenderer(const VulkanGuiRenderer&) = delete;
    VulkanGuiRenderer& operator=(const VulkanGuiRenderer&) = delete;

    // A supplied Vulkan host allocator must remain live through Finalize.
    void Initialize(vk::RenderPass renderPass, uint32_t imageCount,
                    const VkAllocationCallbacks* allocator = nullptr);
    void Finalize();
    bool IsInitialized() const { return m_context != nullptr; }
    void NewFrame();
    void PrepareTextures();
    void Render(ImDrawData& drawData, vk::CommandBuffer command);
    void RegisterTexture(ImTextureID identity, vk::ImageView view, vk::ImageLayout layout, vk::Format format);
    void ReleaseTexture(ImTextureID identity);

   private:
    void CheckContext() const;
    void WithTextures(ImDrawData& drawData, const std::function<void()>& render);
    static VulkanGuiRenderer& Current();
    static VkResult VKAPI_PTR Present(VkQueue queue, const VkPresentInfoKHR* info);
    static void RenderViewport(ImGuiViewport* viewport, void* argument);
    static void CreateViewport(ImGuiViewport* viewport);
    static void DestroyViewport(ImGuiViewport* viewport);
    static ImVec2 GetWindowFramebufferScale(ImGuiViewport* viewport);
    static void SetWindowSize(ImGuiViewport* viewport, ImVec2 size);

    VulkanRHI& m_rhi;
    ImGuiContext* m_context = nullptr;
    void* m_backend = nullptr;
    vk::UniqueDescriptorPool m_pool;
    struct GuiTexture {
        vk::ImageView view;
        vk::ImageLayout layout;
        vk::Format format;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
        bool nearest;
    };
    std::map<ImTextureID, GuiTexture> m_textures;
    void (*m_createViewport)(ImGuiViewport*) = nullptr;
    void (*m_destroyViewport)(ImGuiViewport*) = nullptr;
    void (*m_renderViewport)(ImGuiViewport*, void*) = nullptr;
    ImVec2 (*m_getFramebufferScale)(ImGuiViewport*) = nullptr;
    void (*m_setViewportSize)(ImGuiViewport*, ImVec2) = nullptr;
    std::map<ImGuiViewport*, ImVec2> m_viewportScales;
};
}  // namespace My

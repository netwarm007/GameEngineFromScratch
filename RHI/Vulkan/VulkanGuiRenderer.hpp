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

    void Initialize(vk::RenderPass renderPass, uint32_t imageCount);
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

    VulkanRHI& m_rhi;
    ImGuiContext* m_context = nullptr;
    void* m_backend = nullptr;
    vk::UniqueDescriptorPool m_pool;
    struct GuiTexture {
        VkDescriptorSet descriptor;
        bool nearest;
    };
    std::map<ImTextureID, GuiTexture> m_textures;
    void (*m_createViewport)(ImGuiViewport*) = nullptr;
    void (*m_renderViewport)(ImGuiViewport*, void*) = nullptr;
    void (*m_destroyViewport)(ImGuiViewport*) = nullptr;
};
}  // namespace My

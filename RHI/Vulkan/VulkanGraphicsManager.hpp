#pragma once
#include "VulkanPipelineStateManager.hpp"
#include "GraphicsManager.hpp"
#include "SceneObject.hpp"
#include "VulkanRHI.hpp"

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace My {
class VulkanGraphicsManager : public GraphicsManager {
   public:
    using GraphicsManager::GraphicsManager;
    ~VulkanGraphicsManager() override;

    int Initialize() final;
    void Finalize() final;

    void Present() final;
    void ResizeCanvas(int32_t width, int32_t height) final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    void GenerateCubemapArray(TextureCubeArray& texture_array) final;

    void GenerateTextureArray(Texture2DArray& texture_array) final;

    void BeginShadowMap(const int32_t light_index, const TextureBase* pShadowmap,
                        const int32_t layer_index, const Frame& frame) final;
    void EndShadowMap(const TextureBase* pShadowmap,
                      const int32_t layer_index, const Frame& frame) final;
    void SetShadowMaps(const Frame& frame) final;
    void GenerateTexture(Texture2D& texture) final;
    void CreateTexture(SceneObjectTexture& texture) final;
    Texture2D CreateTexture(Image& image) final;
    void CreateTextureView(Texture2D& textureView,
                           const TextureArrayBase& textureArray,
                           uint32_t slice, uint32_t mip) final;
    void UpdateTexture(Texture2D& texture, Image& image) final;
    void ReleaseTexture(TextureBase& texture) final;
    TextureHandler GetImGuiTextureHandle(
        const TextureBase& texture) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    void GenerateTextureForWrite(Texture2D& texture) final;

    void BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) final;

    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

   protected:
    void EndScene() final;

    void BeginFrame(Frame& frame) final;
    void EndFrame(Frame& frame) final;

    void BeginPass(Frame& frame) final;
    void EndPass(Frame& frame) final;

    void BeginCompute() final;
    void EndCompute() final;

    void initializeGeometries(const Scene& scene) final;
    void initializeSkyBox(const Scene& scene) final;

    void SetPerFrameConstants(const Frame& frame);
    void SetLightInfo(const Frame& frame);

    size_t CreateIndexBuffer(const SceneObjectIndexArray& index_array);
    size_t CreateVertexBuffer(const SceneObjectVertexArray& v_property_array);

   protected:
    struct VulkanDrawBatchContext : public DrawBatchContext {
        Vector4f baseColor{0.8f, 0.8f, 0.8f, 1.0f};
        Vector4f materialParameters{0.0f, 1.0f, 1.0f, 0.0f};
        uint32_t index_count{0};
        size_t index_offset{0};
        size_t position_buffer_offset{0};
        size_t normal_buffer_offset{0};
        size_t texcoord_buffer_offset{0};
        size_t tangent_buffer_offset{0};
        uint32_t property_count{0};
        size_t property_offset{0};
        size_t descriptor_offset{0};
        std::array<vk::ImageView, 5> materialImageViews{};
    };

    struct VulkanSceneBatchConstants {
        Matrix4X4f modelMatrix;
        Vector4f baseColor;
        Vector4f materialParameters;
    };

   private:
    struct SwapchainResources {
        vk::RenderPass renderPass;
        vk::RenderPass offscreenRenderPass;
    };

    struct ShadowMapArrayResource {
        VulkanRHI::Texture texture;
        std::vector<vk::ImageView> layerViews;
        std::vector<vk::Framebuffer> framebuffers;
    };

    struct ShadowDescriptorResources {
        vk::DescriptorPool pool;
        std::vector<vk::DescriptorSet> sets;
    };

    struct OffscreenFrameResources {
        VulkanRHI::Texture multisampleColor;
        VulkanRHI::Texture depth;
        vk::Framebuffer framebuffer;
        uint32_t width{0};
        uint32_t height{0};
        vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
    };

    struct TextureViewResource {
        vk::ImageView view;
        TextureHandler source{0};
        bool owned{false};
    };

    std::shared_ptr<SwapchainResources> m_SwapchainResources;
    vk::RenderPass m_ShadowMapRenderPass;
    VulkanDrawBatchContext m_dbcSkyBox;
    std::vector<VulkanRHI::IndexBuffer> m_IndexBuffers;
    std::vector<VulkanRHI::VertexBuffer> m_VertexBuffers;
    std::vector<VulkanRHI::UniformBuffer> m_PerFrameUniformBuffers;
    std::vector<VulkanRHI::UniformBuffer> m_LightStorageBuffers;
    std::vector<VulkanRHI::UniformBuffer> m_ShadowConstantsBuffers;
    std::vector<std::vector<VulkanRHI::UniformBuffer>> m_PerBatchUniformBuffers;
    std::unordered_map<const SceneObjectTexture*, VulkanRHI::Texture>
        m_MaterialTextures;
    std::unordered_map<TextureHandler, VulkanRHI::Texture> m_StandaloneTextures;
    std::unordered_map<TextureHandler, VulkanRHI::Texture>
        m_RenderTargetTextures;
    std::unordered_map<TextureHandler, TextureViewResource> m_TextureViews;
    VulkanRHI::Texture m_DefaultMaterialTexture;
    VulkanRHI::Texture m_DefaultNormalTexture;
    VulkanRHI::Texture m_DefaultEnvironmentTexture;
    VulkanRHI::Texture m_BRDFTexture;
    VulkanRHI::Texture m_SkyBoxTexture;
    vk::Sampler m_MaterialSampler;
    vk::Sampler m_SkyBoxSampler;
    vk::DescriptorPool m_DescriptorPool;
    vk::DescriptorPool m_SkyBoxDescriptorPool;
    vk::DescriptorPool m_BRDFDescriptorPool;
    std::vector<vk::DescriptorSet> m_DescriptorSets;
    std::vector<vk::DescriptorSet> m_SkyBoxDescriptorSets;
    vk::DescriptorSet m_BRDFDescriptorSet;
    std::unordered_map<const VulkanPipelineState*, ShadowDescriptorResources>
        m_ShadowDescriptors;
    std::shared_ptr<VulkanPipelineState> m_CurrentPipelineState;
    std::vector<VulkanRHI::DrawCommand> m_DrawCommands;
    std::vector<VulkanRHI::DrawCommand> m_ShadowDrawCommands;
    std::unordered_map<TextureHandler, ShadowMapArrayResource>
        m_ShadowMapArrays;
    std::vector<OffscreenFrameResources> m_OffscreenFrames;
    std::unordered_map<TextureHandler, vk::DescriptorSet> m_ImGuiTextures;
    std::optional<VulkanRHI::OffscreenPass> m_CurrentOffscreenPass;
    std::optional<vk::Extent2D> m_PendingCanvasResize;
    vk::Framebuffer m_CurrentShadowFramebuffer;
    vk::Extent2D m_CurrentShadowExtent{};
    ShadowMapConstants m_CurrentShadowConstants{};
    size_t m_CurrentShadowBaseLayer{0};
    std::unordered_set<std::string> m_ReportedUnsupportedPipelineStates;
    bool m_bRenderingShadowMap{false};
    bool m_bCurrentShadowCube{false};
    bool m_bImGuiInitialized{false};

    void createBatchDescriptorSets(const VulkanPipelineState& pipelineState,
                                   const Frame& frame);
    void createSkyBoxDescriptorSets(
        const VulkanPipelineState& pipelineState);
    void createShadowDescriptorSets(const VulkanPipelineState& pipelineState,
                                    const Frame& frame);
    void destroyShadowMapArrays();
    void destroyOffscreenFrames();
    void destroyTextureViewsForSource(TextureHandler source);
    void destroyUniformBuffers(
        std::vector<VulkanRHI::UniformBuffer>& uniformBuffers);
    void initializeImGuiBackend();
    void removeImGuiTexture(TextureHandler texture);
    void clearImGuiTextures();

   protected:
    [[nodiscard]] size_t PendingDrawCommandCount() const {
        return m_DrawCommands.size();
    }
};
}  // namespace My

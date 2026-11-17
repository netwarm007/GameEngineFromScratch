#pragma once
#include "VulkanPipelineStateManager.hpp"
#include "GraphicsManager.hpp"
#include "SceneObject.hpp"

namespace My {
class VulkanGraphicsManager : public GraphicsManager {
   public:
    explicit VulkanGraphicsManager(VulkanRHI& rhi) : m_rhi(rhi) {}
    VulkanGraphicsManager(VulkanRHI& rhi, ISceneManager& scenes,
                          IPipelineStateManager& pipelines)
        : GraphicsManager(scenes, pipelines), m_rhi(rhi) {}
    ~VulkanGraphicsManager() override;

    int Initialize() final;
    void Finalize() final;

    void Present() final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    void GenerateTexture(Texture2D& texture) final;
    void GenerateCubemapArray(TextureCubeArray& texture_array) final;

    void GenerateTextureArray(Texture2DArray& texture_array) final;

    void BeginShadowMap(const int32_t light_index, const TextureBase* pShadowmap,
                        const int32_t layer_index, const Frame& frame) final;
    void EndShadowMap(const TextureBase* pShadowmap,
                      const int32_t layer_index, const Frame& frame) final;
    void SetShadowMaps(const Frame& frame) final;
    void CreateTexture(SceneObjectTexture& texture) final;
    void ReleaseTexture(TextureBase& texture) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    void GenerateTextureForWrite(Texture2D& texture) final;

    void BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) final;

    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

    std::vector<uint8_t> ReadTexture(const TextureBase& texture);

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

   private:
    struct OwnedTexture {
        explicit OwnedTexture(VulkanRHI& owner) : rhi(owner) {}
        ~OwnedTexture() { rhi.DestroyTexture(texture); }
        VulkanRHI& rhi;
        VulkanRHI::Texture texture;
    };

    void GenerateTextureResource(TextureBase& texture, uint32_t layers,
                                 vk::ImageViewType viewType,
                                 vk::ImageUsageFlags usage, vk::Format format);
    VulkanRHI::Texture& GetTextureResource(const TextureBase& texture);

    VulkanRHI& m_rhi;
    std::map<TextureHandler, std::unique_ptr<OwnedTexture>> m_ownedTextures;
    TextureHandler m_nextTexture = 0;
    bool m_computeActive = false;
    std::shared_ptr<VulkanPipelineState> m_computePipeline;
    TextureHandler m_computeOutput = 0;

    struct VulkanDrawBatchContext : public DrawBatchContext {
        uint32_t index_count{0};
        size_t index_offset{0};
        uint32_t property_count{0};
        size_t property_offset{0};
        size_t descriptor_offset{0};
    };

    VulkanDrawBatchContext m_dbcSkyBox;
};
}  // namespace My

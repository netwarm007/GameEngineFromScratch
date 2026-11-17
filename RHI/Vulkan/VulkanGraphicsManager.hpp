#pragma once
#include "VulkanPipelineStateManager.hpp"
#include "VulkanGuiRenderer.hpp"
#include "GraphicsManager.hpp"
#include "SceneObject.hpp"
#include <chrono>
#include <filesystem>

namespace My {
class VulkanGraphicsManager : public GraphicsManager {
   public:
    explicit VulkanGraphicsManager(VulkanRHI& rhi) : m_rhi(rhi) {}
    VulkanGraphicsManager(VulkanRHI& rhi, ISceneManager& scenes,
                          VulkanPipelineStateManager& pipelines)
        : GraphicsManager(scenes, pipelines), m_rhi(rhi), m_pipelines(&pipelines) {}
    ~VulkanGraphicsManager() override;

    int Initialize() final;
    void Finalize() final;

    void Present() final;
    void Draw() final;
    bool RenderTargetHasTopLeftOrigin() const final { return true; }
    void ResizeCanvas(int32_t width, int32_t height) final;
    void RequestScreenshot(const std::filesystem::path& path);

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
    Texture2D CreateTexture(Image& image) final;
    void UpdateTexture(Texture2D& texture, Image& image) final;
    void CreateTextureView(Texture2D& texture, const TextureArrayBase& source,
                           uint32_t slice, uint32_t mip) final;
    void ReleaseTexture(TextureBase& texture) final;

    // skybox
    void DrawSkyBox(const Frame& frame) final;

    void GenerateTextureForWrite(Texture2D& texture) final;

    void BindTextureForWrite(Texture2D& texture, const uint32_t slot_index) final;

    void Dispatch(const uint32_t width, const uint32_t height,
                  const uint32_t depth) final;

    std::vector<uint8_t> ReadTexture(const TextureBase& texture);
    std::vector<uint8_t> ReadTexture(const TextureBase& texture, uint32_t layer, uint32_t mip = 0);
    void MSAAResolve(std::optional<std::reference_wrapper<Texture2D>> target,
                     Texture2D& source) final;

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
    void InitializeGui(vk::RenderPass renderPass);
    void PrepareGui();
    void RenderGui(ImDrawData& drawData, vk::CommandBuffer command);

   private:
    struct OwnedTexture {
        explicit OwnedTexture(VulkanRHI& owner) : rhi(owner) {}
        ~OwnedTexture() { if (!parent) rhi.DestroyTexture(texture); }
        VulkanRHI& rhi;
        VulkanRHI::Texture texture;
        std::shared_ptr<OwnedTexture> parent;
        vk::UniqueImageView view;
        bool guiVisible = false;
        uint32_t slice = 0;
        uint32_t mip = 0;
    };

    void GenerateTextureResource(TextureBase& texture, uint32_t layers,
                                 vk::ImageViewType viewType,
                                 vk::ImageUsageFlags usage, vk::Format format);
    VulkanRHI::Texture& GetTextureResource(const TextureBase& texture);
    TextureHandler StoreTexture(std::shared_ptr<OwnedTexture> texture);

    VulkanRHI& m_rhi;
    VulkanPipelineStateManager* m_pipelines = nullptr;
    std::map<TextureHandler, std::shared_ptr<OwnedTexture>> m_ownedTextures;
    TextureHandler m_nextTexture = 0;
    bool m_computeActive = false;
    std::shared_ptr<VulkanPipelineState> m_computePipeline;
    TextureHandler m_computeOutput = 0;

    struct BufferResource {
        vk::UniqueDeviceMemory memory;
        vk::UniqueBuffer buffer;
    };
    struct VulkanDrawBatchContext : DrawBatchContext {
        std::array<vk::Buffer, 4> vertices{};
        vk::Buffer indices;
        vk::IndexType indexType = vk::IndexType::eUint32;
        uint32_t indexCount = 0;
    };
    struct RenderResources {
        std::vector<vk::UniqueImageView> views;
        std::vector<vk::UniqueRenderPass> passes;
        std::vector<vk::UniqueFramebuffer> framebuffers;
        std::vector<vk::UniqueDescriptorPool> descriptorPools;
        uint32_t descriptorCount = 0;
        void Clear() {
            framebuffers.clear();
            passes.clear();
            views.clear();
            descriptorPools.clear();
            descriptorCount = 0;
        }
    };

    vk::CommandBuffer Command() const;
    void BeginRenderTarget(VulkanRHI::Texture* color, VulkanRHI::Texture& depth,
                           bool clear, uint32_t firstLayer = 0, uint32_t layers = 1);
    void FinishRenderTarget();
    uint32_t UploadConstants(const void* data, size_t size);
    void BindDraw(const Frame& frame, const material_textures& material, uint32_t batchOffset);
    void DrawBatches(const Frame& frame);
    vk::Buffer UploadVertex(const void* data, size_t size);
    vk::Buffer UploadIndex(const void* data, size_t size);
    Texture2D SolidTexture(const std::array<uint8_t, 4>& pixel);
    Texture2D CreateMaterialTexture(Image& image);
    void CopyMaterialBase(VulkanRHI::Texture& source, VulkanRHI::Texture& target);
    Texture2D MaterialTexture(const std::shared_ptr<SceneObjectTexture>& source,
                              const std::array<uint8_t, 4>& value);
    void InitializeFallbacks();
    void InitializeDepth(TextureBase& texture);
    void CreateGuiTargets();
    void ComposeGui();
    void SubmitGuiFrame();
    void InitializeCapture();

    std::vector<BufferResource> m_buffers;
    std::map<const SceneObjectTexture*, Texture2D> m_materialTextures;
    std::array<Texture2D, 4> m_fallback2D;
    Texture2DArray m_fallbackShadow;
    TextureCubeArray m_fallbackCubeShadow;
    TextureCubeArray m_fallbackSky;
    vk::UniqueSampler m_sampler;
    vk::UniqueSampler m_shadowSampler;
    std::vector<VulkanRHI::UniformBuffer> m_uniforms;
    std::array<RenderResources, GfxConfiguration::kMaxInFlightFrameCount> m_renderResources;
    std::vector<uint32_t> m_batchOffsets;
    uint32_t m_uniformCursor = 0;
    uint32_t m_frameOffset = 0;
    uint32_t m_lightOffset = 0;
    uint32_t m_passOffset = 0;
    uint32_t m_skyOffset = 0;
    bool m_rendering = false;
    bool m_frameRecording = false;
    vk::RenderPass m_renderPass;
    VulkanRHI::Texture* m_renderColor = nullptr;
    VulkanRHI::Texture* m_renderDepth = nullptr;
    std::shared_ptr<VulkanPipelineState> m_graphicsPipeline;
    VulkanDrawBatchContext m_skyBatch;
    ShadowMapConstants m_shadowConstants{};
    bool m_shadowOmni = false;
    bool m_shadowSingleFace = false;
    uint32_t m_shadowFirstLayer = 0;
    vk::UniqueRenderPass m_guiPass;
    std::vector<vk::UniqueFramebuffer> m_guiTargets;
    vk::Format m_guiFormat = vk::Format::eUndefined;
    std::unique_ptr<VulkanGuiRenderer> m_gui;
    std::filesystem::path m_screenshot;
    std::filesystem::path m_screenshotGate;
    std::chrono::steady_clock::time_point m_screenshotDue{};
    bool m_quitAfterCapture = false;
    int32_t m_pendingWidth = 0;
    int32_t m_pendingHeight = 0;
    std::unique_ptr<VulkanRHI::Commands> m_commands;
};
}  // namespace My

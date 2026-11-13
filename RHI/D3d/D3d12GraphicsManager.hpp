#pragma once
#include <DXGI1_4.h>
#include <d3d12.h>
#include <stdint.h>

#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Buffer.hpp"
#include "D3d12PipelineStateManager.hpp"
#include "GraphicsManager.hpp"
#include "Image.hpp"
#include "SceneObject.hpp"
#include "D3d12RHI.hpp"

namespace My {
class D3d12GraphicsManager : public GraphicsManager {
   public:
    ~D3d12GraphicsManager() override;

    int Initialize() final;
    void Finalize() final;
    void Tick() final;

    void Present() final;
    bool RenderTargetHasTopLeftOrigin() const final { return true; }
    void ResizeCanvas(int32_t width, int32_t height) final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    void GenerateCubemapArray(TextureCubeArray& texture_array) final;

    void GenerateTextureArray(Texture2DArray& texture_array) final;
    void GenerateTexture(Texture2D& texture) final;
    void CreateTextureView(Texture2D& texture_view, const TextureArrayBase& texture_array,
                           uint32_t slice, uint32_t mip) final;
    void MSAAResolve(std::optional<std::reference_wrapper<Texture2D>> target,
                     Texture2D& source) final;

    Texture2D CreateTexture(Image& img) final;
    void UpdateTexture(Texture2D& texture, Image& img) final; 

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

    void BindTexture(Texture2D& texture, const uint32_t slot_index) final;
    void BindDebugTexture(Texture2D& texture, const uint32_t slot_index) final;

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

    HRESULT CreatePSO(D3d12PipelineState& pipelineState);
    void BindDrawResources(const Frame& frame, const DrawBatchContext& batch);
    void TransitionTexture(ID3D12Resource* resource, D3D12_RESOURCE_STATES state);
    void CreateImGuiSrv(Texture2D& texture, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc);
    void UpdateDebugViews();

   private:
    struct RenderTexture {
        ID3D12DescriptorHeap* viewHeap = nullptr;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };
    std::unordered_map<ID3D12Resource*, RenderTexture> m_renderTextures;
    struct DebugTextureView {
        ID3D12Resource* source;
        uint32_t subresource;
    };
    std::unordered_map<ID3D12Resource*, DebugTextureView> m_debugViews;
    std::unordered_map<std::string, Texture2D> m_materialTextures;
    std::unordered_map<TextureHandler, uint32_t> m_imguiSlots;
    std::vector<uint32_t> m_freeImGuiSlots;
    uint32_t m_nextImGuiSlot = 1;
    int32_t m_pendingWidth = 0;
    int32_t m_pendingHeight = 0;
    ID3D12DescriptorHeap* m_pCbvSrvUavHeapImGui = nullptr;
    ID3D12DescriptorHeap* m_pCbvSrvUavHeap = nullptr;
    ID3D12DescriptorHeap* m_pComputeHeap = nullptr;
    ID3D12DescriptorHeap* m_pSamplerHeap = nullptr;
    ID3D12Resource* m_pComputeTexture = nullptr;
    ID3D12PipelineState* m_pMipmapPipeline = nullptr;
    ID3D12RootSignature* m_pMipmapRoot = nullptr;
    ID3D12DescriptorHeap* m_pShadowDsvHeap = nullptr;
    std::array<ID3D12DescriptorHeap*, GfxConfiguration::kMaxInFlightFrameCount> m_pDrawHeaps{};
    std::array<D3d12RHI::ConstantBuffer, GfxConfiguration::kMaxInFlightFrameCount> m_LightBuffers{};
    std::array<D3d12RHI::ConstantBuffer, GfxConfiguration::kMaxInFlightFrameCount> m_ShadowBuffers{};
    std::unordered_set<ID3D12Resource*> m_RenderedShadowMaps;
    uint32_t m_nextDrawDescriptor = 0;
    int32_t m_currentShadowLight = -1;

    struct D3dDrawBatchContext : public DrawBatchContext {
        uint32_t index_count{0};
        size_t index_offset{0};
        uint32_t property_count{0};
        size_t property_offset{0};
        size_t cbv_srv_uav_offset{0};
    };

    D3dDrawBatchContext m_dbcSkyBox;
    std::vector<D3d12RHI::IndexBuffer> m_IndexBuffers;
    std::vector<D3d12RHI::VertexBuffer> m_VertexBuffers;
    std::array<D3d12RHI::ConstantBuffer, GfxConfiguration::kMaxInFlightFrameCount> m_PerFrameConstantBuffers;
    std::array<D3d12RHI::ConstantBuffer, GfxConfiguration::kMaxInFlightFrameCount> m_PerBatchConstantBuffers;
};
}  // namespace My

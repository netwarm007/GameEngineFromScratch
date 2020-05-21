#pragma once
#include <DXGI1_4.h>
#include <d3d12.h>
#include <stdint.h>

#include <map>
#include <vector>

#include "Buffer.hpp"
#include "D3d12PipelineStateManager.hpp"
#include "GraphicsManager.hpp"
#include "Image.hpp"
#include "SceneObject.hpp"

namespace My {
class D3d12GraphicsManager : public GraphicsManager {
   public:
    int Initialize() final;
    void Finalize() final;

    void Draw() final;
    void Present() final;

    void SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState,
                          const Frame& frame) final;

    void DrawBatch(const Frame& frame) final;

    // skybox
    void DrawSkyBox() final;

   private:
    void EndScene() final;

    void BeginFrame(const Frame& frame) final;
    void EndFrame(const Frame& frame) final;

    void initializeGeometries(const Scene& scene) final;
    void initializeSkyBox(const Scene& scene) final;
    void initializeTerrain(const Scene& scene) final;

    void SetPerFrameConstants(const DrawFrameContext& context);
    void SetLightInfo(const LightInfo& lightInfo);

    HRESULT CreateDescriptorHeaps();
    HRESULT CreateRenderTarget();
    HRESULT CreateDepthStencil();
    HRESULT CreateGraphicsResources();

    uint32_t CreateSamplerBuffer();
    int32_t CreateTextureBuffer(SceneObjectTexture& texture);
    uint32_t CreateConstantBuffer();
    size_t CreateIndexBuffer(const SceneObjectIndexArray& index_array);
    size_t CreateVertexBuffer(const SceneObjectVertexArray& v_property_array);

    HRESULT CreateRootSignature();
    HRESULT WaitForPreviousFrame();
    HRESULT ResetCommandList();
    HRESULT CreatePSO(D3d12PipelineState& pipelineState);
    HRESULT CreateCommandList();
    HRESULT MsaaResolve();

   private:
    ID3D12Device* m_pDev =
        nullptr;  // the pointer to our Direct3D device interface
#if defined(_DEBUG)
    ID3D12Debug* m_pDebugController = nullptr;
    ID3D12DebugDevice* m_pDebugDev = nullptr;
#endif
    D3D12_VIEWPORT m_ViewPort;  // viewport structure
    D3D12_RECT m_ScissorRect;   // scissor rect structure
    IDXGISwapChain3* m_pSwapChain =
        nullptr;  // the pointer to the swap chain interface
    ID3D12Resource*
        m_pRenderTargets[GfxConfiguration::kMaxInFlightFrameCount *
                         2];  // the pointer to rendering buffer. [descriptor]
    ID3D12Resource*
        m_pDepthStencilBuffer;  // the pointer to the depth stencil buffer
    ID3D12CommandAllocator*
        m_pCommandAllocator[GfxConfiguration::kMaxInFlightFrameCount] = {
            nullptr};  // the pointer to command buffer allocator
    ID3D12GraphicsCommandList*
        m_pCommandList[GfxConfiguration::kMaxInFlightFrameCount] = {
            nullptr};  // a list to store GPU commands, which will be submitted
                       // to GPU to execute when done
    ID3D12CommandQueue* m_pCommandQueue =
        nullptr;  // the pointer to command queue
    ID3D12RootSignature* m_pRootSignature =
        nullptr;  // a graphics root signature defines what resources are bound
                  // to the pipeline
    ID3D12DescriptorHeap* m_pRtvHeap =
        nullptr;  // an array of descriptors of GPU objects
    ID3D12DescriptorHeap* m_pDsvHeap =
        nullptr;  // an array of descriptors of GPU objects
    ID3D12DescriptorHeap* m_pSamplerHeap =
        nullptr;  // an array of descriptors of GPU objects
    ID3D12DescriptorHeap* m_pCbvHeap = nullptr;  // main cbv descriptor table
    ID3D12DescriptorHeap* m_pSrvHeap = nullptr;  // main srv descriptor table
    ID3D12DescriptorHeap* m_pPerBatchSrvRingHeap =
        nullptr;  // per batch srv descriptor table
    uint32_t m_nPerBatchSrvRingHeapStart;
    uint32_t m_nPerBatchSrvRingHeapEnd;
    uint32_t m_nPerBatchSrvRingHeapSize;

    std::vector<ID3D12PipelineState*>
        m_pPipelineStates;  // an object maintains the state of all currently
                            // set shaders and certain fixed function state
                            // objects such as the input assembler, tesselator,
                            // rasterizer and output manager

    uint32_t m_nRtvDescriptorSize;
    uint32_t m_nCbvSrvUavDescriptorSize;
    uint32_t m_nSamplerDescriptorSize;

    std::vector<ID3D12Resource*>
        m_Buffers;  // the pointer to the GPU buffer other than texture
    std::vector<ID3D12Resource*>
        m_Textures;  // the pointer to the Texture buffer
    std::vector<D3D12_VERTEX_BUFFER_VIEW>
        m_VertexBufferView;  // vertex buffer descriptors
    std::vector<D3D12_INDEX_BUFFER_VIEW>
        m_IndexBufferView;  // index buffer descriptors

    struct D3dDrawBatchContext : public DrawBatchContext {
        uint32_t index_count;
        size_t index_offset;
        uint32_t property_count;
        size_t property_offset;
    };

    uint8_t* m_pPerFrameCbvDataBegin[GfxConfiguration::kMaxInFlightFrameCount] =
        {nullptr};
    ID3D12Resource* m_pPerFrameConstantUploadBuffer
        [GfxConfiguration::kMaxInFlightFrameCount] = {nullptr};

    uint8_t* m_pLightDataBegin[GfxConfiguration::kMaxInFlightFrameCount] = {
        nullptr};
    ID3D12Resource*
        m_pLightDataUploadBuffer[GfxConfiguration::kMaxInFlightFrameCount] = {
            nullptr};

    // Synchronization objects
    HANDLE m_hFenceEvent;
    ID3D12Fence* m_pFence[GfxConfiguration::kMaxInFlightFrameCount] = {nullptr};
    uint32_t m_nFenceValue[GfxConfiguration::kMaxInFlightFrameCount] = {0};
};
}  // namespace My

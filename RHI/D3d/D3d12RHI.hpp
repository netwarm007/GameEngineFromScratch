#pragma once
#include <array>
#include <functional>
#include <vector>

#include "config.h"

#include <DXGI1_4.h>
#include <d3d12.h>
#include <string_view>

#include "Buffer.hpp"
#include "GfxConfiguration.hpp"
#include "Image.hpp"
#include "geommath.hpp"

#include "RenderGraph/RenderPipeline/RenderPipeline.hpp"

namespace My {

class D3d12RHI {
   public:
    using QueryFrameBufferSizeFunc = std::function<void(uint32_t&, uint32_t&)>;
    using GetWindowHandlerFunc = std::function<HWND()>;
    using GetGfxConfigFunc = std::function<const GfxConfiguration()>;
    using DestroyResourceFunc = std::function<void()>;

    struct IndexBuffer {
        ID3D12Resource* buffer;
        D3D12_INDEX_BUFFER_VIEW descriptor;
        uint32_t indexCount;
    };

    struct VertexBuffer {
        ID3D12Resource* buffer;
        D3D12_VERTEX_BUFFER_VIEW descriptor;
    };

    struct ConstantBuffer {
        ID3D12Resource* buffer;
        size_t size;
    };

   public:
    D3d12RHI();
    ~D3d12RHI();

    static DXGI_FORMAT getDxgiFormat(const Image& img);
    static DXGI_FORMAT getDxgiFormat(const RenderGraph::TextureFormat::Enum &fmt);
    static D3D12_RASTERIZER_DESC getRasterizerDesc(const RenderGraph::RasterizerState &state);
    static D3D12_RENDER_TARGET_BLEND_DESC getRenderTargetBlendDesc(const RenderGraph::RenderTargetBlend &blend);
    static D3D12_DEPTH_STENCILOP_DESC getDepthStencilOpDesc(const RenderGraph::DepthStencilOperation &dsop);
    static D3D12_COMPARISON_FUNC getCompareFunc(const RenderGraph::ComparisonFunction::Enum &cmp);
    static D3D12_DEPTH_WRITE_MASK getDepthWriteMask(const RenderGraph::DepthWriteMask::Enum &mask);
    static D3D12_PRIMITIVE_TOPOLOGY_TYPE getTopologyType(const RenderGraph::TopologyType::Enum &topology);

   public:
    void SetFramebufferSizeQueryCB(const QueryFrameBufferSizeFunc& func) {
        m_fQueryFramebufferSize = func;
    }
    void SetGetWindowHandlerCB(const GetWindowHandlerFunc& func) {
        m_fGetWindowHandler = func;
    }
    void SetGetGfxConfigCB(const GetGfxConfigFunc& func) {
        m_fGetGfxConfigHandler = func;
    }
    void DestroyResourceCB(const DestroyResourceFunc& func) {
        m_fDestroyResourceHandler = func;
    }
    void CreateDevice();
    void EnableDebugLayer();
    void CreateCommandQueues();
    void CreateSwapChain();
    void CleanupSwapChain();
    void RecreateSwapChain();
    void CreateSyncObjects();

    void CreateRenderTargets();
    void CreateDepthStencils();
    void CreateFramebuffers();

    void CreateCommandPools();
    void CreateCommandLists();

    ID3D12Resource* CreateTextureImage(Image& img);
    void UpdateTexture(ID3D12Resource* texture, Image& img);

    ID3D12DescriptorHeap* CreateTextureSampler(uint32_t num_samplers);

    ID3D12Resource* CreateUniformBuffers(size_t bufferSize, std::wstring_view bufferName);

    ID3D12RootSignature* CreateRootSignature(
        const D3D12_SHADER_BYTECODE& shader);

    ID3D12PipelineState* CreateGraphicsPipeline(
        D3D12_GRAPHICS_PIPELINE_STATE_DESC& psod);

    ID3D12PipelineState* CreateComputePipeline(
        D3D12_COMPUTE_PIPELINE_STATE_DESC& psod);

    ID3D12DescriptorHeap* CreateDescriptorHeap(size_t num_descriptors,
                              std::wstring_view heap_group_name);

    void CreateDescriptorSet(ID3D12DescriptorHeap* pHeap,
                                    size_t offset,
                                    ConstantBuffer** pConstantBuffers,
                                    size_t constantBufferCount);

    void CreateDescriptorSet(ID3D12DescriptorHeap* pHeap, 
                                    size_t offset,
                                    ID3D12Resource** ppShaderResources,
                                    size_t shaderResourceCount);

    D3D12_CPU_DESCRIPTOR_HANDLE GetCpuDescriptorHandle(ID3D12DescriptorHeap* pHeap, size_t offset);
    D3D12_GPU_DESCRIPTOR_HANDLE GetGpuDescriptorHandle(ID3D12DescriptorHeap* pHeap, size_t offset);

    void ResetAllBuffers();
    void DestroyAll();

    void BeginFrame();
    void BeginPass(const Vector3f& clearColor);
    void SetPipelineState(ID3D12PipelineState* pPipelineState);
    void SetRootSignature(ID3D12RootSignature* pRootSignature);
    void Draw(const D3D12_VERTEX_BUFFER_VIEW& vertexBufferView,
              const D3D12_INDEX_BUFFER_VIEW& indexBufferView,
              ID3D12DescriptorHeap* pCbvSrvUavHeap,
              ID3D12DescriptorHeap* pSamplerHeap,
              D3D_PRIMITIVE_TOPOLOGY primitive_topology,
              uint32_t index_count_per_instance);
    void DrawGUI(ID3D12DescriptorHeap* pCbvSrvHeap);
    void EndPass();
    void EndFrame();

    void Present();

    IndexBuffer CreateIndexBuffer(const void* pData, size_t element_size,
                                  int32_t stride_size);
    VertexBuffer CreateVertexBuffer(const void* pData, size_t element_size,
                                    int32_t stride_size);

    void UpdateUniformBufer(ConstantBuffer* constantBuffers, const void* buffer);

    ID3D12Device* GetDevice() { return m_pDev; }

   private:
    void msaaResolve();
    void beginSingleTimeCommands();
    void endSingleTimeCommands();
    void waitOnFrame();
    void moveToNextFrame();

   private:
    DXGI_FORMAT m_eSurfaceFormat = DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;

    IDXGIFactory4* m_pFactory = nullptr;

    ID3D12Device4* m_pDev =
        nullptr;  // the pointer to our Direct3D device interface
#if defined(D3D12_RHI_DEBUG)
    ID3D12Debug* m_pDebugController = nullptr;
    ID3D12DebugDevice* m_pDebugDev = nullptr;
#endif
    D3D12_VIEWPORT m_ViewPort;  // viewport structure
    D3D12_RECT m_ScissorRect;   // scissor rect structure
    IDXGISwapChain3* m_pSwapChain =
        nullptr;  // the pointer to the swap chain interface
    std::vector<ID3D12Resource*>
        m_pRenderTargets;  // the pointer to rendering buffer. [descriptor]
    ID3D12Resource* m_pDepthStencilBuffer = nullptr;

    std::vector<ID3D12CommandAllocator*> m_pGraphicsCommandAllocators;

    ID3D12CommandAllocator* m_pComputeCommandAllocator =
        nullptr;  // the pointer to command
                  // buffer allocator
    ID3D12CommandAllocator* m_pCopyCommandAllocator =
        nullptr;  // the pointer to command
                  // buffer allocator
    std::vector<ID3D12GraphicsCommandList*> m_pGraphicsCommandLists;

    ID3D12GraphicsCommandList* m_pComputeCommandList = nullptr;
    ID3D12GraphicsCommandList* m_pCopyCommandList = nullptr;

    ID3D12CommandQueue*
        m_pGraphicsCommandQueue;                 // the pointer to command queue
    ID3D12CommandQueue* m_pComputeCommandQueue;  // the pointer to command queue
    ID3D12CommandQueue* m_pCopyCommandQueue;     // the pointer to command queue

    std::vector<ID3D12DescriptorHeap*> m_pRtvHeaps;

    ID3D12DescriptorHeap* m_pDsvHeap = nullptr;

    uint32_t m_nRtvDescriptorSize;
    uint32_t m_nCbvSrvUavDescriptorSize;
    uint32_t m_nSamplerDescriptorSize;

    std::vector<ID3D12Resource*> m_pRawBuffers;

    // Synchronization objects
    HANDLE m_hGraphicsFenceEvent = INVALID_HANDLE_VALUE;
    HANDLE m_hCopyFenceEvent = INVALID_HANDLE_VALUE;
    HANDLE m_hComputeFenceEvent = INVALID_HANDLE_VALUE;
    ID3D12Fence* m_pGraphicsFence = nullptr;
    std::array<uint64_t, GfxConfiguration::kMaxInFlightFrameCount>
        m_nGraphicsFenceValues;

    QueryFrameBufferSizeFunc m_fQueryFramebufferSize;
    GetWindowHandlerFunc m_fGetWindowHandler;
    GetGfxConfigFunc m_fGetGfxConfigHandler;
    DestroyResourceFunc m_fDestroyResourceHandler;

    uint32_t m_nCurrentFrame = 0;
    bool m_bInitialized = false;
};
}  // namespace My
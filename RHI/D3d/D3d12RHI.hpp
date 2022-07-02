#pragma once
#include <array>
#include <functional>
#include <vector>

#include "config.h"

#include <DXGI1_4.h>
#include <d3d12.h>

#include "Buffer.hpp"
#include "GfxConfiguration.hpp"
#include "Image.hpp"
#include "geommath.hpp"

namespace My {

class D3d12RHI {
   public:
    using QueryFrameBufferSizeFunc = std::function<void(int&, int&)>;
    using GetWindowHandlerFunc = std::function<HWND()>;
    using GetGfxConfigFunc = std::function<const GfxConfiguration&()>;
    using DestroyResourceFunc = std::function<void()>;

    struct Vertex {
        Vector3f pos;
        Vector3f color;
        Vector2f texCoord;
    };

    struct UniformBufferObject {
        Matrix4X4f model;
        Matrix4X4f view;
        Matrix4X4f proj;
    };

   public:
    D3d12RHI();
    ~D3d12RHI();

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

    void CreateTextureSampler();

    void CreateVertexBuffer();

    void CreateIndexBuffer();

    void CreateUniformBuffers();

    ID3D12RootSignature* CreateRootSignature(
        const D3D12_SHADER_BYTECODE& shader);

    ID3D12PipelineState* CreateGraphicsPipeline(
        D3D12_GRAPHICS_PIPELINE_STATE_DESC& psod);

    ID3D12PipelineState* CreateComputePipeline(
        D3D12_COMPUTE_PIPELINE_STATE_DESC& psod);

    void CreateDescriptorPool(size_t num_descriptors,
                              const wchar_t* heap_group_name, size_t num_heaps);

    void CreateDescriptorSets(ID3D12Resource** ppResources, size_t count);

    void ResetAllBuffers();
    void DestroyAll();

    void BeginFrame();
    void BeginPass(const Vector4f& clearColor);
    void SetPipelineState(ID3D12PipelineState* pPipelineState);
    void SetRootSignature(ID3D12RootSignature* pRootSignature);
    void Draw();
    void DrawGUI(ID3D12DescriptorHeap* pCbvSrvHeap);
    void EndPass();
    void EndFrame();

    void Present();

    void setModel(const std::vector<Vertex>& vertices,
                  const std::vector<uint32_t>& indices);

    size_t CreateIndexBuffer(const void* pData, size_t size,
                             int32_t index_size);
    size_t CreateVertexBuffer(const void* pData, size_t size, int32_t stride);

    ID3D12Device* GetDevice() { return m_pDev; }

   private:
    void msaaResolve();
    void beginSingleTimeCommands();
    void endSingleTimeCommands();
    void waitOnFrame();
    void moveToNextFrame();
    void updateUniformBufer();

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

    std::vector<ID3D12DescriptorHeap*> m_pCbvSrvUavHeaps;

    ID3D12DescriptorHeap* m_pSamplerHeap = nullptr;

    uint32_t m_nRtvDescriptorSize;
    uint32_t m_nCbvSrvUavDescriptorSize;
    uint32_t m_nSamplerDescriptorSize;

    std::vector<ID3D12Resource*> m_pRawBuffers;

    std::vector<ID3D12Resource*> m_pIndexBuffers;
    std::vector<D3D12_INDEX_BUFFER_VIEW> m_IndexBufferViews;

    std::vector<ID3D12Resource*> m_pVertexBuffers;
    std::vector<D3D12_VERTEX_BUFFER_VIEW> m_VertexBufferViews;

    std::vector<ID3D12Resource*> m_pUniformBuffers;

    std::vector<Vertex> m_Vertices;
    std::vector<uint32_t> m_Indices;

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
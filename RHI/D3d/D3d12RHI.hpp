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

#include "a2v.hpp"

namespace My {

class D3d12RHI {
    using QueryFrameBufferSizeFunc = std::function<void(int&, int&)>;
    using GetWindowHandlerFunc = std::function<HWND()>;
    using GetGfxConfigFunc = std::function<const GfxConfiguration&()>;
    using ResourceID = size_t;

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

    ResourceID CreateTextureImage(Image& img);

    void CreateTextureSampler();

    void CreateVertexBuffer();

    void CreateIndexBuffer();

    void CreateUniformBuffers();

    void CreateDescriptorSetLayout();

    void CreateGraphicsPipeline();

    void CreateDescriptorPool();

    void CreateDescriptorSets();

    void DestroyAll();

    void DrawFrame();
    void MsaaResolve();

    void setModel(const std::vector<Vertex>& vertices,
                  const std::vector<uint32_t>& indices);

    void setShaders(Buffer& vertShaderCode, Buffer& fragShaderCode);

   private:
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

    ID3D12CommandAllocator* m_pGraphicsCommandAllocator = nullptr;

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

    ID3D12RootSignature* m_pRootSignature;
    ID3D12PipelineState* m_pPipelineState;

    std::vector<ID3D12Resource*> m_pRawBuffers;

    std::vector<ID3D12Resource*> m_pTextureBuffers;

    std::vector<ID3D12Resource*> m_pIndexBuffers;

    std::vector<ID3D12Resource*> m_pVertexBuffers;

    std::vector<ID3D12Resource*> m_pUniformBuffers;

    std::vector<Vertex> m_Vertices;
    std::vector<uint32_t> m_Indices;

    D3D12_SHADER_BYTECODE m_VertexShaderModule;
    D3D12_SHADER_BYTECODE m_PixelShaderModule;

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

    uint32_t m_nCurrentFrame = 0;
    bool m_bInitialized = false;
};
}  // namespace My
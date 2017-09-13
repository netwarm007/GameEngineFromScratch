#pragma once
#include <stdint.h>
#include <d3d12.h>
#include <DXGI1_4.h>
#include "GraphicsManager.hpp"

namespace My {
    class D3d12GraphicsManager : public GraphicsManager
    {
    public:
       	virtual int Initialize();
	    virtual void Finalize();

	    virtual void Tick();

    private:
        HRESULT CreateRenderTarget();
        HRESULT CreateGraphicsResources();

    private:
        static const uint32_t           kFrameCount  = 2;
        ID3D12Device*                   m_pDev       = nullptr;             // the pointer to our Direct3D device interface
        D3D12_VIEWPORT                  m_ViewPort;                         // viewport structure
        D3D12_RECT                      m_ScissorRect;                      // scissor rect structure
        IDXGISwapChain3*                m_pSwapChain = nullptr;             // the pointer to the swap chain interface
        ID3D12Resource*                 m_pRenderTargets[kFrameCount];      // the pointer to rendering buffer. [descriptor]
        ID3D12CommandAllocator*         m_pCommandAllocator = nullptr;      // the pointer to command buffer allocator
        ID3D12CommandQueue*             m_pCommandQueue = nullptr;          // the pointer to command queue
        ID3D12RootSignature*            m_pRootSignature = nullptr;         // a graphics root signature defines what resources are bound to the pipeline
        ID3D12DescriptorHeap*           m_pRtvHeap = nullptr;               // an array of descriptors of GPU objects
        ID3D12PipelineState*            m_pPipelineState = nullptr;         // an object maintains the state of all currently set shaders
                                                                            // and certain fixed function state objects
                                                                            // such as the input assembler, tesselator, rasterizer and output manager
        ID3D12GraphicsCommandList*      m_pCommandList = nullptr;           // a list to store GPU commands, which will be submitted to GPU to execute when done

        uint32_t                        m_nRtvDescriptorSize;

        ID3D12Resource*                 m_pVertexBuffer = nullptr;          // the pointer to the vertex buffer
        D3D12_VERTEX_BUFFER_VIEW        m_VertexBufferView;                 // a view of the vertex buffer

        // Synchronization objects
        uint32_t                        m_nFrameIndex;
        HANDLE                          m_hFenceEvent;
        ID3D12Fence*                    m_pFence = nullptr;
        uint32_t                        m_nFenceValue;
    };
}

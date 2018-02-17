#pragma once
#include <stdint.h>
#include <d3d12.h>
#include <DXGI1_4.h>
#include <vector>
#include "GraphicsManager.hpp"
#include "Buffer.hpp"
#include "Image.hpp"
#include "SceneObject.hpp"

namespace My {
    class D3d12GraphicsManager : public GraphicsManager
    {
    public:
       	virtual int Initialize();
	    virtual void Finalize();

        virtual void Clear();

        virtual void Draw();

    protected:
        bool SetPerFrameShaderParameters();
        bool SetPerBatchShaderParameters(int32_t index);

        void InitializeBuffers(const Scene& scene);
        void ClearBuffers();
        bool InitializeShaders();
        void RenderBuffers();

    private:
        HRESULT CreateDescriptorHeaps();
        HRESULT CreateRenderTarget();
        HRESULT CreateDepthStencil();
        HRESULT CreateGraphicsResources();
        HRESULT CreateSamplerBuffer();
        HRESULT CreateTextureBuffer();
        HRESULT CreateConstantBuffer();
        HRESULT CreateIndexBuffer(const SceneObjectIndexArray& index_array);
        HRESULT CreateVertexBuffer(const SceneObjectVertexArray& v_property_array);
        HRESULT CreateRootSignature();
        HRESULT WaitForPreviousFrame();
        HRESULT PopulateCommandList();

    private:
        static const uint32_t           kFrameCount  = 2;
        static const uint32_t           kMaxSceneObjectCount  = 65535;
        static const uint32_t           kMaxTextureCount  = 2048;
		static const uint32_t		    kTextureDescStartIndex = kFrameCount * (1 + kMaxSceneObjectCount);

        ID3D12Device*                   m_pDev       = nullptr;             // the pointer to our Direct3D device interface
        D3D12_VIEWPORT                  m_ViewPort;                         // viewport structure
        D3D12_RECT                      m_ScissorRect;                      // scissor rect structure
        IDXGISwapChain3*                m_pSwapChain = nullptr;             // the pointer to the swap chain interface
        ID3D12Resource*                 m_pRenderTargets[kFrameCount];      // the pointer to rendering buffer. [descriptor]
        ID3D12Resource*                 m_pDepthStencilBuffer;              // the pointer to the depth stencil buffer
        ID3D12CommandAllocator*         m_pCommandAllocator = nullptr;      // the pointer to command buffer allocator
        ID3D12CommandQueue*             m_pCommandQueue = nullptr;          // the pointer to command queue
        ID3D12RootSignature*            m_pRootSignature = nullptr;         // a graphics root signature defines what resources are bound to the pipeline
        ID3D12DescriptorHeap*           m_pRtvHeap = nullptr;               // an array of descriptors of GPU objects
        ID3D12DescriptorHeap*           m_pDsvHeap = nullptr;               // an array of descriptors of GPU objects
		ID3D12DescriptorHeap*           m_pCbvHeap = nullptr;               // an array of descriptors of GPU objects
        ID3D12DescriptorHeap*           m_pSamplerHeap = nullptr;           // an array of descriptors of GPU objects
        ID3D12PipelineState*            m_pPipelineState = nullptr;         // an object maintains the state of all currently set shaders
                                                                            // and certain fixed function state objects
                                                                            // such as the input assembler, tesselator, rasterizer and output manager
        ID3D12GraphicsCommandList*      m_pCommandList = nullptr;           // a list to store GPU commands, which will be submitted to GPU to execute when done

        uint32_t                        m_nRtvDescriptorSize;
        uint32_t                        m_nCbvSrvDescriptorSize;

        std::vector<ID3D12Resource*>    m_Buffers;                          // the pointer to the vertex buffer
        std::vector<D3D12_VERTEX_BUFFER_VIEW>       m_VertexBufferView;                 // a view of the vertex buffer
        std::vector<D3D12_INDEX_BUFFER_VIEW>        m_IndexBufferView;                  // a view of the vertex buffer
        ID3D12Resource*                 m_pTextureBuffer = nullptr;         // the pointer to the texture buffer

        struct DrawBatchContext {
            int32_t count;
            std::shared_ptr<Matrix4X4f> transform;
            std::shared_ptr<SceneObjectMaterial> material;
        };

        std::vector<DrawBatchContext> m_DrawBatchContext;

        uint8_t*                        m_pCbvDataBegin = nullptr;
		static const size_t				kSizePerFrameConstantBuffer = (sizeof(DrawFrameContext) + 255) & 256; // CB size is required to be 256-byte aligned.
		static const size_t				kSizePerBatchConstantBuffer = (sizeof(DrawBatchContext) + 255) & 256; // CB size is required to be 256-byte aligned.
		static const size_t				kSizeConstantBufferPerFrame = kSizePerFrameConstantBuffer + kSizePerBatchConstantBuffer * kMaxSceneObjectCount;

        // Synchronization objects
        uint32_t                        m_nFrameIndex;
        HANDLE                          m_hFenceEvent;
        ID3D12Fence*                    m_pFence = nullptr;
        uint32_t                        m_nFenceValue;
    };
}

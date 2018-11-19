#pragma once
#include <stdint.h>
#include <d3d12.h>
#include <DXGI1_4.h>
#include <vector>
#include <map>
#include "GraphicsManager.hpp"
#include "Buffer.hpp"
#include "Image.hpp"
#include "SceneObject.hpp"

namespace My {
    class D3d12GraphicsManager : public GraphicsManager
    {
    public:
       	int Initialize() final;
	    void Finalize() final;

        void Clear() final;
        void Draw() final;
        void Present() final;

        void UseShaderProgram(const intptr_t shaderProgram) final;
        void SetPerFrameConstants(const DrawFrameContext& context) final;
        void SetPerBatchConstants(const DrawBatchContext& context) final;

        void DrawBatch(const DrawBatchContext& context) final;

    private:
        void BeginScene(const Scene& scene) final;
        void EndScene() final;

        void BeginFrame() final;
        void EndFrame() final;

        void RenderBuffers();

        HRESULT CreateDescriptorHeaps();
        HRESULT CreateRenderTarget();
        HRESULT CreateDepthStencil();
        HRESULT CreateGraphicsResources();

        uint32_t CreateSamplerBuffer();
        uint32_t CreateTextureBuffer(SceneObjectTexture& texture);
        uint32_t CreateConstantBuffer();
        size_t CreateIndexBuffer(const SceneObjectIndexArray& index_array);
        size_t CreateVertexBuffer(const SceneObjectVertexArray& v_property_array);

        void RegisterMsaaRtAsTexture();

        HRESULT CreateRootSignature();
        HRESULT WaitForPreviousFrame();
        HRESULT ResetCommandList();
        HRESULT InitializePSO();
        HRESULT CreateCommandList();
        HRESULT MsaaResolve();

    private:
        static const uint32_t           m_kFrameCount  = 2;
        static const uint32_t           m_kMaxTextureCount  = 2048;
        static const uint32_t           m_kMaxLightCount = 100;
        static const uint32_t           m_kTextureDescOffset = 2 * m_kFrameCount;

        ID3D12Device*                   m_pDev       = nullptr;             // the pointer to our Direct3D device interface
        D3D12_VIEWPORT                  m_ViewPort;                         // viewport structure
        D3D12_RECT                      m_ScissorRect;                      // scissor rect structure
        IDXGISwapChain3*                m_pSwapChain = nullptr;             // the pointer to the swap chain interface
        ID3D12Resource*                 m_pRenderTargets[m_kFrameCount];    // the pointer to rendering buffer. [descriptor]
        ID3D12Resource*                 m_pDepthStencilBuffer;              // the pointer to the depth stencil buffer
        ID3D12Resource*                 m_pMsaaRenderTarget;                // the pointer to the MSAA rendering target
        ID3D12CommandAllocator*         m_pCommandAllocator = nullptr;      // the pointer to command buffer allocator
        ID3D12CommandQueue*             m_pCommandQueue = nullptr;          // the pointer to command queue
        ID3D12RootSignature*            m_pRootSignature = nullptr;         // a graphics root signature defines what resources are bound to the pipeline
        ID3D12RootSignature*            m_pRootSignatureResolve = nullptr;  // a graphics root signature defines what resources are bound to the pipeline
        ID3D12DescriptorHeap*           m_pRtvHeap = nullptr;               // an array of descriptors of GPU objects
        ID3D12DescriptorHeap*           m_pDsvHeap = nullptr;               // an array of descriptors of GPU objects
		ID3D12DescriptorHeap*           m_pCbvSrvUavHeap = nullptr;               // an array of descriptors of GPU objects
        ID3D12DescriptorHeap*           m_pSamplerHeap = nullptr;           // an array of descriptors of GPU objects
        ID3D12PipelineState*            m_pPipelineState = nullptr;         // an object maintains the state of all currently set shaders
                                                                            // and certain fixed function state objects
                                                                            // such as the input assembler, tesselator, rasterizer and output manager
        ID3D12PipelineState*            m_pPipelineStateResolve = nullptr;
        ID3D12GraphicsCommandList*      m_pCommandList = nullptr;           // a list to store GPU commands, which will be submitted to GPU to execute when done

        uint32_t                        m_nRtvDescriptorSize;
        uint32_t                        m_nCbvSrvDescriptorSize;
        uint32_t                        m_nSamplerDescriptorSize;

        std::vector<ID3D12Resource*>    m_Buffers;                          // the pointer to the GPU buffer other than texture
        std::vector<ID3D12Resource*>    m_Textures;                         // the pointer to the Texture buffer
        std::map<std::string, uint32_t>   m_TextureIndex;                   // the LUT of texture name -> index
        std::vector<D3D12_VERTEX_BUFFER_VIEW>       m_VertexBufferView;     // vertex buffer descriptors
        std::vector<D3D12_INDEX_BUFFER_VIEW>        m_IndexBufferView;      // index buffer descriptors
        std::vector<D3D12_CONSTANT_BUFFER_VIEW_DESC> m_ConstantBufferView;  // constant buffer descriptors

        struct D3dDrawBatchContext : public DrawBatchContext {
            uint32_t index_count;
            size_t   index_offset;
            uint32_t property_count;
            size_t   property_offset;
        };

        uint8_t*                        m_pCbvDataBegin = nullptr;
		size_t				            m_kSizePerFrameConstantBuffer;
		size_t				            m_kSizePerBatchConstantBuffer;
		size_t				            m_kSizeConstantBufferPerFrame;

        // Synchronization objects
        HANDLE                          m_hFenceEvent;
        ID3D12Fence*                    m_pFence = nullptr;
        uint32_t                        m_nFenceValue;
    };
}

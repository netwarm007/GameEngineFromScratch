#include <objbase.h>
#include <d3dcompiler.h>
#include "D3d12GraphicsManager.hpp"
#include "WindowsApplication.hpp"

using namespace My;


namespace My {
    extern IApplication* g_pApp;

	template<class T>
	inline void SafeRelease(T **ppInterfaceToRelease)
	{
		if (*ppInterfaceToRelease != nullptr)
		{
			(*ppInterfaceToRelease)->Release();

			(*ppInterfaceToRelease) = nullptr;
		}
	}

    static void GetHardwareAdapter(IDXGIFactory4* pFactory, IDXGIAdapter1** ppAdapter)
    {
        IDXGIAdapter1* pAdapter = nullptr;
        *ppAdapter = nullptr;

        for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &pAdapter); adapterIndex++)
        {
           DXGI_ADAPTER_DESC1 desc;
           pAdapter->GetDesc1(&desc);

           if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
           {
               // Don't select the Basic Render Driver adapter.
               continue;
           }

           // Check to see if the adapter supports Direct3D 12, but don't create the
           // actual device yet.
           if (SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)))
           {
               break;
           }
        }

        *ppAdapter = pAdapter;
    }
}

// this is the function that loads and prepares the shaders
HRESULT My::D3d12GraphicsManager::InitPipeline() 
{
    HRESULT hr;
    if (FAILED(hr = m_pDev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_pCommandAllocator)))) {
        return hr;
    }

    // create an empty root signature
    D3D12_ROOT_SIGNATURE_DESC rsd;
    rsd.NumParameters = 0;
    rsd.pParameters   = nullptr;
    rsd.NumStaticSamplers = 0;
    rsd.pStaticSamplers   = nullptr;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ID3DBlob* signature;
    ID3DBlob* error;
    if (FAILED(hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error))) {
        return hr;
    }
    if (FAILED(hr = m_pDev->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_pRootSignature)))) {
        return hr;
    }

    // load the shaders
#if defined(_DEBUG)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif
    ID3DBlob* vertexShader;
    ID3DBlob* pixelShader;

    hr = D3DCompileFromFile(
        L"simple.hlsl",
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "VSMain",
        "vs_5_0",
        compileFlags,
        0,
        &vertexShader,
        &error);
    if (error) { OutputDebugString((LPCTSTR)error->GetBufferPointer()); error->Release(); return hr; }

    hr = D3DCompileFromFile(
        L"simple.hlsl",
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "PSMain",
        "ps_5_0",
        compileFlags,
        0,
        &pixelShader,
        &error);
    if (error) { OutputDebugString((LPCTSTR)error->GetBufferPointer()); error->Release(); return hr; }
 
    // create the input layout object
    D3D12_INPUT_ELEMENT_DESC ied[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    // describe and create the graphics pipeline state object (PSO)
    D3D12_RASTERIZER_DESC rd;
    rd.FillMode = D3D12_FILL_MODE_SOLID;
    rd.CullMode = D3D12_CULL_MODE_BACK;
    rd.FrontCounterClockwise = FALSE;
    rd.DepthBias = 0;
    rd.DepthBiasClamp = 0.0f;
    rd.SlopeScaledDepthBias = 0.0f;
    rd.DepthClipEnable = TRUE;
    rd.MultisampleEnable = FALSE;
    rd.AntialiasedLineEnable = FALSE;
    rd.ForcedSampleCount = 0;
    rd.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    D3D12_BLEND_DESC bd;
    bd.AlphaToCoverageEnable = FALSE;
    bd.IndependentBlendEnable = FALSE;
    bd.RenderTarget[0].BlendEnable = FALSE;
    bd.RenderTarget[0].LogicOpEnable = FALSE;
    bd.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    bd.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO;
    bd.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    bd.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    bd.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    bd.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    bd.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_NOOP;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psod = {};
    psod.InputLayout    = { ied, _countof(ied) };
    psod.pRootSignature = m_pRootSignature;
    psod.VS             = { reinterpret_cast<UINT8*>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize() };
    psod.PS             = { reinterpret_cast<UINT8*>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize() };
    psod.RasterizerState= rd;
    psod.BlendState     = bd; 
    psod.DepthStencilState.DepthEnable  = FALSE;
    psod.DepthStencilState.StencilEnable= FALSE;
    psod.SampleMask     = UINT_MAX;
    psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psod.NumRenderTargets = 1;
    psod.RTVFormats[0]  = DXGI_FORMAT_R8G8B8A8_UNORM;
    psod.SampleDesc.Count = 1;
    if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(&psod, IID_PPV_ARGS(&m_pPipelineState)))) {
        return hr;
    }

    if (FAILED(hr = m_pDev->CreateCommandList(0, 
                D3D12_COMMAND_LIST_TYPE_DIRECT, 
                m_pCommandAllocator,
                m_pPipelineState, 
                IID_PPV_ARGS(&m_pCommandList)))) {
        return hr;
    }

    hr = m_pCommandList->Close();

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateRenderTarget() 
{
    HRESULT hr;

    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = kFrameCount;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if(FAILED(hr = m_pDev->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_pRtvHeap)))) {
        return hr;
    }

    m_nRtvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart();

    // Create a RTV for each frame.
    for (uint32_t i = 0; i < kFrameCount; i++)
    {
        if (FAILED(hr = m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_pRenderTargets[i])))) {
            break;
        }
        m_pDev->CreateRenderTargetView(m_pRenderTargets[i], nullptr, rtvHandle);
        rtvHandle.ptr += m_nRtvDescriptorSize;
    }

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateGraphicsResources()
{
    HRESULT hr;

#if defined(_DEBUG)
	// Enable the D3D12 debug layer.
	{
		ID3D12Debug* pDebugController;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDebugController))))
		{
			pDebugController->EnableDebugLayer();
		}
		SafeRelease(&pDebugController);
	}
#endif

	IDXGIFactory4* pFactory;
	if (FAILED(hr = CreateDXGIFactory1(IID_PPV_ARGS(&pFactory)))) {
		return hr;
	}

	IDXGIAdapter1* pHardwareAdapter;
	GetHardwareAdapter(pFactory, &pHardwareAdapter);

	if (FAILED(D3D12CreateDevice(pHardwareAdapter,
		D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_pDev)))) {

		IDXGIAdapter* pWarpAdapter;
		if (FAILED(hr = pFactory->EnumWarpAdapter(IID_PPV_ARGS(&pWarpAdapter)))) {
	        SafeRelease(&pFactory);
            return hr;
        }

        if(FAILED(hr = D3D12CreateDevice(pWarpAdapter, D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&m_pDev)))) {
            SafeRelease(&pFactory);
            return hr;
        }
	}


    HWND hWnd = reinterpret_cast<WindowsApplication*>(g_pApp)->GetMainWindow();

    // Describe and create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;

    if(FAILED(hr = m_pDev->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_pCommandQueue)))) {
        SafeRelease(&pFactory);
        return hr;
    }

    // create a struct to hold information about the swap chain
    DXGI_SWAP_CHAIN_DESC1 scd;

    // clear out the struct for use
    ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC1));

    // fill the swap chain description struct
    scd.Width  = g_pApp->GetConfiguration().screenWidth;
    scd.Height = g_pApp->GetConfiguration().screenHeight;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;     	        // use 32-bit color
    scd.Stereo = FALSE;
    scd.SampleDesc.Count = 1;                               // multi-samples can not be used when in SwapEffect sets to
                                                            // DXGI_SWAP_EFFECT_FLOP_DISCARD
    scd.SampleDesc.Quality = 0;                             // multi-samples can not be used when in SwapEffect sets to
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;      // how swap chain is to be used
    scd.BufferCount = kFrameCount;                          // back buffer count
    scd.Scaling     = DXGI_SCALING_STRETCH;
    scd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;        // DXGI_SWAP_EFFECT_FLIP_DISCARD only supported after Win10
                                                            // use DXGI_SWAP_EFFECT_DISCARD on platforms early than Win10
    scd.AlphaMode   = DXGI_ALPHA_MODE_UNSPECIFIED;
    scd.Flags    = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;  // allow full-screen transition

    IDXGISwapChain1* pSwapChain;
    if (FAILED(hr = pFactory->CreateSwapChainForHwnd(
                m_pCommandQueue,                            // Swap chain needs the queue so that it can force a flush on it
                hWnd,
                &scd,
                NULL,
                NULL,
                &pSwapChain
                )))
    {
        SafeRelease(&pFactory);
        return hr;
    }

    m_pSwapChain = reinterpret_cast<IDXGISwapChain3*>(pSwapChain);

    m_nFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();
    hr = CreateRenderTarget();
    InitPipeline();

    return hr;
}

int  My::D3d12GraphicsManager::Initialize()
{
    int result = 0;

    result = static_cast<int>(CreateGraphicsResources());

    return result;
}

void My::D3d12GraphicsManager::Tick()
{
}

void My::D3d12GraphicsManager::Finalize()
{
    SafeRelease(&m_pFence);
    SafeRelease(&m_pVertexBuffer);
    SafeRelease(&m_pCommandList);
    SafeRelease(&m_pPipelineState);
    SafeRelease(&m_pRtvHeap);
    SafeRelease(&m_pRootSignature);
    SafeRelease(&m_pCommandQueue);
    SafeRelease(&m_pCommandAllocator);
    for (uint32_t i = 0; i < kFrameCount; i++) {
	    SafeRelease(&m_pRenderTargets[kFrameCount]);
    }
	SafeRelease(&m_pSwapChain);
	SafeRelease(&m_pDev);
}


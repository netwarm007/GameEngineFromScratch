#include <iostream>
#include <objbase.h>
#include "D3d12GraphicsManager.hpp"
#include "WindowsApplication.hpp"
#include "SceneManager.hpp"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;


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

HRESULT My::D3d12GraphicsManager::CreateDescriptorHeaps() 
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

    // Describe and create a depth stencil view (DSV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if(FAILED(hr = m_pDev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_pDsvHeap)))) {
        return hr;
    }

    // Describe and create a Shader Resource View (SRV) and 
    // Constant Buffer View (CBV) and 
    // Unordered Access View (UAV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc = {};
    cbvSrvUavHeapDesc.NumDescriptors =
        kFrameCount                                         // FrameCount Cbvs.
        + 100;                                              // + 100 for the SRV(Texture).
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if(FAILED(hr = m_pDev->CreateDescriptorHeap(&cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_pCbvSrvUavHeap)))) {
        return hr;
    }

    m_nCbvSrvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Describe and create a sampler descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
    samplerHeapDesc.NumDescriptors = 2048; // this is the max D3d12 HW support currently
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if(FAILED(hr = m_pDev->CreateDescriptorHeap(&samplerHeapDesc, IID_PPV_ARGS(&m_pSamplerHeap)))) {
        return hr;
    }

    if(FAILED(hr = m_pDev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_pCommandAllocator)))) {
        return hr;
    }

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateRenderTarget() 
{
    HRESULT hr;

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

HRESULT My::D3d12GraphicsManager::CreateDepthStencil()
{
    HRESULT hr;

    // Describe and create a depth stencil view (DSV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if(FAILED(hr = m_pDev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_pDsvHeap)))) {
        return hr;
    }

    // Create the depth stencil view.
    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

    D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
    depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
    depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
    depthOptimizedClearValue.DepthStencil.Stencil = 0;

    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    uint32_t width = g_pApp->GetConfiguration().screenWidth;
    uint32_t height = g_pApp->GetConfiguration().screenHeight;
    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 0;
    resourceDesc.Format = DXGI_FORMAT_D32_FLOAT;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &depthOptimizedClearValue,
        IID_PPV_ARGS(&m_pDepthStencilBuffer)
        ))) {
        return hr;
    }

    m_pDev->CreateDepthStencilView(m_pDepthStencilBuffer, &depthStencilDesc, m_pDsvHeap->GetCPUDescriptorHandleForHeapStart());

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateVertexBuffer(const Buffer& buffer)
{
    HRESULT hr;

    // create vertex GPU heap 
    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = buffer.GetDataSize();
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&m_pVertexBuffer));

    return hr;
}


HRESULT My::D3d12GraphicsManager::CreateIndexBuffer(const Buffer& buffer)
{
    HRESULT hr;

    // create index GPU heap
    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = buffer.GetDataSize();
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&m_pIndexBuffer));

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateTextureBuffer(const Image& image)
{
    HRESULT hr;

    // Describe and create a Texture2D.
    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.Width = image.Width;
    textureDesc.Height = image.Height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&m_pTextureBuffer));

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateSamplerBuffer()
{
    // Describe and create a sampler.
    D3D12_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
    samplerDesc.MipLODBias = 0.0f;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    m_pDev->CreateSampler(&samplerDesc, m_pSamplerHeap->GetCPUDescriptorHandleForHeapStart());

    return S_OK;
}

HRESULT My::D3d12GraphicsManager::CreateConstantBuffer(const Buffer& buffer)
{

    return S_OK;
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
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;              // use 32-bit color
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

    cout << "Creating Descriptor Heaps ...";
    if (FAILED(hr = CreateDescriptorHeaps())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating Render Targets ...";
    if (FAILED(hr = CreateRenderTarget())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating Depth Stencil Buffers ...";
    if (FAILED(hr = CreateDepthStencil())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating Root Signatures ...";
    if (FAILED(hr = CreateRootSignature())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Loading Shaders ...";
    if (FAILED(hr = InitializeShader("Shaders/simple.hlsl.vs", "Shaders/simple.hlsl.ps"))) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Initialize Buffers ...";
    if (FAILED(hr = InitializeBuffers())) {
        return hr;
    }
    cout << "Done!" << endl;

    return hr;
}

HRESULT My::D3d12GraphicsManager::CreateRootSignature()
{
    HRESULT hr = S_OK;

    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

    // This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

    if (FAILED(m_pDev->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
    {
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    D3D12_DESCRIPTOR_RANGE1 ranges[3] = {
        { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC },
        { D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 },
        { D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 6, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC }
    };

    D3D12_ROOT_PARAMETER1 rootParameters[3] = {
        { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[0] }, D3D12_SHADER_VISIBILITY_PIXEL },
        { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[1] }, D3D12_SHADER_VISIBILITY_PIXEL },
        { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[2] }, D3D12_SHADER_VISIBILITY_ALL },
    };

    // Allow input layout and deny uneccessary access to certain pipeline stages.
    D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

    D3D12_ROOT_SIGNATURE_DESC1 rootSignatureDesc = {
            _countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags
        };

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedRootSignatureDesc = {
        D3D_ROOT_SIGNATURE_VERSION_1_1,
    };

    versionedRootSignatureDesc.Desc_1_1 = rootSignatureDesc;

    ID3DBlob* signature = nullptr;
    ID3DBlob* error = nullptr;
    if (SUCCEEDED(hr = D3D12SerializeVersionedRootSignature(&versionedRootSignatureDesc, &signature, &error)))
    {
        hr = m_pDev->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_pRootSignature));
    }

    if (signature) signature->Release();
    if (error) error->Release();

    return hr;
}


// this is the function that loads and prepares the shaders
HRESULT My::D3d12GraphicsManager::InitializeShader(const char* vsFilename, const char* fsFilename) {
    HRESULT hr = S_OK;

    // load the shaders
#if defined(_DEBUG_SHADER)
    // Enable better shader debugging with the graphics debugging tools.
    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    UINT compileFlags = 0;
#endif
    Buffer vertexShader = g_pAssetLoader->SyncOpenAndReadBinary(vsFilename);
    Buffer pixelShader = g_pAssetLoader->SyncOpenAndReadBinary(fsFilename);

    D3D12_SHADER_BYTECODE vertexShaderByteCode;
    vertexShaderByteCode.pShaderBytecode = vertexShader.GetData();
    vertexShaderByteCode.BytecodeLength = vertexShader.GetDataSize();

    D3D12_SHADER_BYTECODE pixelShaderByteCode;
    pixelShaderByteCode.pShaderBytecode = pixelShader.GetData();
    pixelShaderByteCode.BytecodeLength = pixelShader.GetDataSize();

    // create the input layout object
    D3D12_INPUT_ELEMENT_DESC ied[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_RASTERIZER_DESC rsd = { D3D12_FILL_MODE_SOLID, D3D12_CULL_MODE_BACK, FALSE, D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                                  TRUE, FALSE, FALSE, 0, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF };
    const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlend = { FALSE, FALSE,
        D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
        D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
        D3D12_LOGIC_OP_NOOP,
        D3D12_COLOR_WRITE_ENABLE_ALL
    };

    D3D12_BLEND_DESC bld = { FALSE, FALSE,
                                             {
                                               defaultRenderTargetBlend,
                                               defaultRenderTargetBlend,
                                               defaultRenderTargetBlend,
                                               defaultRenderTargetBlend,
                                               defaultRenderTargetBlend,
                                               defaultRenderTargetBlend,
                                               defaultRenderTargetBlend,
                                             }
                                    };

    const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp = { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS };
    D3D12_DEPTH_STENCIL_DESC dsd = { TRUE, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS, FALSE, D3D12_DEFAULT_STENCIL_READ_MASK, D3D12_DEFAULT_STENCIL_WRITE_MASK, 
        defaultStencilOp, defaultStencilOp };

    // describe and create the graphics pipeline state object (PSO)
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psod = {};
    psod.InputLayout    = { ied, _countof(ied) };
    psod.pRootSignature = m_pRootSignature;
    psod.VS             = vertexShaderByteCode;
    psod.PS             = pixelShaderByteCode;
    psod.RasterizerState= rsd;
    psod.BlendState     = bld;
    psod.DepthStencilState = dsd;
    psod.SampleMask     = UINT_MAX;
    psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psod.NumRenderTargets = 1;
    psod.RTVFormats[0]  = DXGI_FORMAT_R8G8B8A8_UNORM;
    psod.SampleDesc.Count = 1;

    if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(&psod, IID_PPV_ARGS(&m_pPipelineState))))
    {
        return hr;
    }

    hr = m_pDev->CreateCommandList(0, 
                D3D12_COMMAND_LIST_TYPE_DIRECT, 
                m_pCommandAllocator, 
                m_pPipelineState, 
                IID_PPV_ARGS(&m_pCommandList));

    return hr;
}

HRESULT My::D3d12GraphicsManager::InitializeBuffers()
{
    HRESULT hr = S_OK;

    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pGeometryNode = scene.GetFirstGeometryNode();
    while(pGeometryNode)
    {
        if (pGeometryNode->Visible())
        {
            auto pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            auto pMesh = pGeometry->GetMesh().lock();
            if(!pMesh) continue;
            
            // Set the number of vertex properties.
            auto vertexPropertiesCount = pMesh->GetVertexPropertiesCount();
            
            // Set the number of vertices in the vertex array.
            auto vertexCount = pMesh->GetVertexCount();

            Buffer buff;

            for (decltype(vertexPropertiesCount) i = 0; i < vertexPropertiesCount; i++)
            {
                CreateVertexBuffer(buff);
            }

            auto indexGroupCount = pMesh->GetIndexGroupCount();

            for (decltype(indexGroupCount) i = 0; i < indexGroupCount; i++)
            {
                CreateIndexBuffer(buff);
            }

            int textureCount = 0;
            Image image;

            for (decltype(textureCount) i = 0; i < textureCount; i++)
            {
                CreateTextureBuffer(image);
                CreateSamplerBuffer();
            }

            CreateConstantBuffer(buff);
        }

        pGeometryNode = scene.GetNextGeometryNode();
    }

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

void My::D3d12GraphicsManager::Clear()
{
}

void My::D3d12GraphicsManager::Draw()
{
}


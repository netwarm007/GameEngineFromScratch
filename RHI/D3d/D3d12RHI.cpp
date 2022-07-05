#include <chrono>
#include <string>

#include "D3d/D3d12Utility.hpp"
#include "D3d12RHI.hpp"
#include "portable.hpp"

using namespace My;

static DXGI_FORMAT getDxgiFormat(const Image& img) {
    DXGI_FORMAT format;

    if (img.compressed) {
        switch (img.compress_format) {
            case COMPRESSED_FORMAT::BC1:
            case COMPRESSED_FORMAT::DXT1:
                format = ::DXGI_FORMAT_BC1_UNORM;
                break;
            case COMPRESSED_FORMAT::BC2:
            case COMPRESSED_FORMAT::DXT3:
                format = ::DXGI_FORMAT_BC2_UNORM;
                break;
            case COMPRESSED_FORMAT::BC3:
            case COMPRESSED_FORMAT::DXT5:
                format = ::DXGI_FORMAT_BC3_UNORM;
                break;
            case COMPRESSED_FORMAT::BC4:
                format = ::DXGI_FORMAT_BC4_UNORM;
                break;
            case COMPRESSED_FORMAT::BC5:
                format = ::DXGI_FORMAT_BC5_UNORM;
                break;
            case COMPRESSED_FORMAT::BC6H:
                format = ::DXGI_FORMAT_BC6H_UF16;
                break;
            case COMPRESSED_FORMAT::BC7:
                format = ::DXGI_FORMAT_BC7_UNORM;
                break;
            default:
                assert(0);
        }
    } else {
        switch (img.pixel_format) {
            case PIXEL_FORMAT::R8:
                format = ::DXGI_FORMAT_R8_UNORM;
                break;
            case PIXEL_FORMAT::RG8:
                format = ::DXGI_FORMAT_R8G8_UNORM;
                break;
            case PIXEL_FORMAT::RGBA8:
                format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
                break;
            case PIXEL_FORMAT::RGBA16:
                format = ::DXGI_FORMAT_R16G16B16A16_FLOAT;
                break;
            default:
                assert(0);
        }
    }

    return format;
}

D3d12RHI::D3d12RHI() {}
D3d12RHI::~D3d12RHI() {}

void D3d12RHI::CreateDevice() {
#if defined(D3D12_RHI_DEBUG)
    // Enable the D3D12 debug layer.
    {
        if (SUCCEEDED(
                D3D12GetDebugInterface(IID_PPV_ARGS(&m_pDebugController)))) {
            m_pDebugController->EnableDebugLayer();
        }
    }
#endif

    m_pFactory;
    assert(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&m_pFactory))) &&
           "CreateDXGIFactory1 failed");

    IDXGIAdapter1* pHardwareAdapter;
    GetHardwareAdapter(m_pFactory, &pHardwareAdapter);

    if (FAILED(D3D12CreateDevice(pHardwareAdapter, D3D_FEATURE_LEVEL_12_0,
                                 IID_PPV_ARGS(&m_pDev)))) {
        IDXGIAdapter* pWarpAdapter;
        if (FAILED(m_pFactory->EnumWarpAdapter(IID_PPV_ARGS(&pWarpAdapter)))) {
            SafeRelease(&m_pFactory);
            assert(0);
        }

        if (FAILED(D3D12CreateDevice(pWarpAdapter, D3D_FEATURE_LEVEL_12_0,
                                     IID_PPV_ARGS(&m_pDev)))) {
            SafeRelease(&m_pFactory);
            assert(0);
        }
    }

    static const D3D_FEATURE_LEVEL s_featureLevels[] = {D3D_FEATURE_LEVEL_12_1,
                                                        D3D_FEATURE_LEVEL_12_0};

    D3D12_FEATURE_DATA_FEATURE_LEVELS featLevels = {
        _countof(s_featureLevels), s_featureLevels, D3D_FEATURE_LEVEL_12_0};

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
    assert(SUCCEEDED(m_pDev->CheckFeatureSupport(
        D3D12_FEATURE_FEATURE_LEVELS, &featLevels, sizeof(featLevels))));
    featureLevel = featLevels.MaxSupportedFeatureLevel;
    switch (featureLevel) {
        case D3D_FEATURE_LEVEL_12_0:
            std::cerr << "Device Feature Level: 12.0" << std::endl;
            break;
        case D3D_FEATURE_LEVEL_12_1:
            std::cerr << "Device Feature Level: 12.1" << std::endl;
            break;
    }
}

void D3d12RHI::EnableDebugLayer() {
#if defined(D3D12_RHI_DEBUG)
    // Enable the D3D12 debug layer.
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&m_pDebugController)))) {
        m_pDebugController->EnableDebugLayer();

        m_pDev->QueryInterface(IID_PPV_ARGS(&m_pDebugDev));

        ID3D12InfoQueue* d3dInfoQueue;
        if (SUCCEEDED(m_pDev->QueryInterface(IID_PPV_ARGS(&d3dInfoQueue)))) {
            d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION,
                                             true);
            d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR,
                                             true);
            d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING,
                                             false);

            D3D12_MESSAGE_ID blockedIds[] = {
                D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_ID_COPY_DESCRIPTORS_INVALID_RANGES};

            D3D12_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.pIDList = blockedIds;
            filter.DenyList.NumIDs = 3;
            d3dInfoQueue->AddRetrievalFilterEntries(&filter);
            d3dInfoQueue->AddStorageFilterEntries(&filter);
        }
    }
#endif
}

void D3d12RHI::CreateCommandQueues() {
    // Describe and create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    if (FAILED(m_pDev->CreateCommandQueue(
            &queueDesc, IID_PPV_ARGS(&m_pGraphicsCommandQueue)))) {
        assert(0);
    }

    m_pGraphicsCommandQueue->SetName(L"Graphics Command Queue");

    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;

    if (FAILED(m_pDev->CreateCommandQueue(
            &queueDesc, IID_PPV_ARGS(&m_pComputeCommandQueue)))) {
        assert(0);
    }

    m_pComputeCommandQueue->SetName(L"Compute Command Queue");

    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;

    if (FAILED(m_pDev->CreateCommandQueue(
            &queueDesc, IID_PPV_ARGS(&m_pCopyCommandQueue)))) {
        assert(0);
    }

    m_pCopyCommandQueue->SetName(L"Copy Command Queue");
}

void D3d12RHI::CreateSwapChain() {
    uint32_t width, height;
    m_fQueryFramebufferSize(width, height);

    m_ViewPort = {
        0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height),
        0.0f, 1.0f};

    m_ScissorRect = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};

    // create a struct to hold information about the swap chain
    DXGI_SWAP_CHAIN_DESC1 scd;

    // clear out the struct for use
    ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC1));

    // fill the swap chain description struct
    scd.Width = width;
    scd.Height = height;
    scd.Format = m_eSurfaceFormat;
    scd.Stereo = FALSE;
    scd.SampleDesc.Count =
        1;  // multi-samples can not be used when in SwapEffect sets to
            // DXGI_SWAP_EFFECT_FLOP_DISCARD
    scd.SampleDesc.Quality =
        0;  // multi-samples can not be used when in SwapEffect sets to
            // DXGI_SWAP_EFFECT_FLOP_DISCARD
    scd.BufferUsage =
        DXGI_USAGE_RENDER_TARGET_OUTPUT;  // how swap chain is to be used
    scd.BufferCount =
        GfxConfiguration::kMaxInFlightFrameCount;  // back buffer count
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.SwapEffect =
        DXGI_SWAP_EFFECT_FLIP_DISCARD;  // DXGI_SWAP_EFFECT_FLIP_DISCARD only
                                        // supported after Win10 use
                                        // DXGI_SWAP_EFFECT_DISCARD on platforms
                                        // early than Win10
    scd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    scd.Flags =
        DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;  // allow full-screen transition

    IDXGISwapChain1* pSwapChain;
    HWND hWnd = m_fGetWindowHandler();

    if (FAILED(m_pFactory->CreateSwapChainForHwnd(
            m_pGraphicsCommandQueue,  // Swap chain needs the queue so
                                      // that it can force a flush on it
            hWnd, &scd, NULL, NULL, &pSwapChain))) {
        assert(0);
    }

    m_pSwapChain = reinterpret_cast<IDXGISwapChain3*>(pSwapChain);
}

void D3d12RHI::CreateSyncObjects() {
    assert(SUCCEEDED(m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                         IID_PPV_ARGS(&m_pGraphicsFence))) &&
           "failed to create fence object");

    m_nGraphicsFenceValues.fill(0);

    m_hGraphicsFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    assert(m_hGraphicsFenceEvent);

    m_hComputeFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    assert(m_hComputeFenceEvent);

    m_hCopyFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    assert(m_hCopyFenceEvent);
}

void D3d12RHI::CreateRenderTargets() {
    m_pRenderTargets.resize(GfxConfiguration::kMaxInFlightFrameCount);

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        assert(SUCCEEDED(
            m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_pRenderTargets[i]))));

        m_pRenderTargets[i]->SetName(
            (std::wstring(L"Render Target ") + std::to_wstring(i)).c_str());
    }

    uint32_t width, height;
    m_fQueryFramebufferSize(width, height);

    auto config = m_fGetGfxConfigHandler();
    if (config.msaaSamples > 1) {
        D3D12_RESOURCE_DESC textureDesc{};
        textureDesc.MipLevels = 1;
        textureDesc.Format = m_eSurfaceFormat;
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.SampleDesc.Count = config.msaaSamples;
        textureDesc.SampleDesc.Quality =
            DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;

        // Create intermediate MSAA RT
        D3D12_CLEAR_VALUE optimizedClearValue = {m_eSurfaceFormat,
                                                 {0.2f, 0.3f, 0.4f, 1.0f}};

        D3D12_HEAP_PROPERTIES prop{};
        prop.Type = D3D12_HEAP_TYPE_DEFAULT;
        prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        prop.CreationNodeMask = 1;
        prop.VisibleNodeMask = 1;

        ID3D12Resource* pMsaaRT;
        assert(SUCCEEDED(m_pDev->CreateCommittedResource(
            &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, &optimizedClearValue,
            IID_PPV_ARGS(&pMsaaRT))));

        pMsaaRT->SetName(L"MSAA Render Target");

        m_pRenderTargets.push_back(pMsaaRT);
    }
}

void D3d12RHI::CreateDepthStencils() {
    uint32_t width, height;
    m_fQueryFramebufferSize(width, height);

    auto config = m_fGetGfxConfigHandler();

    D3D12_CLEAR_VALUE depthOptimizedClearValue{};
    depthOptimizedClearValue.Format = ::DXGI_FORMAT_D32_FLOAT;
    depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
    depthOptimizedClearValue.DepthStencil.Stencil = 0;

    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_D32_FLOAT;
    resourceDesc.SampleDesc.Count = config.msaaSamples;
    if (config.msaaSamples > 1) {
        resourceDesc.SampleDesc.Quality =
            DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    } else {
        resourceDesc.SampleDesc.Quality = 0;
    }
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthOptimizedClearValue,
        IID_PPV_ARGS(&m_pDepthStencilBuffer))));

    m_pDepthStencilBuffer->SetName(L"DepthStencilBuffer");
}

void D3d12RHI::CreateFramebuffers() {
    auto config = m_fGetGfxConfigHandler();
    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.NumDescriptors =
        (config.msaaSamples > 1) ? 2 : 1;  // 1 for present + 1 for MSAA
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    // Describe and create RTV descriptor heap, then fill with RTV descriptors.
    m_pRtvHeaps.resize(GfxConfiguration::kMaxInFlightFrameCount);
    m_nRtvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        assert(SUCCEEDED(m_pDev->CreateDescriptorHeap(
            &rtvHeapDesc, IID_PPV_ARGS(&m_pRtvHeaps[i]))));

        m_pRtvHeaps[i]->SetName(
            (std::wstring(L"RTV Descriptor Heap") + std::to_wstring(i))
                .c_str());

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle =
            m_pRtvHeaps[i]->GetCPUDescriptorHandleForHeapStart();

        m_pDev->CreateRenderTargetView(m_pRenderTargets[i], nullptr, rtvHandle);

        if (m_fGetGfxConfigHandler().msaaSamples > 1) {
            rtvHandle.ptr += m_nRtvDescriptorSize;

            m_pDev->CreateRenderTargetView(m_pRenderTargets.back(), nullptr,
                                           rtvHandle);
        }
    }

    // Describe and create a depth stencil view (DSV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.NumDescriptors = 1;  // 1 for scene
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    assert(SUCCEEDED(
        m_pDev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_pDsvHeap))));
    m_pDsvHeap->SetName(L"DSV Descriptor Heap");

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle =
        m_pDsvHeap->GetCPUDescriptorHandleForHeapStart();

    m_pDev->CreateDepthStencilView(m_pDepthStencilBuffer, nullptr, dsvHandle);
}

void D3d12RHI::CreateCommandPools() {
    // Graphics
    m_pGraphicsCommandAllocators.resize(GfxConfiguration::kMaxInFlightFrameCount);

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        assert(SUCCEEDED(m_pDev->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&m_pGraphicsCommandAllocators[i]))));
        m_pGraphicsCommandAllocators[i]->SetName((std::wstring(L"Graphics Command Allocator") + std::to_wstring(i)).c_str());
    }

    // Compute
    assert(SUCCEEDED(m_pDev->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&m_pComputeCommandAllocator))));

    m_pComputeCommandAllocator->SetName(L"Compute Command Allocator");

    // Copy
    assert(SUCCEEDED(m_pDev->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_pCopyCommandAllocator))));

    m_pCopyCommandAllocator->SetName(L"Copy Command Allocator");
}

void D3d12RHI::CreateCommandLists() {
    // Graphics
    m_pGraphicsCommandLists.resize(1);

    assert(SUCCEEDED(m_pDev->CreateCommandList1(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE,
        IID_PPV_ARGS(&m_pGraphicsCommandLists[0]))));

    m_pGraphicsCommandLists[0]->SetName(
        (std::wstring(L"Graphics Command List ") + std::to_wstring(0))
            .c_str());

    // Compute
    assert(SUCCEEDED(m_pDev->CreateCommandList1(
        0, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_FLAG_NONE,
        IID_PPV_ARGS(&m_pComputeCommandList))));

    m_pComputeCommandList->SetName(L"Compute Command List");

    // Copy
    assert(SUCCEEDED(m_pDev->CreateCommandList1(
        0, D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_FLAG_NONE,
        IID_PPV_ARGS(&m_pCopyCommandList))));

    m_pCopyCommandList->SetName(L"Copy Command List");
}

void D3d12RHI::beginSingleTimeCommands() {
    m_pCopyCommandList->Reset(m_pCopyCommandAllocator, nullptr);
}

void D3d12RHI::endSingleTimeCommands() {
    if (SUCCEEDED(m_pCopyCommandList->Close())) {
        ID3D12CommandList* ppCommandLists[] = {m_pCopyCommandList};
        m_pCopyCommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                                 ppCommandLists);
    }

    ID3D12Fence* pCopyQueueFence;
    assert(SUCCEEDED(m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                         IID_PPV_ARGS(&pCopyQueueFence))));

    assert(SUCCEEDED(m_pCopyCommandQueue->Signal(pCopyQueueFence, 1)));

    assert(
        SUCCEEDED(pCopyQueueFence->SetEventOnCompletion(1, m_hCopyFenceEvent)));

    WaitForSingleObject(m_hCopyFenceEvent, INFINITE);

    SafeRelease(&pCopyQueueFence);
}

ID3D12Resource* D3d12RHI::CreateTextureImage(Image& img) {
    // Describe and create a Texture2D.
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    DXGI_FORMAT format = getDxgiFormat(img);

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.MipLevels = 1;
    textureDesc.Format = format;
    textureDesc.Width = img.Width;
    textureDesc.Height = img.Height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    ID3D12Resource* pTextureBuffer;
    ID3D12Resource* pTextureUploadHeap;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&pTextureBuffer))));

    const UINT subresourceCount =
        textureDesc.DepthOrArraySize * textureDesc.MipLevels;
    const UINT64 uploadBufferSize =
        GetRequiredIntermediateSize(pTextureBuffer, 0, subresourceCount);

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = uploadBufferSize;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&pTextureUploadHeap))));

    // Copy data to the intermediate upload heap and then schedule a copy
    // from the upload heap to the Texture2D.
    D3D12_SUBRESOURCE_DATA textureData{};
    if (img.compressed) {
        textureData.pData = img.data;
        textureData.RowPitch = img.pitch;
        textureData.SlicePitch = img.data_size;
    } else {
        textureData.pData = img.data;
        textureData.RowPitch = img.pitch;
        textureData.SlicePitch = img.pitch * img.Height;
    }

    beginSingleTimeCommands();
    UpdateSubresources(m_pCopyCommandList, pTextureBuffer, pTextureUploadHeap,
                       0, 0, subresourceCount, &textureData);
    endSingleTimeCommands();

    SafeRelease(&pTextureUploadHeap);

    return pTextureBuffer;
}

void D3d12RHI::CreateTextureSampler() {
    const uint32_t num_samplers = 8;
    // Describe and create a sampler descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc{};
    samplerHeapDesc.NumDescriptors =
        num_samplers;  // this is the max D3d12 HW support currently
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    assert(SUCCEEDED(m_pDev->CreateDescriptorHeap(
        &samplerHeapDesc, IID_PPV_ARGS(&m_pSamplerHeap))));
    m_pSamplerHeap->SetName(L"Sampler Descriptor Heap");

    // Describe and create a sampler.
    D3D12_SAMPLER_DESC samplerDesc{};
    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
    samplerDesc.MipLODBias = 0.0f;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;

    // create samplers
    m_nSamplerDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    D3D12_CPU_DESCRIPTOR_HANDLE samplerHandle =
        m_pSamplerHeap->GetCPUDescriptorHandleForHeapStart();
    for (int32_t i = 0; i < num_samplers; i++) {
        m_pDev->CreateSampler(&samplerDesc, samplerHandle);
        samplerHandle.ptr += m_nSamplerDescriptorSize;
    }
}

ID3D12RootSignature* D3d12RHI::CreateRootSignature(
    const D3D12_SHADER_BYTECODE& shader) {
    // create the root signature
    ID3D12RootSignature* pRootSignature;
    assert(SUCCEEDED(m_pDev->CreateRootSignature(
        0, shader.pShaderBytecode, shader.BytecodeLength,
        IID_PPV_ARGS(&pRootSignature))));

    return pRootSignature;
}

ID3D12PipelineState* D3d12RHI::CreateGraphicsPipeline(
    D3D12_GRAPHICS_PIPELINE_STATE_DESC& psod) {
    ID3D12PipelineState* pPipelineState;

    assert(SUCCEEDED(m_pDev->CreateGraphicsPipelineState(
        &psod, IID_PPV_ARGS(&pPipelineState))));

    return pPipelineState;
}

ID3D12PipelineState* D3d12RHI::CreateComputePipeline(
    D3D12_COMPUTE_PIPELINE_STATE_DESC& psod) {
    ID3D12PipelineState* pPipelineState;

    assert(SUCCEEDED(m_pDev->CreateComputePipelineState(
        &psod, IID_PPV_ARGS(&pPipelineState))));

    return pPipelineState;
}

void D3d12RHI::CreateDescriptorPool(size_t num_descriptors,
                                    const wchar_t* heap_group_name,
                                    size_t num_heaps) {
    // Describe and create a CBV SRV UAV descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc{};
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvSrvUavHeapDesc.NumDescriptors = num_descriptors;
    m_nCbvSrvUavDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    m_pCbvSrvUavHeaps.resize(num_heaps);
    for (int i = 0; i < num_heaps; i++) {
        assert(SUCCEEDED(m_pDev->CreateDescriptorHeap(
            &cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_pCbvSrvUavHeaps[i]))));
        m_pCbvSrvUavHeaps[i]->SetName(heap_group_name);
    }
}

void D3d12RHI::CreateDescriptorSets(ID3D12Resource** ppResources,
                                    size_t count) {
    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        // uniform buffer (constant buffer)
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = m_pUniformBuffers[i]->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = ALIGN(sizeof(UniformBufferObject), 256);

        D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
        cbvHandle = m_pCbvSrvUavHeaps[i]->GetCPUDescriptorHandleForHeapStart();
        m_pDev->CreateConstantBufferView(&cbvDesc, cbvHandle);

        cbvHandle.ptr += m_nCbvSrvUavDescriptorSize;

        // SRVs
        for (size_t n = 0; n < count; n++) {
            m_pDev->CreateShaderResourceView(ppResources[n], NULL, cbvHandle);
            cbvHandle.ptr += m_nCbvSrvUavDescriptorSize;
        }
    }
}

void D3d12RHI::waitOnFrame() {
    if (m_pGraphicsCommandQueue && m_pGraphicsFence &&
        m_hGraphicsFenceEvent != INVALID_HANDLE_VALUE) {
        auto fence_value = m_nGraphicsFenceValues[m_nCurrentFrame];

        assert(SUCCEEDED(
            m_pGraphicsCommandQueue->Signal(m_pGraphicsFence, fence_value)));

        assert(SUCCEEDED(m_pGraphicsFence->SetEventOnCompletion(
            fence_value, m_hGraphicsFenceEvent)));

        std::ignore =
            WaitForSingleObjectEx(m_hGraphicsFenceEvent, INFINITE, FALSE);

        m_nGraphicsFenceValues[m_nCurrentFrame]++;
    }
}

void D3d12RHI::moveToNextFrame() {
    auto current_fence_value = m_nGraphicsFenceValues[m_nCurrentFrame];

    assert(SUCCEEDED(m_pGraphicsCommandQueue->Signal(m_pGraphicsFence,
                                                     current_fence_value)));

    m_nCurrentFrame = m_pSwapChain->GetCurrentBackBufferIndex();

    if (m_pGraphicsFence->GetCompletedValue() <
        m_nGraphicsFenceValues[m_nCurrentFrame]) {
        assert(SUCCEEDED(m_pGraphicsFence->SetEventOnCompletion(
            m_nGraphicsFenceValues[m_nCurrentFrame], m_hGraphicsFenceEvent)));

        std::ignore =
            WaitForSingleObjectEx(m_hGraphicsFenceEvent, INFINITE, FALSE);
    }

    m_nGraphicsFenceValues[m_nCurrentFrame] = current_fence_value + 1;
}

void D3d12RHI::CleanupSwapChain() {
    waitOnFrame();

    for (auto& heap : m_pRtvHeaps) {
        SafeRelease(&heap);
    }

    for (auto& rt : m_pRenderTargets) {
        SafeRelease(&rt);
    }

    SafeRelease(&m_pDsvHeap);

    SafeRelease(&m_pDepthStencilBuffer);

    SafeRelease(&m_pSwapChain);
}

void D3d12RHI::RecreateSwapChain() {
    if (m_pSwapChain) {
        CleanupSwapChain();
        CreateSwapChain();
        CreateRenderTargets();
        CreateDepthStencils();
        CreateFramebuffers();

        m_nCurrentFrame = 0;
        m_nGraphicsFenceValues.fill(0);
    }
}

void D3d12RHI::ResetAllBuffers() {
    for (auto& buf : m_pRawBuffers) {
        SafeRelease(&buf);
    }

    for (auto& buf : m_pVertexBuffers) {
        SafeRelease(&buf);
    }

    for (auto& buf : m_pIndexBuffers) {
        SafeRelease(&buf);
    }

    for (auto& buf : m_pUniformBuffers) {
        SafeRelease(&buf);
    }

    m_VertexBufferViews.clear();
    m_IndexBufferViews.clear();
}

void D3d12RHI::DestroyAll() {
    CleanupSwapChain();

    if (m_fDestroyResourceHandler) {
        m_fDestroyResourceHandler();
    }

    ResetAllBuffers();

    for (auto& pool : m_pCbvSrvUavHeaps) {
        SafeRelease(&pool);
    }

    SafeRelease(&m_pSamplerHeap);

    SafeRelease(&m_pGraphicsFence);

    for (auto& commandList : m_pGraphicsCommandLists) {
        SafeRelease(&commandList);
    }
    SafeRelease(&m_pComputeCommandList);
    SafeRelease(&m_pCopyCommandList);

    SafeRelease(&m_pCopyCommandAllocator);
    SafeRelease(&m_pComputeCommandAllocator);

    for (auto& commandAllocator : m_pGraphicsCommandAllocators) {
        SafeRelease(&commandAllocator);
    }

    CloseHandle(m_hComputeFenceEvent);
    CloseHandle(m_hCopyFenceEvent);
    CloseHandle(m_hGraphicsFenceEvent);

    SafeRelease(&m_pGraphicsCommandQueue);
    SafeRelease(&m_pComputeCommandQueue);
    SafeRelease(&m_pCopyCommandQueue);

    SafeRelease(&m_pDev);
    SafeRelease(&m_pFactory);

#if defined(D3D12_RHI_DEBUG)
    SafeRelease(&m_pDebugController);

    if (m_pDebugDev) {
        m_pDebugDev->ReportLiveDeviceObjects(D3D12_RLDO_DETAIL);
    }

    SafeRelease(&m_pDebugDev);
#endif
}

void D3d12RHI::BeginPass(const Vector4f& clearColor) {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];

    m_pCmdList->Reset(m_pGraphicsCommandAllocators[m_nCurrentFrame], NULL);

    m_pCmdList->RSSetViewports(1, &m_ViewPort);
    m_pCmdList->RSSetScissorRects(1, &m_ScissorRect);

    // bind the frame buffer
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;

    rtvHandle.ptr =
        m_pRtvHeaps[m_nCurrentFrame]->GetCPUDescriptorHandleForHeapStart().ptr;

    if (m_fGetGfxConfigHandler().msaaSamples > 1) {
        // point to MASS buffer
        rtvHandle.ptr += m_nRtvDescriptorSize;
    }

    // transfer resource state
    D3D12_RESOURCE_BARRIER barrier;

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[m_nCurrentFrame];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    m_pCmdList->ResourceBarrier(1, &barrier);

    // set frame buffers
    dsvHandle = m_pDsvHeap->GetCPUDescriptorHandleForHeapStart();
    m_pCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    // clear the back buffer to a deep blue
    m_pCmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    m_pCmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f,
                                      0, 0, nullptr);
}

void D3d12RHI::SetPipelineState(ID3D12PipelineState* pPipelineState) {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];

    m_pCmdList->SetPipelineState(pPipelineState);
}

void D3d12RHI::SetRootSignature(ID3D12RootSignature* pRootSignature) {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];
    m_pCmdList->SetGraphicsRootSignature(pRootSignature);
}

void D3d12RHI::Draw() {
    auto config = m_fGetGfxConfigHandler();

    auto& m_pCmdList = m_pGraphicsCommandLists[0];

    // set which vertex buffer to use
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    vertexBufferView.BufferLocation =
        m_pVertexBuffers[0]->GetGPUVirtualAddress();
    vertexBufferView.StrideInBytes = (UINT)sizeof(m_Vertices[0]);
    vertexBufferView.SizeInBytes = m_Vertices.size() * sizeof(m_Vertices[0]);

    m_pCmdList->IASetVertexBuffers(0, 1, &vertexBufferView);

    // set which index to use
    D3D12_INDEX_BUFFER_VIEW indexBufferView;
    indexBufferView.BufferLocation = m_pIndexBuffers[0]->GetGPUVirtualAddress();
    indexBufferView.Format = DXGI_FORMAT::DXGI_FORMAT_R32_UINT;
    indexBufferView.SizeInBytes = (UINT)m_Indices.size() * sizeof(m_Indices[0]);

    m_pCmdList->IASetIndexBuffer(&indexBufferView);

    // set primitive topology
    m_pCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // set descriptor heaps
    ID3D12DescriptorHeap* ppHeaps[] = {m_pCbvSrvUavHeaps[m_nCurrentFrame],
                                       m_pSamplerHeap};
    m_pCmdList->SetDescriptorHeaps(static_cast<int32_t>(_countof(ppHeaps)),
                                   ppHeaps);

    // Bind per batch Descriptor Table
    auto descriptorHandler = m_pCbvSrvUavHeaps[m_nCurrentFrame]
                                 ->GetGPUDescriptorHandleForHeapStart();
    m_pCmdList->SetGraphicsRootDescriptorTable(0, descriptorHandler);

    // Sampler
    descriptorHandler = m_pSamplerHeap->GetGPUDescriptorHandleForHeapStart();
    m_pCmdList->SetGraphicsRootDescriptorTable(1, descriptorHandler);

    // draw the vertex buffer to the back buffer
    m_pCmdList->DrawIndexedInstanced(m_Indices.size(), 1, 0, 0, 0);

    // 更新常量
    updateUniformBufer();
}

void D3d12RHI::EndPass() {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];
}

void D3d12RHI::Present() {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];

    D3D12_RESOURCE_BARRIER barrier;

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[m_nCurrentFrame];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    m_pCmdList->ResourceBarrier(1, &barrier);

    if (SUCCEEDED(m_pCmdList->Close())) {
        ID3D12CommandList* ppCommandLists[] = {m_pCmdList};
        m_pGraphicsCommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                                     ppCommandLists);

        assert(SUCCEEDED(m_pGraphicsCommandQueue->Signal(
            m_pGraphicsFence, m_nGraphicsFenceValues[m_nCurrentFrame])));
    }

    // swap the back buffer and the front buffer
    assert(SUCCEEDED(m_pSwapChain->Present(1, 0)));

    moveToNextFrame();
}

void D3d12RHI::msaaResolve() {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];

    D3D12_RESOURCE_BARRIER barrier[2];

    barrier[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[0].Transition.pResource = m_pRenderTargets.back();
    barrier[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barrier[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[1].Transition.pResource = m_pRenderTargets[m_nCurrentFrame];
    barrier[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCmdList->ResourceBarrier(2, barrier);

    m_pCmdList->ResolveSubresource(m_pRenderTargets[m_nCurrentFrame], 0,
                                   m_pRenderTargets.back(), 0,
                                   m_eSurfaceFormat);

    barrier[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[0].Transition.pResource = m_pRenderTargets.back();
    barrier[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barrier[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[1].Transition.pResource = m_pRenderTargets[m_nCurrentFrame];
    barrier[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCmdList->ResourceBarrier(2, barrier);
}

void D3d12RHI::CreateVertexBuffer() {
    auto bufferSize = sizeof(m_Vertices[0]) * m_Vertices.size();

    ID3D12Resource* pVertexBufferUploadBuffer;

    // create vertex GPU heap
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = bufferSize;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* pVertexBuffer;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&pVertexBuffer))));

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&pVertexBufferUploadBuffer))));

    D3D12_SUBRESOURCE_DATA vertexData{};
    vertexData.pData = m_Vertices.data();

    beginSingleTimeCommands();
    UpdateSubresources<1>(m_pCopyCommandList, pVertexBuffer,
                          pVertexBufferUploadBuffer, 0, 0, 1, &vertexData);
    endSingleTimeCommands();

    SafeRelease(&pVertexBufferUploadBuffer);

    m_pVertexBuffers.push_back(pVertexBuffer);
}

void D3d12RHI::CreateIndexBuffer() {
    auto bufferSize = sizeof(m_Indices[0]) * m_Indices.size();

    ID3D12Resource* pIndexBufferUploadBuffer;

    // create index GPU heap
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = bufferSize;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* pIndexBuffer;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&pIndexBuffer))));

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

    assert(SUCCEEDED(m_pDev->CreateCommittedResource(
        &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&pIndexBufferUploadBuffer))));

    D3D12_SUBRESOURCE_DATA indexData{};
    indexData.pData = m_Indices.data();

    beginSingleTimeCommands();
    UpdateSubresources<1>(m_pCopyCommandList, pIndexBuffer,
                          pIndexBufferUploadBuffer, 0, 0, 1, &indexData);
    endSingleTimeCommands();

    SafeRelease(&pIndexBufferUploadBuffer);

    m_pIndexBuffers.push_back(pIndexBuffer);
}

void D3d12RHI::CreateUniformBuffers() {
    auto bufferSize = ALIGN(sizeof(UniformBufferObject), 256);

    D3D12_HEAP_PROPERTIES prop = {D3D12_HEAP_TYPE_UPLOAD,
                                  D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                  D3D12_MEMORY_POOL_UNKNOWN, 1, 1};

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    m_pUniformBuffers.resize(GfxConfiguration::kMaxInFlightFrameCount);

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        resourceDesc.Width = bufferSize;

        assert(SUCCEEDED(m_pDev->CreateCommittedResource(
            &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_pUniformBuffers[i]))));

        m_pUniformBuffers[i]->SetName(L"Per Frame Constant Buffer");
    }
}

void D3d12RHI::setModel(const std::vector<Vertex>& vertices,
                        const std::vector<uint32_t>& indices) {
    m_Vertices = vertices;
    m_Indices = indices;
}

void D3d12RHI::updateUniformBufer() {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    UniformBufferObject ubo{};
    BuildIdentityMatrix(ubo.model);
    MatrixRotationAxis(ubo.model, {0.0f, 0.0f, 1.0f}, time * PI / 8.0f);
    BuildViewRHMatrix(ubo.view, {2.0f, 2.0f, 2.0f}, {0.0f, 0.0f, 0.0f},
                      {0.0f, 0.0f, 1.0f});
    BuildPerspectiveFovRHMatrix(
        ubo.proj, PI / 4.0f, m_ViewPort.Width / m_ViewPort.Height, 0.1f, 10.0f);

    // 上传数据
    void* data;
    D3D12_RANGE readRange = {0, 0};
    assert(SUCCEEDED(
        m_pUniformBuffers[m_nCurrentFrame]->Map(0, &readRange, &data)));

    std::memcpy(data, &ubo, sizeof(ubo));

    m_pUniformBuffers[m_nCurrentFrame]->Unmap(0, &readRange);
}

size_t D3d12RHI::CreateVertexBuffer(const void* pData, size_t data_size,
                                    int32_t stride_size) {
    HRESULT hr;

    ID3D12Resource* pVertexBufferUploadHeap;

    // create vertex GPU heap
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = data_size;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* pVertexBuffer;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&pVertexBuffer)))) {
        return hr;
    }

    m_pVertexBuffers.push_back(pVertexBuffer);

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&pVertexBufferUploadHeap)))) {
        return hr;
    }

    D3D12_SUBRESOURCE_DATA vertexData{};
    vertexData.pData = pData;

    beginSingleTimeCommands();
    UpdateSubresources<1>(m_pCopyCommandList, pVertexBuffer,
                          pVertexBufferUploadHeap, 0, 0, 1, &vertexData);
    endSingleTimeCommands();

    SafeRelease(&pVertexBufferUploadHeap);

    // initialize the vertex buffer view
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    vertexBufferView.BufferLocation = pVertexBuffer->GetGPUVirtualAddress();
    vertexBufferView.StrideInBytes = (UINT)(stride_size);
    vertexBufferView.SizeInBytes = (UINT)data_size;
    auto offset = m_VertexBufferViews.size();
    m_VertexBufferViews.push_back(vertexBufferView);

    return offset;
}

size_t D3d12RHI::CreateIndexBuffer(const void* pData, size_t size,
                                   int32_t index_size) {
    HRESULT hr;

    ID3D12Resource* pIndexBufferUploadHeap;

    // create index GPU heap
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = size;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* pIndexBuffer;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&pIndexBuffer)))) {
        return hr;
    }

    m_pIndexBuffers.push_back(pIndexBuffer);

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&pIndexBufferUploadHeap)))) {
        return hr;
    }

    D3D12_SUBRESOURCE_DATA indexData{};
    indexData.pData = pData;

    beginSingleTimeCommands();
    UpdateSubresources<1>(m_pCopyCommandList, pIndexBuffer,
                          pIndexBufferUploadHeap, 0, 0, 1, &indexData);
    endSingleTimeCommands();

    // initialize the index buffer view
    D3D12_INDEX_BUFFER_VIEW indexBufferView;
    indexBufferView.BufferLocation = pIndexBuffer->GetGPUVirtualAddress();
    switch (index_size) {
        case 2:
            indexBufferView.Format = ::DXGI_FORMAT_R16_UINT;
            break;
        case 4:
            indexBufferView.Format = ::DXGI_FORMAT_R32_UINT;
            break;
        default:
            assert(0);
    }
    indexBufferView.SizeInBytes = (UINT)size;
    auto offset = m_IndexBufferViews.size();
    m_IndexBufferViews.push_back(indexBufferView);

    return offset;
}

void D3d12RHI::BeginFrame() {
    m_pGraphicsCommandAllocators[m_nCurrentFrame]->Reset();
}

void D3d12RHI::EndFrame() {
    // MSAA Resolve
    if (m_fGetGfxConfigHandler().msaaSamples > 1) {
        msaaResolve();
    }
}

#include "imgui_impl_dx12.h"

void D3d12RHI::DrawGUI(ID3D12DescriptorHeap* pCbvRsvHeap) {
    auto& m_pCmdList = m_pGraphicsCommandLists[0];

    // now draw GUI overlay
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    // bind the final RTV
    rtvHandle =
        m_pRtvHeaps[m_nCurrentFrame]->GetCPUDescriptorHandleForHeapStart();
    m_pCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    // set descriptor heaps
    ID3D12DescriptorHeap* ppHeaps[] = {pCbvRsvHeap};
    m_pCmdList->SetDescriptorHeaps(static_cast<int32_t>(_countof(ppHeaps)),
                                   ppHeaps);

    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_pCmdList);
}

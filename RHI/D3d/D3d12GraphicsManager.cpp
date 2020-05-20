#include <iostream>
#include <objbase.h>
#include "D3d12GraphicsManager.hpp"
#include "IApplication.hpp"
#include "SceneManager.hpp"
#include "AssetLoader.hpp"
#include "IPhysicsManager.hpp"
#include "D3d12Utility.hpp"

using namespace My;
using namespace std;

D3d12GraphicsManager::D3dDrawBatchContext::~D3dDrawBatchContext()
{
    SafeRelease(&pCbvSrvUavHeap);
}

D3d12GraphicsManager::~D3d12GraphicsManager()
{
#if defined(D3D12_RHI_DEBUG)
    if (m_pDebugDev)
    {
        m_pDebugDev->ReportLiveDeviceObjects(D3D12_RLDO_DETAIL);
    }

    SafeRelease(&m_pDebugDev);
    SafeRelease(&m_pDebugController);
#endif
}

int  D3d12GraphicsManager::Initialize()
{
    int result = GraphicsManager::Initialize();

	if (!result)
	{
		const GfxConfiguration& config = g_pApp->GetConfiguration();
		m_ViewPort = { 0.0f, 0.0f, static_cast<float>(config.screenWidth), static_cast<float>(config.screenHeight), 0.0f, 1.0f };
		m_ScissorRect = { 0, 0, static_cast<LONG>(config.screenWidth), static_cast<LONG>(config.screenHeight) };
		result = static_cast<int>(CreateGraphicsResources());
	}

    return result;
}

void D3d12GraphicsManager::Finalize()
{
    GraphicsManager::Finalize();

    g_pPipelineStateManager->Clear();

    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        SafeRelease(&m_pFence[i]);
        SafeRelease(&m_pRtvHeap[i]);
        SafeRelease(&m_pDsvHeap[i]);
        SafeRelease(&m_pSamplerHeap[i]);
        SafeRelease(&m_pDepthStencilBuffer[i]);
        SafeRelease(&m_pPerFrameConstantUploadBuffer[i]);
        SafeRelease(&m_pLightInfoUploadBuffer[i]);
#ifdef DEBUG
        SafeRelease(&m_pDebugConstantsUploadBuffer[i]);
#endif
        SafeRelease(&m_pRenderTargets[i << 1]);
        SafeRelease(&m_pRenderTargets[(i << 1) | 1]);
        SafeRelease(&m_pCommandList[i]);
        SafeRelease(&m_pCommandAllocator[i]);
    }
    SafeRelease(&m_pCommandQueue);
    SafeRelease(&m_pSwapChain);

    SafeRelease(&m_pDev);
}

HRESULT D3d12GraphicsManager::WaitForPreviousFrame(uint32_t frame_index) {
    HRESULT hr = S_OK;
    // Wait until the previous frame is finished.
    auto fence = m_nFenceValue[frame_index];

    if (m_pFence[frame_index]->GetCompletedValue() < fence)
    {
        if(FAILED(hr = m_pFence[frame_index]->SetEventOnCompletion(fence, m_hFenceEvent)))
        {
            return hr;
        }
        WaitForSingleObject(m_hFenceEvent, INFINITE);
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDescriptorHeaps() 
{
    HRESULT hr;

    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 2; // 1 for present + 1 for MSAA Resolver
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    // Describe and create a depth stencil view (DSV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    // Describe and create a sampler descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
    samplerHeapDesc.NumDescriptors = 8; // this is the max D3d12 HW support currently
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    m_nRtvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    m_nCbvSrvUavDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_nSamplerDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        if(FAILED(hr = m_pDev->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_pRtvHeap[i])))) {
            return hr;
        }
        m_pRtvHeap[i]->SetName(L"RTV Descriptors");

        if(FAILED(hr = m_pDev->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_pDsvHeap[i])))) {
            return hr;
        }
        m_pDsvHeap[i]->SetName(L"DSV Descriptors");

        if(FAILED(hr = m_pDev->CreateDescriptorHeap(&samplerHeapDesc, IID_PPV_ARGS(&m_pSamplerHeap[i])))) {
            return hr;
        }
        m_pSamplerHeap[i]->SetName(L"Sampler Descriptors");
    }


    return hr;
}

HRESULT D3d12GraphicsManager::CreateRenderTarget() 
{
    HRESULT hr = S_OK;

    D3D12_RENDER_TARGET_VIEW_DESC renderTargetDesc;
    renderTargetDesc.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    renderTargetDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;

    D3D12_CLEAR_VALUE optimizedClearValue = {};
    optimizedClearValue.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    optimizedClearValue.Color[0] = 0.2f;
    optimizedClearValue.Color[1] = 0.3f;
    optimizedClearValue.Color[2] = 0.4f;
    optimizedClearValue.Color[3] = 1.0f;

    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.Width = g_pApp->GetConfiguration().screenWidth;
    textureDesc.Height = g_pApp->GetConfiguration().screenHeight;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = g_pApp->GetConfiguration().msaaSamples;
    textureDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_pRtvHeap[i]->GetCPUDescriptorHandleForHeapStart();

        // Create a RTV
        if (FAILED(hr = m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_pRenderTargets[2 * i])))) {
            return hr;
        }

        m_pDev->CreateRenderTargetView(m_pRenderTargets[2 * i], nullptr, rtvHandle);
        m_pRenderTargets[2 * i]->SetName(L"Render Target");
        rtvHandle.ptr += m_nRtvDescriptorSize;

        // Create intermediate MSAA RT
        if (FAILED(hr = m_pDev->CreateCommittedResource(
            &prop,
            D3D12_HEAP_FLAG_NONE,
            &textureDesc,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            &optimizedClearValue,
            IID_PPV_ARGS(&m_pRenderTargets[2 * i + 1])
        )))
        {
            return hr;
        }

        m_pRenderTargets[2 * i + 1]->SetName(L"MSAA Render Target");

        m_pDev->CreateRenderTargetView(m_pRenderTargets[2 * i + 1], &renderTargetDesc, rtvHandle);
        rtvHandle.ptr += m_nRtvDescriptorSize;
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDepthStencil()
{
    HRESULT hr;

    // Create the depth stencil view.
    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = ::DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

    D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
    depthOptimizedClearValue.Format = ::DXGI_FORMAT_D32_FLOAT;
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
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = ::DXGI_FORMAT_D32_FLOAT;
    resourceDesc.SampleDesc.Count = g_pApp->GetConfiguration().msaaSamples;
    resourceDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        if (FAILED(hr = m_pDev->CreateCommittedResource(
            &prop,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &depthOptimizedClearValue,
            IID_PPV_ARGS(&m_pDepthStencilBuffer[i])
            ))) {
            return hr;
        }

        m_pDepthStencilBuffer[i]->SetName(L"DepthStencilBuffer0");

        m_pDev->CreateDepthStencilView(m_pDepthStencilBuffer[i], &depthStencilDesc, m_pDsvHeap[i]->GetCPUDescriptorHandleForHeapStart());
    }

    return hr;
}

size_t D3d12GraphicsManager::CreateVertexBuffer(const SceneObjectVertexArray& v_property_array)
{
    HRESULT hr;

	ID3D12Resource* pVertexBufferUploadHeap;

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
    resourceDesc.Width = v_property_array.GetDataSize();
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
		&prop,
		D3D12_HEAP_FLAG_NONE,
		&resourceDesc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&pVertexBuffer)
	)))
	{
		return hr;
	}

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

	if (FAILED(hr = m_pDev->CreateCommittedResource(
		&prop,
		D3D12_HEAP_FLAG_NONE,
		&resourceDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&pVertexBufferUploadHeap)
	)))
	{
		return hr;
	}

	D3D12_SUBRESOURCE_DATA vertexData = {};
	vertexData.pData = v_property_array.GetData();

	UpdateSubresources<1>(m_pCommandList[m_nFrameIndex], pVertexBuffer, pVertexBufferUploadHeap, 0, 0, 1, &vertexData);
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource = pVertexBuffer;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

	// initialize the vertex buffer view
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
	vertexBufferView.BufferLocation = pVertexBuffer->GetGPUVirtualAddress();
	vertexBufferView.StrideInBytes = (UINT)(v_property_array.GetDataSize() / v_property_array.GetVertexCount());
	vertexBufferView.SizeInBytes = (UINT)v_property_array.GetDataSize();
    auto offset = m_VertexBufferView.size();
	m_VertexBufferView.push_back(vertexBufferView);

    m_Buffers.push_back(pVertexBuffer);
    m_Buffers.push_back(pVertexBufferUploadHeap);

    return offset;
}


size_t D3d12GraphicsManager::CreateIndexBuffer(const SceneObjectIndexArray& index_array)
{
    HRESULT hr;

	ID3D12Resource* pIndexBufferUploadHeap;

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
    resourceDesc.Width = index_array.GetDataSize();
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
		&prop,
		D3D12_HEAP_FLAG_NONE,
		&resourceDesc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&pIndexBuffer)
	)))
	{
		return hr;
	}

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

	if (FAILED(hr = m_pDev->CreateCommittedResource(
		&prop,
		D3D12_HEAP_FLAG_NONE,
		&resourceDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&pIndexBufferUploadHeap)
	)))
	{
		return hr;
	}

	D3D12_SUBRESOURCE_DATA indexData = {};
	indexData.pData = index_array.GetData();
	
	UpdateSubresources<1>(m_pCommandList[m_nFrameIndex], pIndexBuffer, pIndexBufferUploadHeap, 0, 0, 1, &indexData);
	D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pIndexBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

	// initialize the index buffer view
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	indexBufferView.BufferLocation = pIndexBuffer->GetGPUVirtualAddress();
	indexBufferView.Format = ::DXGI_FORMAT_R32_UINT;
	indexBufferView.SizeInBytes = (UINT)index_array.GetDataSize();
    auto offset = m_IndexBufferView.size();
	m_IndexBufferView.push_back(indexBufferView);

    m_Buffers.push_back(pIndexBuffer);
    m_Buffers.push_back(pIndexBufferUploadHeap);

    return offset;
}

static DXGI_FORMAT getDxgiFormat(const Image& img)
{
    DXGI_FORMAT format;

    if (img.compressed)
    {
        switch (img.compress_format)
        {
            case "DXT1"_u32:
                format = ::DXGI_FORMAT_BC1_UNORM;
                break;
            case "DXT3"_u32:
                format = ::DXGI_FORMAT_BC3_UNORM;
                break;
            case "DXT5"_u32:
                format = ::DXGI_FORMAT_BC5_UNORM;
                break;
            default:
                assert(0);
        }
    }
    else
    {
        switch (img.bitcount)
        {
        case 8:
            format = ::DXGI_FORMAT_R8_UNORM;
            break;
        case 16:
            format = ::DXGI_FORMAT_R8G8_UNORM;
            break;
        case 32:
            format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
            break;
        case 64:
            format = ::DXGI_FORMAT_R16G16B16A16_FLOAT;
            break;
        default:
            assert(0);
        }
    }

    return format;
}

int32_t D3d12GraphicsManager::CreateTextureBuffer(SceneObjectTexture& texture)
{
    HRESULT hr = S_OK;

    const auto& pImage = texture.GetTextureImage();

    // Describe and create a Texture2D.
    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    DXGI_FORMAT format = getDxgiFormat(*pImage);

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = format;
    textureDesc.Width = pImage->Width;
    textureDesc.Height = pImage->Height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    ID3D12Resource* pTextureBuffer;
    ID3D12Resource* pTextureUploadHeap;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&pTextureBuffer))))
    {
        return -1;
    }

    const UINT subresourceCount = textureDesc.DepthOrArraySize * textureDesc.MipLevels;
    const UINT64 uploadBufferSize = GetRequiredIntermediateSize(pTextureBuffer, 0, subresourceCount);

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC resourceDesc = {};
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

    if (FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&pTextureUploadHeap)
    )))
    {
        return -1;
    }

    // Copy data to the intermediate upload heap and then schedule a copy 
    // from the upload heap to the Texture2D.
    D3D12_SUBRESOURCE_DATA textureData = {};
    textureData.pData = pImage->data;
    textureData.RowPitch = pImage->pitch;
    textureData.SlicePitch = static_cast<uint64_t>(pImage->pitch) * static_cast<uint64_t>(pImage->Height);

    UpdateSubresources(m_pCommandList[m_nFrameIndex], pTextureBuffer, pTextureUploadHeap, 0, 0, subresourceCount, &textureData);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pTextureBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    m_Buffers.push_back(pTextureUploadHeap);
    auto texture_id = m_Textures.size();
    m_Textures.push_back(pTextureBuffer);

    return static_cast<int32_t>(texture_id);
}

uint32_t D3d12GraphicsManager::CreateSamplerBuffer()
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

    // create samplers
    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        for (int32_t j = 0; j < 8; j++)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE samplerHandle;
            samplerHandle.ptr = m_pSamplerHeap[i]->GetCPUDescriptorHandleForHeapStart().ptr + static_cast<int64_t>(j) * m_nSamplerDescriptorSize;
            m_pDev->CreateSampler(&samplerDesc, samplerHandle);
        }
    }

    return S_OK;
}

uint32_t D3d12GraphicsManager::CreateConstantBuffer()
{
    HRESULT hr;

    D3D12_HEAP_PROPERTIES prop = { D3D12_HEAP_TYPE_UPLOAD, 
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN, 
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1 };

    D3D12_RESOURCE_DESC resourceDesc = {};
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

    D3D12_RANGE readRange = { 0, 0 };

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        resourceDesc.Width = kSizePerFrameConstantBuffer;

        if(FAILED(hr = m_pDev->CreateCommittedResource(
            &prop,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_pPerFrameConstantUploadBuffer[i]))))
        {
            return hr;
        }

        hr = m_pPerFrameConstantUploadBuffer[i]->Map(0, &readRange, reinterpret_cast<void**>(&m_pPerFrameCbvDataBegin[i]));
        m_pPerFrameConstantUploadBuffer[i]->SetName(L"Per Frame Constant Buffer");

        resourceDesc.Width = kSizeLightInfo;

        if(FAILED(hr = m_pDev->CreateCommittedResource(
            &prop,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_pLightInfoUploadBuffer[i]))))
        {
            return hr;
        }

        hr = m_pLightInfoUploadBuffer[i]->Map(0, &readRange, reinterpret_cast<void**>(&m_pLightInfoBegin[i]));
        m_pLightInfoUploadBuffer[i]->SetName(L"Light Info Buffer");

#ifdef DEBUG
        resourceDesc.Width = kSizeDebugConstantBuffer;

        if(FAILED(hr = m_pDev->CreateCommittedResource(
            &prop,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_pDebugConstantsUploadBuffer[i]))))
        {
            return hr;
        }

        hr = m_pDebugConstantsUploadBuffer[i]->Map(0, &readRange, reinterpret_cast<void**>(&m_pDebugConstantsBegin[i]));
        m_pDebugConstantsUploadBuffer[i]->SetName(L"Debug Constants Buffer");
#endif
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateGraphicsResources()
{
    HRESULT hr;

#if defined(D3D12_RHI_DEBUG)
    // Enable the D3D12 debug layer.
    {
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&m_pDebugController))))
        {
            m_pDebugController->EnableDebugLayer();
        }
    }
#endif

    IDXGIFactory4* pFactory;
    if (FAILED(hr = CreateDXGIFactory1(IID_PPV_ARGS(&pFactory)))) {
        return hr;
    }

    IDXGIAdapter1* pHardwareAdapter;
    GetHardwareAdapter(pFactory, &pHardwareAdapter);

    if (FAILED(hr = D3D12CreateDevice(pHardwareAdapter,
        D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_pDev)))) {

        IDXGIAdapter* pWarpAdapter;
        if (FAILED(hr = pFactory->EnumWarpAdapter(IID_PPV_ARGS(&pWarpAdapter)))) {
            SafeRelease(&pFactory);
            return hr;
        }

        if(FAILED(hr = D3D12CreateDevice(pWarpAdapter, D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(&m_pDev)))) {
            SafeRelease(&pFactory);
            return hr;
        }
    }

#if defined(D3D12_RHI_DEBUG)
    if (m_pDebugController)
    {
        m_pDev->QueryInterface(IID_PPV_ARGS(&m_pDebugDev));

        ID3D12InfoQueue* d3dInfoQueue;
        if(SUCCEEDED(m_pDev->QueryInterface(IID_PPV_ARGS(&d3dInfoQueue))))
        {
            d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
            d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
            d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, false);

            D3D12_MESSAGE_ID blockedIds[] = { D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                                              D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE, 
                                              D3D12_MESSAGE_ID_COPY_DESCRIPTORS_INVALID_RANGES };

            D3D12_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.pIDList = blockedIds;
            filter.DenyList.NumIDs = 3;
            d3dInfoQueue->AddRetrievalFilterEntries(&filter);
            d3dInfoQueue->AddStorageFilterEntries(&filter);
        }
    }
#endif

	static const D3D_FEATURE_LEVEL s_featureLevels[] =
	{
		D3D_FEATURE_LEVEL_12_1,
		D3D_FEATURE_LEVEL_12_0
	};

	D3D12_FEATURE_DATA_FEATURE_LEVELS featLevels =
	{
		_countof(s_featureLevels), s_featureLevels, D3D_FEATURE_LEVEL_12_0
	};

	D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
	hr = m_pDev->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS,
		&featLevels, sizeof(featLevels));
	if (SUCCEEDED(hr))
	{
		featureLevel = featLevels.MaxSupportedFeatureLevel;
		switch (featureLevel)
		{
		case D3D_FEATURE_LEVEL_12_0:
			cerr << "Device Feature Level: 12.0" << endl;
			break;
		case D3D_FEATURE_LEVEL_12_1:
			cerr << "Device Feature Level: 12.1" << endl;
			break;
		}
	}

    HWND hWnd = reinterpret_cast<HWND>(g_pApp->GetMainWindowHandler());

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
    scd.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;              // use 32-bit color
    scd.Stereo = FALSE;
    scd.SampleDesc.Count = 1;                               // multi-samples can not be used when in SwapEffect sets to
                                                            // DXGI_SWAP_EFFECT_FLOP_DISCARD
    scd.SampleDesc.Quality = 0;                             // multi-samples can not be used when in SwapEffect sets to
                                                            // DXGI_SWAP_EFFECT_FLOP_DISCARD
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;      // how swap chain is to be used
    scd.BufferCount = GfxConfiguration::kMaxInFlightFrameCount;                          // back buffer count
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

    SafeRelease(&pFactory);

    m_pSwapChain = reinterpret_cast<IDXGISwapChain3*>(pSwapChain);

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        if (FAILED(hr = m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pFence[i]))))
        {
            return hr;
        }

        m_nFenceValue[i] = 0;
    }

    m_hFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (m_hFenceEvent == NULL)
    {
        hr = HRESULT_FROM_WIN32(GetLastError());
        if (FAILED(hr))
            return hr;
    }

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

    cout << "Creating Depth Stencil Buffer ...";
	if (FAILED(hr = CreateDepthStencil())) {
		return hr;
	}
    cout << "Done!" << endl;

    cout << "Creating Command List ...";
    if (FAILED(hr = CreateCommandList())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating Constant Buffer ...";
	CreateConstantBuffer();
    cout << "Done!" << endl;

    cout << "Creating Sampler Buffer ...";
	CreateSamplerBuffer();
    cout << "Done!" << endl;

    return hr;
}

static std::wstring s2ws(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0); 
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}

// this is the function that loads and prepares the pso 
HRESULT D3d12GraphicsManager::CreatePSO(D3d12PipelineState& pipelineState) {
    HRESULT hr = S_OK;

    D3D12_SHADER_BYTECODE vertexShaderByteCode;
    vertexShaderByteCode.pShaderBytecode = pipelineState.vertexShaderByteCode.pShaderBytecode;
    vertexShaderByteCode.BytecodeLength = pipelineState.vertexShaderByteCode.BytecodeLength;

    D3D12_SHADER_BYTECODE pixelShaderByteCode;
    pixelShaderByteCode.pShaderBytecode = pipelineState.pixelShaderByteCode.pShaderBytecode;
    pixelShaderByteCode.BytecodeLength = pipelineState.pixelShaderByteCode.BytecodeLength;

    D3D12_SHADER_BYTECODE computeShaderByteCode;
    computeShaderByteCode.pShaderBytecode = pipelineState.computeShaderByteCode.pShaderBytecode;
    computeShaderByteCode.BytecodeLength = pipelineState.computeShaderByteCode.BytecodeLength;

    if (pipelineState.pipelineType == PIPELINE_TYPE::GRAPHIC)
    {
        // create the input layout object
        D3D12_INPUT_ELEMENT_DESC ied_full[] =
        {
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TANGENT", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 3, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
        };

        D3D12_INPUT_ELEMENT_DESC ied_simple[] =
        {
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
        };

        D3D12_INPUT_ELEMENT_DESC ied_cube[] =
        {
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 3, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
        };

        D3D12_INPUT_ELEMENT_DESC ied_pos_only[] =
        {
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
        };

        // create rasterizer descriptor
        D3D12_RASTERIZER_DESC rsd = { D3D12_FILL_MODE_SOLID, D3D12_CULL_MODE_BACK, TRUE, 
                                    D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
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

        const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp = { D3D12_STENCIL_OP_KEEP, 
            D3D12_STENCIL_OP_KEEP, 
            D3D12_STENCIL_OP_KEEP, 
            D3D12_COMPARISON_FUNC_ALWAYS };

        D3D12_DEPTH_STENCIL_DESC dsd = { TRUE, 
            D3D12_DEPTH_WRITE_MASK_ALL, 
            D3D12_COMPARISON_FUNC_LESS, 
            FALSE, 
            D3D12_DEFAULT_STENCIL_READ_MASK, 
            D3D12_DEFAULT_STENCIL_WRITE_MASK, 
            defaultStencilOp, defaultStencilOp };

        // create the root signature
        if (FAILED(hr = m_pDev->CreateRootSignature(0, pixelShaderByteCode.pShaderBytecode, pixelShaderByteCode.BytecodeLength, IID_PPV_ARGS(&pipelineState.rootSignature))))
        {
            return false;
        }


        // describe and create the graphics pipeline state object (PSO)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psod {};
        psod.pRootSignature = pipelineState.rootSignature;
        psod.VS             = vertexShaderByteCode;
        psod.PS             = pixelShaderByteCode;
        psod.BlendState     = bld;
        psod.SampleMask     = UINT_MAX;
        psod.RasterizerState= rsd;
        psod.DepthStencilState = dsd;
        switch(pipelineState.a2vType)
        {
            case A2V_TYPES::A2V_TYPES_FULL:
                psod.InputLayout    = { ied_full, _countof(ied_full) };
                break;
            case A2V_TYPES::A2V_TYPES_SIMPLE:
                psod.InputLayout    = { ied_simple, _countof(ied_simple) };
                break;
            case A2V_TYPES::A2V_TYPES_CUBE:
                psod.InputLayout    = { ied_cube, _countof(ied_cube) };
                break;
            case A2V_TYPES::A2V_TYPES_POS_ONLY:
                psod.InputLayout    = { ied_pos_only, _countof(ied_pos_only) };
                break;
            default:
                assert(0);
        }
        psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psod.NumRenderTargets = 1;
        psod.RTVFormats[0]  = ::DXGI_FORMAT_R8G8B8A8_UNORM;
        psod.DSVFormat = ::DXGI_FORMAT_D32_FLOAT;
        psod.SampleDesc.Count = 4; // 4X MSAA
        psod.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;

        if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(&psod, IID_PPV_ARGS(&pipelineState.pipelineState))))
        {
            return false;
        }
    }
    else
    {
        assert(pipelineState.pipelineType == PIPELINE_TYPE::COMPUTE);

        // create the root signature
        if (FAILED(hr = m_pDev->CreateRootSignature(0, computeShaderByteCode.pShaderBytecode, computeShaderByteCode.BytecodeLength, IID_PPV_ARGS(&pipelineState.rootSignature))))
        {
            return false;
        }

        D3D12_CACHED_PIPELINE_STATE cachedPSO;
        cachedPSO.pCachedBlob = nullptr;
        cachedPSO.CachedBlobSizeInBytes = 0;
        D3D12_COMPUTE_PIPELINE_STATE_DESC psod;
        psod.pRootSignature = pipelineState.rootSignature; 
        psod.CS = computeShaderByteCode; 
        psod.NodeMask = 0; 
        psod.CachedPSO.pCachedBlob = nullptr;
        psod.CachedPSO.CachedBlobSizeInBytes = 0;
        psod.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        if (FAILED(hr = m_pDev->CreateComputePipelineState(&psod, IID_PPV_ARGS(&pipelineState.pipelineState))))
        {
            return false;
        }
    }
    pipelineState.pipelineState->SetName(s2ws(pipelineState.pipelineStateName).c_str());

    return hr;
}

HRESULT D3d12GraphicsManager::CreateCommandList()
{
    HRESULT hr = S_OK;

    for(uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        if(FAILED(hr = m_pDev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_pCommandAllocator[i])))) {
            return hr;
        }
        m_pCommandAllocator[i]->SetName((wstring(L"Command Allocator") + to_wstring(i)).c_str());

		hr = m_pDev->CreateCommandList(0, 
					D3D12_COMMAND_LIST_TYPE_DIRECT, 
					m_pCommandAllocator[i], 
					NULL, 
					IID_PPV_ARGS(&m_pCommandList[i]));

        if(SUCCEEDED(hr))
        {
            m_pCommandList[i]->SetName((wstring(L"Command List") + to_wstring(i)).c_str());
            if (i) // close except command list 0
            {
                m_pCommandList[i]->Close();
            }
        }
    }

    return hr;
}

void D3d12GraphicsManager::initializeGeometries(const Scene& scene)
{
    cout << "Creating Draw Batch Contexts ...";
    uint32_t batch_index = 0;
    for (const auto& _it : scene.GeometryNodes)
    {
	    const auto& pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible())
        {
            const auto& pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            const auto& pMesh = pGeometry->GetMesh().lock();
            if(!pMesh) continue;
            
            // Set the number of vertex properties.
            const auto vertexPropertiesCount = pMesh->GetVertexPropertiesCount();
            
            // Set the number of vertices in the vertex array.
            const auto vertexCount = pMesh->GetVertexCount();

            auto dbc = make_shared<D3dDrawBatchContext>();

            for (uint32_t i = 0; i < vertexPropertiesCount; i++)
            {
                const SceneObjectVertexArray& v_property_array = pMesh->GetVertexPropertyArray(i);

                auto offset = CreateVertexBuffer(v_property_array);
                if (i == 0)
                {
                    dbc->property_offset = offset;
                }
            }

            const SceneObjectIndexArray& index_array = pMesh->GetIndexArray(0);
            dbc->index_offset = CreateIndexBuffer(index_array);

			const auto material_index = index_array.GetMaterialIndex();
			const auto material_key = pGeometryNode->GetMaterialRef(material_index);
			const auto& material = scene.GetMaterial(material_key);

            dbc->batchIndex = batch_index++;
			dbc->index_count = (UINT)index_array.GetIndexCount();
            dbc->property_count = vertexPropertiesCount;

            // Describe and create a Constant Buffer View (CBV) descriptor heap.
            D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc = {};
            cbvSrvUavHeapDesc.NumDescriptors = 32;
            cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

            m_pDev->CreateDescriptorHeap(&cbvSrvUavHeapDesc, IID_PPV_ARGS(&dbc->pCbvSrvUavHeap));
            dbc->pCbvSrvUavHeap->SetName(L"PerBatch Descriptor Table");

            // load material textures
            D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle = dbc->pCbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();

            // Jump over per batch CBVs
            srvCpuHandle.ptr += 2 * m_nCbvSrvUavDescriptorSize;

            // SRV
			if (material) {
                if (auto& texture = material->GetBaseColor().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.diffuseMap = texture_id;

                    m_pDev->CreateShaderResourceView(m_Textures[texture_id], NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetNormal().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.normalMap = texture_id;

                    m_pDev->CreateShaderResourceView(m_Textures[texture_id], NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetMetallic().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.metallicMap = texture_id;

                    m_pDev->CreateShaderResourceView(m_Textures[texture_id], NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetRoughness().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.roughnessMap = texture_id;

                    m_pDev->CreateShaderResourceView(m_Textures[texture_id], NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetAO().ValueMap)
                {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.aoMap = texture_id;

                    m_pDev->CreateShaderResourceView(m_Textures[texture_id], NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;
			}

            // UAV
            // ; temporary nothing here

            dbc->node = pGeometryNode;

            for(auto& frame : m_Frames)
            {
                frame.batchContexts.push_back(dbc);
            }
        }
    }
    cout << "Done!" << endl;
}

void D3d12GraphicsManager::initializeSkyBox(const Scene& scene)
{
    HRESULT hr = S_OK;

    assert(scene.SkyBox);

    auto& texture = scene.SkyBox->GetTexture(0);
    const auto& pImage = texture.GetTextureImage();
    DXGI_FORMAT format = getDxgiFormat(*pImage);

    // Describe and create a Cubemap.
    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 2;
    textureDesc.Format = format;
    textureDesc.Width = pImage->Width;
    textureDesc.Height = pImage->Height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    textureDesc.DepthOrArraySize = 6;  // eatch cubemap made by 6 face
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    ID3D12Resource* pTextureBuffer;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&pTextureBuffer))))
    {
        return;
    }

    const UINT subresourceCount = textureDesc.DepthOrArraySize * textureDesc.MipLevels;
    const UINT64 uploadBufferSize = GetRequiredIntermediateSize(pTextureBuffer, 0, subresourceCount);

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC resourceDesc = {};
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

    ID3D12Resource* pTextureUploadHeap;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&pTextureUploadHeap)
    )))
    {
        return;
    }

    // skybox, irradiance map
    for (uint32_t i = 0; i < 6; i++)
    {
        auto& texture = scene.SkyBox->GetTexture(i);
        const auto& pImage = texture.GetTextureImage();

        // Copy data to the intermediate upload heap and then schedule a copy 
        // from the upload heap to the Texture2D.
        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = pImage->data;
        textureData.RowPitch = pImage->pitch;
        textureData.SlicePitch = static_cast<uint64_t>(pImage->pitch) * static_cast<uint64_t>(pImage->Height);

        UpdateSubresources(m_pCommandList[m_nFrameIndex], pTextureBuffer, pTextureUploadHeap, 0, i, 1, &textureData);
    }


    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pTextureBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    m_Buffers.push_back(pTextureUploadHeap);

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        m_Frames[i].skybox = m_Textures.size();
    }

    m_Textures.push_back(pTextureBuffer);
}

void D3d12GraphicsManager::EndScene()
{
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++)
    {
        WaitForPreviousFrame(i);
    }

    for (auto& p : m_Buffers) {
        SafeRelease(&p);
    }
    m_Buffers.clear();
    for (auto& p : m_Textures) {
        SafeRelease(&p);
    }
    m_Textures.clear();
    m_VertexBufferView.clear();
    m_IndexBufferView.clear();

    GraphicsManager::EndScene();
}

void D3d12GraphicsManager::BeginFrame(const Frame& frame)
{
    GraphicsManager::BeginFrame(frame);

    // Indicate that the back buffer will be used as a resolve source.
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex + 1];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    // bind the MSAA RTV and DSV
    rtvHandle.ptr = m_pRtvHeap[frame.frameIndex]->GetCPUDescriptorHandleForHeapStart().ptr + m_nRtvDescriptorSize;
    dsvHandle = m_pDsvHeap[frame.frameIndex]->GetCPUDescriptorHandleForHeapStart();
    m_pCommandList[m_nFrameIndex]->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    m_pCommandList[m_nFrameIndex]->RSSetViewports(1, &m_ViewPort);
    m_pCommandList[m_nFrameIndex]->RSSetScissorRects(1, &m_ScissorRect);

    // clear the back buffer to a deep blue
    const FLOAT clearColor[] = { 0.2f, 0.3f, 0.4f, 1.0f };
    m_pCommandList[m_nFrameIndex]->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    m_pCommandList[m_nFrameIndex]->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    SetPerFrameConstants(frame.frameContext);
    SetLightInfo(frame.lightInfo);
}

void D3d12GraphicsManager::EndFrame(const Frame& frame)
{
    HRESULT hr;

    if (SUCCEEDED(hr = m_pCommandList[m_nFrameIndex]->Close()))
    {
        m_nFenceValue[m_nFrameIndex]++;

        ID3D12CommandList* ppCommandLists[] = { m_pCommandList[m_nFrameIndex] };
        m_pCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

        const uint64_t fence = m_nFenceValue[m_nFrameIndex];
        if(FAILED(hr = m_pCommandQueue->Signal(m_pFence[m_nFrameIndex], fence)))
        {
            assert(0);
        }
    }

    GraphicsManager::EndFrame(frame); // m_nFrameIndex += 1

    if(FAILED(WaitForPreviousFrame(m_nFrameIndex)))
    {
        assert(0);
    }

    ResetCommandList();
}

void D3d12GraphicsManager::Draw()
{
    GraphicsManager::Draw();

    MsaaResolve();
}

void D3d12GraphicsManager::DrawBatch(const Frame& frame)
{
    for (const auto& pDbc : frame.batchContexts)
    {
        const D3dDrawBatchContext& dbc = dynamic_cast<const D3dDrawBatchContext&>(*pDbc);

        // select which vertex buffer(s) to use
        for (uint32_t i = 0; i < dbc.property_count; i++)
        {
            m_pCommandList[frame.frameIndex]->IASetVertexBuffers(i, 1, &m_VertexBufferView[dbc.property_offset + i]);
        }

        // select which index buffer to use
        m_pCommandList[frame.frameIndex]->IASetIndexBuffer(&m_IndexBufferView[dbc.index_offset]);

        // set primitive topology
        m_pCommandList[frame.frameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        // PerFrame CBV (b11)
        m_pCommandList[frame.frameIndex]->SetGraphicsRoot32BitConstants(1, 16, dbc.modelMatrix, 0);

        ID3D12DescriptorHeap* ppHeaps[] = { dbc.pCbvSrvUavHeap, m_pSamplerHeap[frame.frameIndex] };
        m_pCommandList[m_nFrameIndex]->SetDescriptorHeaps(static_cast<int32_t>(_countof(ppHeaps)), ppHeaps);

        // Bind LightInfo (b12)
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = m_pLightInfoUploadBuffer[frame.frameIndex]->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = kSizeLightInfo;

        D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
        cbvHandle = dbc.pCbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
        m_pDev->CreateConstantBufferView(&cbvDesc, cbvHandle);

        cbvHandle.ptr += 2 * m_nCbvSrvUavDescriptorSize;
        // Bind global textures (t6, t10)
        //D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
        cbvHandle.ptr += 6 * m_nCbvSrvUavDescriptorSize;
        m_pDev->CreateShaderResourceView(m_Textures[frame.brdfLUT], NULL, cbvHandle);

        cbvHandle.ptr += 4 * m_nCbvSrvUavDescriptorSize;
        m_pDev->CreateShaderResourceView(m_Textures[frame.skybox], NULL, cbvHandle);

        // Bind per batch Descriptor Table
        m_pCommandList[frame.frameIndex]->SetGraphicsRootDescriptorTable(2, dbc.pCbvSrvUavHeap->GetGPUDescriptorHandleForHeapStart());

        // Sampler (s0)
        D3D12_GPU_DESCRIPTOR_HANDLE cbvSrvUavGpuHandle = m_pSamplerHeap[frame.frameIndex]->GetGPUDescriptorHandleForHeapStart();
        m_pCommandList[m_nFrameIndex]->SetGraphicsRootDescriptorTable(3, cbvSrvUavGpuHandle);

        // draw the vertex buffer to the back buffer
        m_pCommandList[m_nFrameIndex]->DrawIndexedInstanced(dbc.index_count, 1, 0, 0, 0);
    }
}

void D3d12GraphicsManager::SetPipelineState(const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame)
{
    if (pipelineState)
    {
        std::shared_ptr<D3d12PipelineState> pState = dynamic_pointer_cast<D3d12PipelineState>(pipelineState);

        if (!pState->pipelineState)
        {
            CreatePSO(*pState);
        }

        m_pCommandList[m_nFrameIndex]->SetPipelineState(pState->pipelineState);

        m_pCommandList[m_nFrameIndex]->SetGraphicsRootSignature(pState->rootSignature);

        switch(pState->pipelineType)
        {
            case PIPELINE_TYPE::GRAPHIC:
            {
                // Per Frame CBV (b10)
                m_pCommandList[frame.frameIndex]->SetGraphicsRootConstantBufferView(0, 
                    m_pPerFrameConstantUploadBuffer[frame.frameIndex]->GetGPUVirtualAddress());
            }
            break;
            case PIPELINE_TYPE::COMPUTE:
            break;
            default:
                assert(0);
        }
    }
}

void D3d12GraphicsManager::SetPerFrameConstants(const DrawFrameContext& context)
{
    memcpy(m_pPerFrameCbvDataBegin[m_nFrameIndex]
            , &static_cast<const PerFrameConstants&>(context), sizeof(PerFrameConstants));
}

void D3d12GraphicsManager::SetLightInfo(const LightInfo& lightInfo)
{
    memcpy(m_pLightInfoBegin[m_nFrameIndex]
            , &lightInfo, sizeof(LightInfo));
}

HRESULT D3d12GraphicsManager::ResetCommandList()
{
    HRESULT hr;

	// command list allocators can only be reset when the associated 
	// command lists have finished execution on the GPU; apps should use 
	// fences to determine GPU execution progress.
	if (SUCCEEDED(hr = m_pCommandAllocator[m_nFrameIndex]->Reset()))
	{
        // however, when ExecuteCommandList() is called on a particular command 
        // list, that command list can then be reset at any time and must be before 
        // re-recording.
        hr = m_pCommandList[m_nFrameIndex]->Reset(m_pCommandAllocator[m_nFrameIndex], NULL);
	}

    return hr;
}

HRESULT D3d12GraphicsManager::MsaaResolve()
{
    D3D12_RESOURCE_BARRIER barrier = {};

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex + 1];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    m_pCommandList[m_nFrameIndex]->ResolveSubresource(m_pRenderTargets[2 * m_nFrameIndex], 0, m_pRenderTargets[2 * m_nFrameIndex + 1], 0, ::DXGI_FORMAT_R8G8B8A8_UNORM);

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    return S_OK;
}

void D3d12GraphicsManager::Present()
{
    HRESULT hr;

    // swap the back buffer and the front buffer
    hr = m_pSwapChain->Present(1, 0);

    (void)hr;
}

void D3d12GraphicsManager::DrawSkyBox()
{
#if 0
    // set primitive topology
    m_pCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // set vertex buffer
    m_pCommandList->IASetVertexBuffers(0, 1, &vertexBufferView);

    // select index buffer
    m_pCommandList->IASetIndexBuffer(&indexBufferView);

    // Texture

    // draw the vertex buffer to the back buffer
    m_pCommandList->DrawIndexedInstanced(sizeof(skyboxIndices) / sizeof(skyboxIndices[0]), 1, 0, 0, 0);
#endif
}

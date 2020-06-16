#include "D3d12GraphicsManager.hpp"

#include <objbase.h>

#include <iostream>

#include "AssetLoader.hpp"
#include "D3d12Utility.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"

#include "imgui/examples/imgui_impl_dx12.h"
#include "imgui/examples/imgui_impl_win32.h"

using namespace My;
using namespace std;

D3d12GraphicsManager::~D3d12GraphicsManager() {
#if defined(D3D12_RHI_DEBUG)
    if (m_pDebugDev) {
        m_pDebugDev->ReportLiveDeviceObjects(D3D12_RLDO_DETAIL);
    }

    SafeRelease(&m_pDebugDev);
    SafeRelease(&m_pDebugController);
#endif
}

int D3d12GraphicsManager::Initialize() {
    int result = GraphicsManager::Initialize();

    if (!result) {
        const GfxConfiguration& config = g_pApp->GetConfiguration();
        m_ViewPort = {0.0f,
                      0.0f,
                      static_cast<float>(config.screenWidth),
                      static_cast<float>(config.screenHeight),
                      0.0f,
                      1.0f};
        m_ScissorRect = {0, 0, static_cast<LONG>(config.screenWidth),
                         static_cast<LONG>(config.screenHeight)};
        result = static_cast<int>(CreateGraphicsResources());
    }

    auto cpuDescriptorHandle =
        m_pCbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
    auto gpuDescriptorHandle =
        m_pCbvSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
    cpuDescriptorHandle.ptr +=
        32 * GfxConfiguration::kMaxSceneObjectCount;  // 2 CBV, 12 SRV, 18 UAV
                                                      // for each object
    gpuDescriptorHandle.ptr +=
        32 * GfxConfiguration::kMaxSceneObjectCount;  // 2 CBV, 12 SRV, 18 UAV
                                                      // for each object
    ImGui_ImplDX12_Init(m_pDev, GfxConfiguration::kMaxInFlightFrameCount,
                        ::DXGI_FORMAT_R8G8B8A8_UNORM, m_pCbvSrvUavHeap,
                        cpuDescriptorHandle, gpuDescriptorHandle);

    return result;
}

void D3d12GraphicsManager::Finalize() {
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        WaitForPreviousFrame(i);
    }

    ImGui_ImplDX12_Shutdown();

    GraphicsManager::Finalize();

    g_pPipelineStateManager->Clear();

    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        SafeRelease(&m_pGraphicsFence[i]);
        SafeRelease(&m_pRtvHeap[i]);
        SafeRelease(&m_pDsvHeap[i]);
        SafeRelease(&m_pDepthStencilBuffer[i]);
        SafeRelease(&m_pPerFrameConstantUploadBuffer[i]);
        SafeRelease(&m_pLightInfoUploadBuffer[i]);
#ifdef DEBUG
        SafeRelease(&m_pDebugConstantsUploadBuffer[i]);
#endif
        SafeRelease(&m_pRenderTargets[i << 1]);
        SafeRelease(&m_pRenderTargets[(i << 1) | 1]);
        SafeRelease(&m_pGraphicsCommandList[i]);
        SafeRelease(&m_pGraphicsCommandAllocator[i]);
    }
    SafeRelease(&m_pSamplerHeap);
    SafeRelease(&m_pCbvSrvUavHeap);
    SafeRelease(&m_pComputeCommandList);
    SafeRelease(&m_pComputeCommandAllocator);
    SafeRelease(&m_pCopyCommandList);
    SafeRelease(&m_pCopyCommandAllocator);
    SafeRelease(&m_pGraphicsCommandQueue);
    SafeRelease(&m_pComputeCommandQueue);
    SafeRelease(&m_pCopyCommandQueue);
    SafeRelease(&m_pSwapChain);

    SafeRelease(&m_pDev);

    CloseHandle(m_hComputeFenceEvent);
    CloseHandle(m_hCopyFenceEvent);
    CloseHandle(m_hGraphicsFenceEvent);
}

HRESULT D3d12GraphicsManager::WaitForPreviousFrame(uint32_t frame_index) {
    HRESULT hr = S_OK;
    // Wait until the previous frame is finished.
    auto fence = m_nGraphicsFenceValue[frame_index];

    if (m_pGraphicsFence[frame_index]->GetCompletedValue() < fence) {
        if (FAILED(hr = m_pGraphicsFence[frame_index]->SetEventOnCompletion(
                       fence, m_hGraphicsFenceEvent))) {
            assert(0);
            return hr;
        }
        WaitForSingleObject(m_hGraphicsFenceEvent, INFINITE);

        // command list allocators can only be reset when the associated
        // command lists have finished execution on the GPU; apps should use
        // fences to determine GPU execution progress.
        if (SUCCEEDED(hr = m_pGraphicsCommandAllocator[frame_index]->Reset())) {
            // however, when ExecuteCommandList() is called on a particular
            // command list, that command list can then be reset at any time and
            // must be before re-recording.
            hr = m_pGraphicsCommandList[frame_index]->Reset(
                m_pGraphicsCommandAllocator[frame_index], NULL);
        } else {
            assert(0);
        }
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDescriptorHeaps() {
    HRESULT hr;

    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.NumDescriptors = 2;  // 1 for present + 1 for MSAA Resolver
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    // Describe and create a depth stencil view (DSV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.NumDescriptors =
        1 + MAX_LIGHTS;  // 1 for scene + MAX_LIGHTS for shadow maps
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    // Describe and create a CBV SRV UAV descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC cbvSrvUavHeapDesc{};
    cbvSrvUavHeapDesc.NumDescriptors =
        32 * GfxConfiguration::kMaxSceneObjectCount  // 2 CBV, 12 SRV, 18 UAV
                                                     // for each object
        + 32;                                        // for ImGui
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    // Describe and create a sampler descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc{};
    samplerHeapDesc.NumDescriptors =
        8;  // this is the max D3d12 HW support currently
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    m_nRtvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    m_nCbvSrvUavDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_nSamplerDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        if (FAILED(hr = m_pDev->CreateDescriptorHeap(
                       &rtvHeapDesc, IID_PPV_ARGS(&m_pRtvHeap[i])))) {
            return hr;
        }
        m_pRtvHeap[i]->SetName(
            (wstring(L"RTV Descriptors") + to_wstring(i)).c_str());

        if (FAILED(hr = m_pDev->CreateDescriptorHeap(
                       &dsvHeapDesc, IID_PPV_ARGS(&m_pDsvHeap[i])))) {
            return hr;
        }
        m_pDsvHeap[i]->SetName(L"DSV Descriptors");
    }

    if (FAILED(hr = m_pDev->CreateDescriptorHeap(
                   &cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_pCbvSrvUavHeap)))) {
        return hr;
    }
    m_pCbvSrvUavHeap->SetName(L"Per Batch CBV SRV UAV Descriptors");

    if (FAILED(hr = m_pDev->CreateDescriptorHeap(
                   &samplerHeapDesc, IID_PPV_ARGS(&m_pSamplerHeap)))) {
        return hr;
    }
    m_pSamplerHeap->SetName(L"Sampler Descriptors");

    return hr;
}

HRESULT D3d12GraphicsManager::CreateRenderTarget() {
    HRESULT hr = S_OK;

    D3D12_RENDER_TARGET_VIEW_DESC renderTargetDesc;
    renderTargetDesc.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    renderTargetDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;

    D3D12_CLEAR_VALUE optimizedClearValue = {::DXGI_FORMAT_R8G8B8A8_UNORM,
                                             {0.2f, 0.3f, 0.4f, 1.0f}};

    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.MipLevels = 1;
    textureDesc.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.Width = g_pApp->GetConfiguration().screenWidth;
    textureDesc.Height = g_pApp->GetConfiguration().screenHeight;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.SampleDesc.Count = g_pApp->GetConfiguration().msaaSamples;
    textureDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle =
            m_pRtvHeap[i]->GetCPUDescriptorHandleForHeapStart();

        // Create a RTV
        if (FAILED(hr = m_pSwapChain->GetBuffer(
                       i, IID_PPV_ARGS(&m_pRenderTargets[2 * i])))) {
            return hr;
        }

        m_pRenderTargets[2 * i]->SetName(
            (wstring(L"Render Target") + to_wstring(i)).c_str());

        // Create intermediate MSAA RT
        if (FAILED(hr = m_pDev->CreateCommittedResource(
                       &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, &optimizedClearValue,
                       IID_PPV_ARGS(&m_pRenderTargets[2 * i + 1])))) {
            return hr;
        }

        m_pRenderTargets[2 * i + 1]->SetName(
            (wstring(L"MSAA Render Target") + to_wstring(i)).c_str());

        m_pDev->CreateRenderTargetView(m_pRenderTargets[2 * i], nullptr,
                                       rtvHandle);

        rtvHandle.ptr += m_nRtvDescriptorSize;

        m_pDev->CreateRenderTargetView(m_pRenderTargets[2 * i + 1],
                                       &renderTargetDesc, rtvHandle);
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDepthStencil() {
    HRESULT hr;

    // Create the depth stencil view.
    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc{};
    depthStencilDesc.Format = ::DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

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

    uint32_t width = g_pApp->GetConfiguration().screenWidth;
    uint32_t height = g_pApp->GetConfiguration().screenHeight;
    D3D12_RESOURCE_DESC resourceDesc{};
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

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        if (FAILED(hr = m_pDev->CreateCommittedResource(
                       &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                       D3D12_RESOURCE_STATE_DEPTH_WRITE,
                       &depthOptimizedClearValue,
                       IID_PPV_ARGS(&m_pDepthStencilBuffer[i])))) {
            return hr;
        }

        m_pDepthStencilBuffer[i]->SetName(
            (wstring(L"DepthStencilBuffer") + to_wstring(i)).c_str());

        m_pDev->CreateDepthStencilView(
            m_pDepthStencilBuffer[i], &depthStencilDesc,
            m_pDsvHeap[i]->GetCPUDescriptorHandleForHeapStart());
    }

    return hr;
}

size_t D3d12GraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
    const void* pData = v_property_array.GetData();
    auto size = v_property_array.GetDataSize();
    auto stride = size / v_property_array.GetVertexCount();
    return CreateVertexBuffer(pData, size, (int32_t)stride);
}

size_t D3d12GraphicsManager::CreateVertexBuffer(const void* pData,
                                                size_t data_size,
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

    UpdateSubresources<1>(m_pCopyCommandList, pVertexBuffer,
                          pVertexBufferUploadHeap, 0, 0, 1, &vertexData);

    // initialize the vertex buffer view
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    vertexBufferView.BufferLocation = pVertexBuffer->GetGPUVirtualAddress();
    vertexBufferView.StrideInBytes = (UINT)(stride_size);
    vertexBufferView.SizeInBytes = (UINT)data_size;
    auto offset = m_VertexBufferView.size();
    m_VertexBufferView.push_back(vertexBufferView);

    m_Buffers.push_back(pVertexBuffer);
    m_Buffers.push_back(pVertexBufferUploadHeap);

    return offset;
}

size_t D3d12GraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
    const void* pData = index_array.GetData();
    auto size = index_array.GetDataSize();
    int32_t index_size =
        static_cast<int32_t>(size / index_array.GetIndexCount());
    return CreateIndexBuffer(pData, size, index_size);
}

size_t D3d12GraphicsManager::CreateIndexBuffer(const void* pData, size_t size,
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

    prop.Type = D3D12_HEAP_TYPE_UPLOAD;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&pIndexBufferUploadHeap)))) {
        return hr;
    }

    D3D12_SUBRESOURCE_DATA indexData{};
    indexData.pData = pData;

    UpdateSubresources<1>(m_pCopyCommandList, pIndexBuffer,
                          pIndexBufferUploadHeap, 0, 0, 1, &indexData);

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
    auto offset = m_IndexBufferView.size();
    m_IndexBufferView.push_back(indexBufferView);

    m_Buffers.push_back(pIndexBuffer);
    m_Buffers.push_back(pIndexBufferUploadHeap);

    return offset;
}

static DXGI_FORMAT getDxgiFormat(const Image& img) {
    DXGI_FORMAT format;

    if (img.compressed) {
        switch (img.compress_format) {
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
    } else {
        switch (img.bitcount) {
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

void D3d12GraphicsManager::CreateTexture(SceneObjectTexture& texture) {
    HRESULT hr = S_OK;

    const auto& pImage = texture.GetTextureImage();

    // Describe and create a Texture2D.
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    DXGI_FORMAT format = getDxgiFormat(*pImage);

    D3D12_RESOURCE_DESC textureDesc{};
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
                   &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&pTextureBuffer)))) {
        assert(0);
        return;
    }

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

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&pTextureUploadHeap)))) {
        assert(0);
        return;
    }

    // Copy data to the intermediate upload heap and then schedule a copy
    // from the upload heap to the Texture2D.
    D3D12_SUBRESOURCE_DATA textureData{};
    textureData.pData = pImage->data;
    textureData.RowPitch = pImage->pitch;
    textureData.SlicePitch = static_cast<uint64_t>(pImage->pitch) *
                             static_cast<uint64_t>(pImage->Height);

    UpdateSubresources(m_pCopyCommandList, pTextureBuffer, pTextureUploadHeap,
                       0, 0, subresourceCount, &textureData);

    m_Buffers.push_back(pTextureUploadHeap);
    m_Textures.emplace(texture.GetName(),
                       reinterpret_cast<intptr_t>(pTextureBuffer));
}

uint32_t D3d12GraphicsManager::CreateSamplerBuffer() {
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
    for (int32_t i = 0; i < 8; i++) {
        D3D12_CPU_DESCRIPTOR_HANDLE samplerHandle;
        samplerHandle.ptr =
            m_pSamplerHeap->GetCPUDescriptorHandleForHeapStart().ptr +
            static_cast<int64_t>(i) * m_nSamplerDescriptorSize;
        m_pDev->CreateSampler(&samplerDesc, samplerHandle);
    }

    return S_OK;
}

uint32_t D3d12GraphicsManager::CreateConstantBuffer() {
    HRESULT hr;

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

    D3D12_RANGE readRange = {0, 0};

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        resourceDesc.Width = kSizePerFrameConstantBuffer;

        if (FAILED(hr = m_pDev->CreateCommittedResource(
                       &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                       IID_PPV_ARGS(&m_pPerFrameConstantUploadBuffer[i])))) {
            return hr;
        }

        hr = m_pPerFrameConstantUploadBuffer[i]->Map(
            0, &readRange,
            reinterpret_cast<void**>(&m_pPerFrameCbvDataBegin[i]));
        m_pPerFrameConstantUploadBuffer[i]->SetName(
            L"Per Frame Constant Buffer");

        resourceDesc.Width = kSizeLightInfo;

        if (FAILED(hr = m_pDev->CreateCommittedResource(
                       &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                       IID_PPV_ARGS(&m_pLightInfoUploadBuffer[i])))) {
            return hr;
        }

        hr = m_pLightInfoUploadBuffer[i]->Map(
            0, &readRange, reinterpret_cast<void**>(&m_pLightInfoBegin[i]));
        m_pLightInfoUploadBuffer[i]->SetName(L"Light Info Buffer");

#ifdef DEBUG
        resourceDesc.Width = kSizeDebugConstantBuffer;

        if (FAILED(hr = m_pDev->CreateCommittedResource(
                       &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                       IID_PPV_ARGS(&m_pDebugConstantsUploadBuffer[i])))) {
            return hr;
        }

        hr = m_pDebugConstantsUploadBuffer[i]->Map(
            0, &readRange,
            reinterpret_cast<void**>(&m_pDebugConstantsBegin[i]));
        m_pDebugConstantsUploadBuffer[i]->SetName(L"Debug Constants Buffer");
#endif
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateGraphicsResources() {
    HRESULT hr;

#if defined(D3D12_RHI_DEBUG)
    // Enable the D3D12 debug layer.
    {
        if (SUCCEEDED(
                D3D12GetDebugInterface(IID_PPV_ARGS(&m_pDebugController)))) {
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

    if (FAILED(hr = D3D12CreateDevice(pHardwareAdapter, D3D_FEATURE_LEVEL_12_0,
                                      IID_PPV_ARGS(&m_pDev)))) {
        IDXGIAdapter* pWarpAdapter;
        if (FAILED(
                hr = pFactory->EnumWarpAdapter(IID_PPV_ARGS(&pWarpAdapter)))) {
            SafeRelease(&pFactory);
            return hr;
        }

        if (FAILED(hr = D3D12CreateDevice(pWarpAdapter, D3D_FEATURE_LEVEL_12_0,
                                          IID_PPV_ARGS(&m_pDev)))) {
            SafeRelease(&pFactory);
            return hr;
        }
    }

#if defined(D3D12_RHI_DEBUG)
    if (m_pDebugController) {
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

    static const D3D_FEATURE_LEVEL s_featureLevels[] = {D3D_FEATURE_LEVEL_12_1,
                                                        D3D_FEATURE_LEVEL_12_0};

    D3D12_FEATURE_DATA_FEATURE_LEVELS featLevels = {
        _countof(s_featureLevels), s_featureLevels, D3D_FEATURE_LEVEL_12_0};

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
    hr = m_pDev->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &featLevels,
                                     sizeof(featLevels));
    if (SUCCEEDED(hr)) {
        featureLevel = featLevels.MaxSupportedFeatureLevel;
        switch (featureLevel) {
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
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    if (FAILED(hr = m_pDev->CreateCommandQueue(
                   &queueDesc, IID_PPV_ARGS(&m_pGraphicsCommandQueue)))) {
        assert(0);
        return hr;
    }

    m_pGraphicsCommandQueue->SetName(L"Graphics Command Queue");

    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;

    if (FAILED(hr = m_pDev->CreateCommandQueue(
                   &queueDesc, IID_PPV_ARGS(&m_pComputeCommandQueue)))) {
        assert(0);
        return hr;
    }

    m_pComputeCommandQueue->SetName(L"Compute Command Queue");

    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;

    if (FAILED(hr = m_pDev->CreateCommandQueue(
                   &queueDesc, IID_PPV_ARGS(&m_pCopyCommandQueue)))) {
        assert(0);
        return hr;
    }

    m_pCopyCommandQueue->SetName(L"Copy Command Queue");

    // create a struct to hold information about the swap chain
    DXGI_SWAP_CHAIN_DESC1 scd;

    // clear out the struct for use
    ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC1));

    // fill the swap chain description struct
    scd.Width = g_pApp->GetConfiguration().screenWidth;
    scd.Height = g_pApp->GetConfiguration().screenHeight;
    scd.Format = ::DXGI_FORMAT_R8G8B8A8_UNORM;  // use 32-bit color
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
    if (FAILED(hr = pFactory->CreateSwapChainForHwnd(
                   m_pGraphicsCommandQueue,  // Swap chain needs the queue so
                                             // that it can force a flush on it
                   hWnd, &scd, NULL, NULL, &pSwapChain))) {
        assert(0);
        return hr;
    }

    SafeRelease(&pFactory);

    m_pSwapChain = reinterpret_cast<IDXGISwapChain3*>(pSwapChain);

    m_nFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        if (FAILED(
                hr = m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                         IID_PPV_ARGS(&m_pGraphicsFence[i])))) {
            return hr;
        }
    }

    memset(m_nGraphicsFenceValue, 0, sizeof(m_nGraphicsFenceValue));

    m_hGraphicsFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (m_hGraphicsFenceEvent == NULL) {
        hr = HRESULT_FROM_WIN32(GetLastError());
        if (FAILED(hr)) return hr;
    }

    m_hComputeFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (m_hComputeFenceEvent == NULL) {
        hr = HRESULT_FROM_WIN32(GetLastError());
        if (FAILED(hr)) return hr;
    }

    m_hCopyFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (m_hCopyFenceEvent == NULL) {
        hr = HRESULT_FROM_WIN32(GetLastError());
        if (FAILED(hr)) return hr;
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

static std::wstring s2ws(const std::string& s) {
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

    if (pipelineState.pipelineType == PIPELINE_TYPE::GRAPHIC) {
        D3D12_SHADER_BYTECODE vertexShaderByteCode;
        vertexShaderByteCode.pShaderBytecode =
            pipelineState.vertexShaderByteCode.pShaderBytecode;
        vertexShaderByteCode.BytecodeLength =
            pipelineState.vertexShaderByteCode.BytecodeLength;

        D3D12_SHADER_BYTECODE pixelShaderByteCode;
        pixelShaderByteCode.pShaderBytecode =
            pipelineState.pixelShaderByteCode.pShaderBytecode;
        pixelShaderByteCode.BytecodeLength =
            pipelineState.pixelShaderByteCode.BytecodeLength;

        // create rasterizer descriptor
        D3D12_RASTERIZER_DESC rsd{D3D12_FILL_MODE_SOLID,
                                  D3D12_CULL_MODE_BACK,
                                  TRUE,
                                  D3D12_DEFAULT_DEPTH_BIAS,
                                  D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                                  TRUE,
                                  FALSE,
                                  FALSE,
                                  0,
                                  D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF};

        switch (pipelineState.cullFaceMode) {
            case CULL_FACE_MODE::FRONT:
                rsd.CullMode = D3D12_CULL_MODE_FRONT;
                break;
            case CULL_FACE_MODE::BACK:
                rsd.CullMode = D3D12_CULL_MODE_BACK;
                break;
            case CULL_FACE_MODE::NONE:
                rsd.CullMode = D3D12_CULL_MODE_NONE;
                break;
            default:
                assert(0);
        }

        const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlend{
            FALSE,
            FALSE,
            D3D12_BLEND_ONE,
            D3D12_BLEND_ZERO,
            D3D12_BLEND_OP_ADD,
            D3D12_BLEND_ONE,
            D3D12_BLEND_ZERO,
            D3D12_BLEND_OP_ADD,
            D3D12_LOGIC_OP_NOOP,
            D3D12_COLOR_WRITE_ENABLE_ALL};

        const D3D12_BLEND_DESC bld{FALSE,
                                   FALSE,
                                   {
                                       defaultRenderTargetBlend,
                                       defaultRenderTargetBlend,
                                       defaultRenderTargetBlend,
                                       defaultRenderTargetBlend,
                                       defaultRenderTargetBlend,
                                       defaultRenderTargetBlend,
                                       defaultRenderTargetBlend,
                                   }};

        static const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp{
            D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
            D3D12_COMPARISON_FUNC_ALWAYS};

        D3D12_DEPTH_STENCIL_DESC dsd{TRUE,
                                     D3D12_DEPTH_WRITE_MASK_ALL,
                                     D3D12_COMPARISON_FUNC_LESS,
                                     FALSE,
                                     D3D12_DEFAULT_STENCIL_READ_MASK,
                                     D3D12_DEFAULT_STENCIL_WRITE_MASK,
                                     defaultStencilOp,
                                     defaultStencilOp};

        if (pipelineState.bDepthWrite) {
            dsd.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        } else {
            dsd.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        }

        if (pipelineState.depthTestMode == DEPTH_TEST_MODE::NONE) {
            dsd.DepthEnable = FALSE;
        } else {
            dsd.DepthEnable = TRUE;
            switch (pipelineState.depthTestMode) {
                case DEPTH_TEST_MODE::ALWAYS:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
                    break;
                case DEPTH_TEST_MODE::EQUAL:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
                    break;
                case DEPTH_TEST_MODE::LARGE:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
                    break;
                case DEPTH_TEST_MODE::LARGE_EQUAL:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
                    break;
                case DEPTH_TEST_MODE::LESS:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
                    break;
                case DEPTH_TEST_MODE::LESS_EQUAL:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
                    break;
                case DEPTH_TEST_MODE::NEVER:
                    dsd.DepthFunc = D3D12_COMPARISON_FUNC_NEVER;
                    break;
                default:
                    assert(0);
            }
        }

        // create the root signature
        if (FAILED(hr = m_pDev->CreateRootSignature(
                       0, pixelShaderByteCode.pShaderBytecode,
                       pixelShaderByteCode.BytecodeLength,
                       IID_PPV_ARGS(&pipelineState.rootSignature)))) {
            return false;
        }

        // create the input layout object
        static const D3D12_INPUT_ELEMENT_DESC ied_full[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 1, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TANGENT", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 3, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        static const D3D12_INPUT_ELEMENT_DESC ied_simple[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        static const D3D12_INPUT_ELEMENT_DESC ied_cube[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 3, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        static const D3D12_INPUT_ELEMENT_DESC ied_pos_only[]{
            {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        // describe and create the graphics pipeline state object (PSO)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psod{};
        psod.pRootSignature = pipelineState.rootSignature;
        psod.VS = vertexShaderByteCode;
        psod.PS = pixelShaderByteCode;
        psod.BlendState = bld;
        psod.SampleMask = UINT_MAX;
        psod.RasterizerState = rsd;
        psod.DepthStencilState = dsd;
        switch (pipelineState.a2vType) {
            case A2V_TYPES::A2V_TYPES_FULL:
                psod.InputLayout = {ied_full, _countof(ied_full)};
                break;
            case A2V_TYPES::A2V_TYPES_SIMPLE:
                psod.InputLayout = {ied_simple, _countof(ied_simple)};
                break;
            case A2V_TYPES::A2V_TYPES_CUBE:
                psod.InputLayout = {ied_cube, _countof(ied_cube)};
                break;
            case A2V_TYPES::A2V_TYPES_POS_ONLY:
                psod.InputLayout = {ied_pos_only, _countof(ied_pos_only)};
                break;
            default:
                assert(0);
        }

        psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        if (pipelineState.flag == PIPELINE_FLAG::SHADOW) {
            psod.NumRenderTargets = 0;
            psod.RTVFormats[0] = ::DXGI_FORMAT_UNKNOWN;
            psod.SampleDesc.Count = 1;
            psod.SampleDesc.Quality = 0;
        } else {
            psod.NumRenderTargets = 1;
            psod.RTVFormats[0] = ::DXGI_FORMAT_R8G8B8A8_UNORM;
            psod.SampleDesc.Count = 4;  // 4X MSAA
            psod.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
        }
        psod.DSVFormat = ::DXGI_FORMAT_D32_FLOAT;

        if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(
                       &psod, IID_PPV_ARGS(&pipelineState.pipelineState)))) {
            return false;
        }
    } else {
        assert(pipelineState.pipelineType == PIPELINE_TYPE::COMPUTE);

        D3D12_SHADER_BYTECODE computeShaderByteCode;
        computeShaderByteCode.pShaderBytecode =
            pipelineState.computeShaderByteCode.pShaderBytecode;
        computeShaderByteCode.BytecodeLength =
            pipelineState.computeShaderByteCode.BytecodeLength;

        // create the root signature
        if (FAILED(hr = m_pDev->CreateRootSignature(
                       0, computeShaderByteCode.pShaderBytecode,
                       computeShaderByteCode.BytecodeLength,
                       IID_PPV_ARGS(&pipelineState.rootSignature)))) {
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

        if (FAILED(hr = m_pDev->CreateComputePipelineState(
                       &psod, IID_PPV_ARGS(&pipelineState.pipelineState)))) {
            return false;
        }
    }
    pipelineState.pipelineState->SetName(
        s2ws(pipelineState.pipelineStateName).c_str());

    return hr;
}

HRESULT D3d12GraphicsManager::CreateCommandList() {
    HRESULT hr = S_OK;

    // Graphics
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        if (FAILED(hr = m_pDev->CreateCommandAllocator(
                       D3D12_COMMAND_LIST_TYPE_DIRECT,
                       IID_PPV_ARGS(&m_pGraphicsCommandAllocator[i])))) {
            assert(0);
            return hr;
        }
        m_pGraphicsCommandAllocator[i]->SetName(
            (wstring(L"Graphics Command Allocator") + to_wstring(i)).c_str());

        hr = m_pDev->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_pGraphicsCommandAllocator[i],
            NULL, IID_PPV_ARGS(&m_pGraphicsCommandList[i]));

        if (SUCCEEDED(hr)) {
            m_pGraphicsCommandList[i]->SetName(
                (wstring(L"Graphics Command List") + to_wstring(i)).c_str());
        }
    }

    // Compute
    if (FAILED(hr = m_pDev->CreateCommandAllocator(
                   D3D12_COMMAND_LIST_TYPE_COMPUTE,
                   IID_PPV_ARGS(&m_pComputeCommandAllocator)))) {
        assert(0);
        return hr;
    }

    m_pComputeCommandAllocator->SetName(L"Compute Command Allocator");

    hr = m_pDev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                   m_pComputeCommandAllocator, NULL,
                                   IID_PPV_ARGS(&m_pComputeCommandList));

    if (SUCCEEDED(hr)) {
        m_pComputeCommandList->SetName(L"Compute Command List");
    }

    // Copy
    if (FAILED(hr = m_pDev->CreateCommandAllocator(
                   D3D12_COMMAND_LIST_TYPE_COPY,
                   IID_PPV_ARGS(&m_pCopyCommandAllocator)))) {
        assert(0);
        return hr;
    }

    m_pCopyCommandAllocator->SetName(L"Copy Command Allocator");

    hr = m_pDev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY,
                                   m_pCopyCommandAllocator, NULL,
                                   IID_PPV_ARGS(&m_pCopyCommandList));

    if (SUCCEEDED(hr)) {
        m_pCopyCommandList->SetName(L"Copy Command List");
    }

    return hr;
}

void D3d12GraphicsManager::initializeGeometries(const Scene& scene) {
    cout << "Creating Draw Batch Contexts ...";
    uint32_t batch_index = 0;
    for (const auto& _it : scene.GeometryNodes) {
        const auto& pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible()) {
            const auto& pGeometry =
                scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            const auto& pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            // Set the number of vertex properties.
            const auto vertexPropertiesCount =
                pMesh->GetVertexPropertiesCount();

            // Set the number of vertices in the vertex array.
            const auto vertexCount = pMesh->GetVertexCount();

            auto dbc = make_shared<D3dDrawBatchContext>();

            for (uint32_t i = 0; i < vertexPropertiesCount; i++) {
                const SceneObjectVertexArray& v_property_array =
                    pMesh->GetVertexPropertyArray(i);

                auto offset = CreateVertexBuffer(v_property_array);
                if (i == 0) {
                    dbc->property_offset = offset;
                }
            }

            const SceneObjectIndexArray& index_array = pMesh->GetIndexArray(0);
            dbc->index_offset = CreateIndexBuffer(index_array);

            const auto material_index = index_array.GetMaterialIndex();
            const auto material_key =
                pGeometryNode->GetMaterialRef(material_index);
            const auto& material = scene.GetMaterial(material_key);

            dbc->batchIndex = batch_index++;
            dbc->index_count = (UINT)index_array.GetIndexCount();
            dbc->property_count = vertexPropertiesCount;

            // load material textures
            dbc->cbv_srv_uav_offset =
                (size_t)dbc->batchIndex * 32 * m_nCbvSrvUavDescriptorSize;
            D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle =
                m_pCbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
            srvCpuHandle.ptr += dbc->cbv_srv_uav_offset;

            // Jump over per batch CBVs
            srvCpuHandle.ptr += 2 * m_nCbvSrvUavDescriptorSize;

            // SRV
            if (material) {
                if (auto& texture = material->GetBaseColor().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetNormal().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetMetallic().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetRoughness().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }

                srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

                if (auto& texture = material->GetAO().ValueMap) {
                    CreateTexture(*texture);

                    m_pDev->CreateShaderResourceView(
                        reinterpret_cast<ID3D12Resource*>(
                            m_Textures[texture->GetName()]),
                        NULL, srvCpuHandle);
                }
            }

            // UAV
            // ; temporary nothing here

            dbc->node = pGeometryNode;

            for (auto& frame : m_Frames) {
                frame.batchContexts.push_back(dbc);
            }
        }
    }
    cout << "Done!" << endl;
}

void D3d12GraphicsManager::initializeSkyBox(const Scene& scene) {
    HRESULT hr = S_OK;

    assert(scene.SkyBox);

    m_dbcSkyBox.property_offset = CreateVertexBuffer(
        skyboxVertices, sizeof(skyboxVertices), sizeof(skyboxVertices[0]));
    m_dbcSkyBox.property_count = 1;
    m_dbcSkyBox.index_offset =
        CreateIndexBuffer(skyboxIndices, sizeof(skyboxIndices),
                          static_cast<int32_t>(sizeof(skyboxIndices[0])));
    m_dbcSkyBox.index_count = sizeof(skyboxIndices) / sizeof(skyboxIndices[0]);

    // Describe and create a Cubemap.
    auto& texture = scene.SkyBox->GetTexture(0);
    const auto& pImage = texture.GetTextureImage();
    DXGI_FORMAT format = getDxgiFormat(*pImage);

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
                   &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&pTextureBuffer)))) {
        return;
    }

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

    ID3D12Resource* pTextureUploadHeap;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&pTextureUploadHeap)))) {
        return;
    }

    // skybox, irradiance map
    for (uint32_t i = 0; i < 6; i++) {
        auto& texture = scene.SkyBox->GetTexture(i);
        const auto& pImage = texture.GetTextureImage();

        // Copy data to the intermediate upload heap and then schedule a copy
        // from the upload heap to the Texture2D.
        D3D12_SUBRESOURCE_DATA textureData{};
        textureData.pData = pImage->data;
        textureData.RowPitch = pImage->pitch;
        textureData.SlicePitch = static_cast<uint64_t>(pImage->pitch) *
                                 static_cast<uint64_t>(pImage->Height);

        UpdateSubresources(m_pCopyCommandList, pTextureBuffer,
                           pTextureUploadHeap, 0, i, 1, &textureData);
    }

    m_Buffers.push_back(pTextureUploadHeap);

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        m_Frames[i].skybox = static_cast<int32_t>(m_Textures.size());
    }

    m_Textures.emplace("SKYBOX", reinterpret_cast<intptr_t>(pTextureBuffer));
}

void D3d12GraphicsManager::BeginScene(const Scene& scene) {
    GraphicsManager::BeginScene(scene);

    if (SUCCEEDED(m_pCopyCommandList->Close())) {
        ID3D12CommandList* ppCommandLists[] = {m_pCopyCommandList};
        m_pCopyCommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                                 ppCommandLists);
    }

    ID3D12Fence* pCopyQueueFence;
    if (FAILED(m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                   IID_PPV_ARGS(&pCopyQueueFence)))) {
        assert(0);
    }

    if (FAILED(m_pCopyCommandQueue->Signal(pCopyQueueFence, 1))) {
        assert(0);
    }

    if (FAILED(
            pCopyQueueFence->SetEventOnCompletion(1, m_hGraphicsFenceEvent))) {
        assert(0);
    }
    WaitForSingleObject(m_hGraphicsFenceEvent, INFINITE);

    SafeRelease(&pCopyQueueFence);

    m_pCopyCommandAllocator->Reset();
    m_pCopyCommandList->Reset(m_pCopyCommandAllocator, nullptr);
}

void D3d12GraphicsManager::EndScene() {
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        WaitForPreviousFrame(i);
    }

    for (auto& p : m_Buffers) {
        SafeRelease(&p);
    }
    m_Buffers.clear();
    m_VertexBufferView.clear();
    m_IndexBufferView.clear();

    GraphicsManager::EndScene();
}

void D3d12GraphicsManager::BeginFrame(const Frame& frame) {
    GraphicsManager::BeginFrame(frame);
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();

    SetPerFrameConstants(frame);
    SetLightInfo(frame);

    assert(frame.frameIndex == m_nFrameIndex);
    if (FAILED(WaitForPreviousFrame(frame.frameIndex))) {
        assert(0);
    }
}

void D3d12GraphicsManager::EndFrame(const Frame& frame) {
    HRESULT hr;

    MsaaResolve();

    // now draw GUI overlay
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    // bind the final RTV
    rtvHandle =
        m_pRtvHeap[frame.frameIndex]->GetCPUDescriptorHandleForHeapStart();
    m_pGraphicsCommandList[m_nFrameIndex]->OMSetRenderTargets(1, &rtvHandle,
                                                              FALSE, nullptr);

    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(),
                                  m_pGraphicsCommandList[frame.frameIndex]);

    D3D12_RESOURCE_BARRIER barrier;

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    m_pGraphicsCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    if (SUCCEEDED(hr = m_pGraphicsCommandList[frame.frameIndex]->Close())) {
        m_nGraphicsFenceValue[frame.frameIndex]++;

        ID3D12CommandList* ppCommandLists[] = {
            m_pGraphicsCommandList[frame.frameIndex]};
        m_pGraphicsCommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                                     ppCommandLists);

        const uint64_t fence = m_nGraphicsFenceValue[frame.frameIndex];
        if (FAILED(hr = m_pGraphicsCommandQueue->Signal(
                       m_pGraphicsFence[frame.frameIndex], fence))) {
            assert(0);
        }
    }
}

void D3d12GraphicsManager::BeginPass(const Frame& frame) {
    m_pGraphicsCommandList[m_nFrameIndex]->RSSetViewports(1, &m_ViewPort);
    m_pGraphicsCommandList[m_nFrameIndex]->RSSetScissorRects(1, &m_ScissorRect);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    // bind the MSAA RTV and DSV
    rtvHandle.ptr =
        m_pRtvHeap[frame.frameIndex]->GetCPUDescriptorHandleForHeapStart().ptr +
        m_nRtvDescriptorSize;
    dsvHandle =
        m_pDsvHeap[frame.frameIndex]->GetCPUDescriptorHandleForHeapStart();
    m_pGraphicsCommandList[m_nFrameIndex]->OMSetRenderTargets(
        1, &rtvHandle, FALSE, &dsvHandle);

    ID3D12DescriptorHeap* ppHeaps[] = {m_pCbvSrvUavHeap, m_pSamplerHeap};
    m_pGraphicsCommandList[frame.frameIndex]->SetDescriptorHeaps(
        static_cast<int32_t>(_countof(ppHeaps)), ppHeaps);

    // clear the back buffer to a deep blue
    const FLOAT clearColor[] = {0.2f, 0.3f, 0.4f, 1.0f};
    m_pGraphicsCommandList[m_nFrameIndex]->ClearRenderTargetView(
        rtvHandle, clearColor, 0, nullptr);
    m_pGraphicsCommandList[m_nFrameIndex]->ClearDepthStencilView(
        dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

void D3d12GraphicsManager::Draw() { GraphicsManager::Draw(); }

void D3d12GraphicsManager::DrawBatch(const Frame& frame) {
    for (const auto& pDbc : frame.batchContexts) {
        const D3dDrawBatchContext& dbc =
            dynamic_cast<const D3dDrawBatchContext&>(*pDbc);

        // select which vertex buffer(s) to use
        for (uint32_t i = 0; i < dbc.property_count; i++) {
            m_pGraphicsCommandList[frame.frameIndex]->IASetVertexBuffers(
                i, 1, &m_VertexBufferView[dbc.property_offset + i]);
        }

        // select which index buffer to use
        m_pGraphicsCommandList[frame.frameIndex]->IASetIndexBuffer(
            &m_IndexBufferView[dbc.index_offset]);

        // set primitive topology
        m_pGraphicsCommandList[frame.frameIndex]->IASetPrimitiveTopology(
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        // PerFrame CBV (b11)
        m_pGraphicsCommandList[frame.frameIndex]->SetGraphicsRoot32BitConstants(
            1, 16, dbc.modelMatrix, 0);

        // Bind LightInfo (b12)
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation =
            m_pLightInfoUploadBuffer[frame.frameIndex]->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = kSizeLightInfo;

        D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
        cbvHandle = m_pCbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
        cbvHandle.ptr += dbc.cbv_srv_uav_offset;
        m_pDev->CreateConstantBufferView(&cbvDesc, cbvHandle);

        cbvHandle.ptr += 2 * m_nCbvSrvUavDescriptorSize;
        // Bind global textures (t6, t10)
        // D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
        cbvHandle.ptr += 6 * m_nCbvSrvUavDescriptorSize;
        m_pDev->CreateShaderResourceView(
            reinterpret_cast<ID3D12Resource*>(m_Textures["BRDF_LUT"]), NULL,
            cbvHandle);

        cbvHandle.ptr += 4 * m_nCbvSrvUavDescriptorSize;
        m_pDev->CreateShaderResourceView(
            reinterpret_cast<ID3D12Resource*>(m_Textures["SKYBOX"]), NULL,
            cbvHandle);

        // Bind per batch Descriptor Table
        D3D12_GPU_DESCRIPTOR_HANDLE cbvSrvUavGpuHandle =
            m_pCbvSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
        cbvSrvUavGpuHandle.ptr += dbc.cbv_srv_uav_offset;
        m_pGraphicsCommandList[frame.frameIndex]
            ->SetGraphicsRootDescriptorTable(2, cbvSrvUavGpuHandle);

        // Sampler (s0)
        cbvSrvUavGpuHandle =
            m_pSamplerHeap->GetGPUDescriptorHandleForHeapStart();
        m_pGraphicsCommandList[frame.frameIndex]
            ->SetGraphicsRootDescriptorTable(3, cbvSrvUavGpuHandle);

        // draw the vertex buffer to the back buffer
        m_pGraphicsCommandList[frame.frameIndex]->DrawIndexedInstanced(
            dbc.index_count, 1, 0, 0, 0);
    }
}

void D3d12GraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    if (pipelineState) {
        std::shared_ptr<D3d12PipelineState> pState =
            dynamic_pointer_cast<D3d12PipelineState>(pipelineState);

        if (!pState->pipelineState) {
            CreatePSO(*pState);
        }

        switch (pState->pipelineType) {
            case PIPELINE_TYPE::GRAPHIC: {
                m_pGraphicsCommandList[frame.frameIndex]->SetPipelineState(
                    pState->pipelineState);

                m_pGraphicsCommandList[frame.frameIndex]
                    ->SetGraphicsRootSignature(pState->rootSignature);

                // Per Frame CBV (b10)
                m_pGraphicsCommandList[frame.frameIndex]
                    ->SetGraphicsRootConstantBufferView(
                        0, m_pPerFrameConstantUploadBuffer[frame.frameIndex]
                               ->GetGPUVirtualAddress());
            } break;
            case PIPELINE_TYPE::COMPUTE:
                m_pComputeCommandList->SetPipelineState(pState->pipelineState);

                m_pComputeCommandList->SetComputeRootSignature(
                    pState->rootSignature);
                break;
            default:
                assert(0);
        }
    }
}

void D3d12GraphicsManager::SetPerFrameConstants(const Frame& frame) {
    memcpy(m_pPerFrameCbvDataBegin[frame.frameIndex],
           &static_cast<const PerFrameConstants&>(frame.frameContext),
           sizeof(PerFrameConstants));
}

void D3d12GraphicsManager::SetLightInfo(const Frame& frame) {
    memcpy(m_pLightInfoBegin[frame.frameIndex], &frame.lightInfo,
           sizeof(LightInfo));
}

HRESULT D3d12GraphicsManager::MsaaResolve() {
    D3D12_RESOURCE_BARRIER barrier[2];

    barrier[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[0].Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex + 1];
    barrier[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barrier[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[1].Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pGraphicsCommandList[m_nFrameIndex]->ResourceBarrier(2, barrier);

    m_pGraphicsCommandList[m_nFrameIndex]->ResolveSubresource(
        m_pRenderTargets[2 * m_nFrameIndex], 0,
        m_pRenderTargets[2 * m_nFrameIndex + 1], 0,
        ::DXGI_FORMAT_R8G8B8A8_UNORM);

    // Indicate that the back buffer will be used as a resolve source.
    barrier[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[0].Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex + 1];
    barrier[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barrier[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier[1].Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pGraphicsCommandList[m_nFrameIndex]->ResourceBarrier(2, barrier);

    return S_OK;
}

void D3d12GraphicsManager::Present() {
    [[maybe_unused]] HRESULT hr;

    // swap the back buffer and the front buffer
    hr = m_pSwapChain->Present(1, 0);

    m_nFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();
}

void D3d12GraphicsManager::DrawSkyBox(const Frame& frame) {
    // set primitive topology
    m_pGraphicsCommandList[frame.frameIndex]->IASetPrimitiveTopology(
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // set vertex buffer
    m_pGraphicsCommandList[frame.frameIndex]->IASetVertexBuffers(
        0, 1, &m_VertexBufferView[m_dbcSkyBox.property_offset]);

    // set index buffer
    m_pGraphicsCommandList[frame.frameIndex]->IASetIndexBuffer(
        &m_IndexBufferView[m_dbcSkyBox.index_offset]);

    // (t10)
    D3D12_GPU_DESCRIPTOR_HANDLE cbvSrvUavGpuHandle =
        m_pCbvSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
    cbvSrvUavGpuHandle.ptr +=
        12 * m_nCbvSrvUavDescriptorSize;  // always use 1st batch context
    m_pGraphicsCommandList[frame.frameIndex]->SetGraphicsRootDescriptorTable(
        2, cbvSrvUavGpuHandle);

    // Sampler (s0)
    cbvSrvUavGpuHandle = m_pSamplerHeap->GetGPUDescriptorHandleForHeapStart();
    m_pGraphicsCommandList[frame.frameIndex]->SetGraphicsRootDescriptorTable(
        3, cbvSrvUavGpuHandle);

    // draw the vertex buffer to the back buffer
    m_pGraphicsCommandList[m_nFrameIndex]->DrawIndexedInstanced(
        m_dbcSkyBox.index_count, 1, 0, 0, 0);
}

int32_t D3d12GraphicsManager::GenerateCubeShadowMapArray(const uint32_t width,
                                                         const uint32_t height,
                                                         const uint32_t count) {
    int32_t texture_id = 0;

    return texture_id;
}

int32_t D3d12GraphicsManager::GenerateShadowMapArray(const uint32_t width,
                                                     const uint32_t height,
                                                     const uint32_t count) {
    int32_t texture_id = 0;

    return texture_id;
}

void D3d12GraphicsManager::BeginShadowMap(
    const int32_t light_index, const int32_t shadowmap, const uint32_t width,
    const uint32_t height, const int32_t layer_index, const Frame& frame) {
    D3D12_VIEWPORT view_port = {
        0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height),
        0.0f, 1.0f};
    D3D12_RECT scissor_rect = {0, 0, static_cast<LONG>(width),
                               static_cast<LONG>(height)};

    m_pGraphicsCommandList[m_nFrameIndex]->RSSetViewports(1, &view_port);
    m_pGraphicsCommandList[m_nFrameIndex]->RSSetScissorRects(1, &scissor_rect);

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    // bind shadow map DSV
    dsvHandle =
        m_pDsvHeap[frame.frameIndex]->GetCPUDescriptorHandleForHeapStart();
    dsvHandle.ptr += light_index * m_nCbvSrvUavDescriptorSize;
    m_pGraphicsCommandList[m_nFrameIndex]->OMSetRenderTargets(0, nullptr, FALSE,
                                                              &dsvHandle);

    ID3D12DescriptorHeap* ppHeaps[] = {m_pCbvSrvUavHeap, m_pSamplerHeap};
    m_pGraphicsCommandList[frame.frameIndex]->SetDescriptorHeaps(
        static_cast<int32_t>(_countof(ppHeaps)), ppHeaps);

    m_pGraphicsCommandList[m_nFrameIndex]->ClearDepthStencilView(
        dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
}

void D3d12GraphicsManager::EndShadowMap(const int32_t shadowmap,
                                        const int32_t layer_index) {}

void D3d12GraphicsManager::SetShadowMaps(const Frame& frame) {}

void D3d12GraphicsManager::ReleaseTexture(intptr_t texture) {
    ID3D12Resource* pTmp = reinterpret_cast<ID3D12Resource*>(texture);
    SafeRelease(&pTmp);
}

void D3d12GraphicsManager::GenerateTextureForWrite(const char* id,
                                                   const uint32_t width,
                                                   const uint32_t height) {
    // Describe and create a Texture2D.
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    ID3D12Resource* pTextureBuffer;

    if (FAILED(m_pDev->CreateCommittedResource(
            &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&pTextureBuffer)))) {
        assert(0);
        return;
    }

    m_Textures.emplace(id, reinterpret_cast<intptr_t>(pTextureBuffer));
}

void D3d12GraphicsManager::BindTextureForWrite(const char* texture,
                                               const uint32_t slot_index) {
    auto it = m_Textures.find(texture);
    if (it != m_Textures.end()) {
        ID3D12DescriptorHeap* ppHeaps[] = {m_pCbvSrvUavHeap};
        m_pComputeCommandList->SetDescriptorHeaps(
            static_cast<int32_t>(_countof(ppHeaps)), ppHeaps);

        D3D12_CPU_DESCRIPTOR_HANDLE uavCpuHandle;
        uavCpuHandle = m_pCbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
        m_pDev->CreateUnorderedAccessView(
            reinterpret_cast<ID3D12Resource*>(it->second), NULL, NULL,
            uavCpuHandle);

        D3D12_GPU_DESCRIPTOR_HANDLE uavGpuHandle;
        uavGpuHandle = m_pCbvSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
        m_pComputeCommandList->SetComputeRootDescriptorTable(0, uavGpuHandle);
    }
}

void D3d12GraphicsManager::Dispatch(const uint32_t width, const uint32_t height,
                                    const uint32_t depth) {
    m_pComputeCommandList->Dispatch(width, height, depth);
}

void D3d12GraphicsManager::BeginCompute() {}

void D3d12GraphicsManager::EndCompute() {
    if (SUCCEEDED(m_pComputeCommandList->Close())) {
        ID3D12CommandList* ppCommandLists[] = {m_pComputeCommandList};
        m_pComputeCommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                                    ppCommandLists);
    }

    ID3D12Fence* pComputeQueueFence;
    if (FAILED(m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                   IID_PPV_ARGS(&pComputeQueueFence)))) {
        assert(0);
    }

    if (FAILED(m_pComputeCommandQueue->Signal(pComputeQueueFence, 1))) {
        assert(0);
    }

    if (FAILED(pComputeQueueFence->SetEventOnCompletion(
            1, m_hComputeFenceEvent))) {
        assert(0);
    }
    WaitForSingleObject(m_hComputeFenceEvent, INFINITE);

    SafeRelease(&pComputeQueueFence);

    m_pComputeCommandAllocator->Reset();
    m_pComputeCommandList->Reset(m_pComputeCommandAllocator, nullptr);
}

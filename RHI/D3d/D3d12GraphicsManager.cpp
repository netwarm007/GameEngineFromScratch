#include "D3d12GraphicsManager.hpp"

#include <objbase.h>

#include <iostream>

#include "AssetLoader.hpp"
#include "IApplication.hpp"
#include "IPhysicsManager.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

#define D3D12_DEBUG

namespace My {
extern IApplication* g_pApp;

template <class T>
inline void SafeRelease(T** ppInterfaceToRelease) {
    if (*ppInterfaceToRelease != nullptr) {
        (*ppInterfaceToRelease)->Release();

        (*ppInterfaceToRelease) = nullptr;
    }
}

static void GetHardwareAdapter(IDXGIFactory4* pFactory,
                               IDXGIAdapter1** ppAdapter) {
    *ppAdapter = nullptr;
    for (UINT adapterIndex = 0;; ++adapterIndex) {
        IDXGIAdapter1* pAdapter = nullptr;
        if (DXGI_ERROR_NOT_FOUND ==
            pFactory->EnumAdapters1(adapterIndex, &pAdapter)) {
            // No more adapters to enumerate.
            break;
        }

        // Check to see if the adapter supports Direct3D 12, but don't create
        // the actual device yet.
        if (SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_12_0,
                                        _uuidof(ID3D12Device), nullptr))) {
            *ppAdapter = pAdapter;
            return;
        }
        pAdapter->Release();
    }
}

//------------------------------------------------------------------------------------------------
// Row-by-row memcpy
inline void MemcpySubresource(_In_ const D3D12_MEMCPY_DEST* pDest,
                              _In_ const D3D12_SUBRESOURCE_DATA* pSrc,
                              SIZE_T RowSizeInBytes, UINT NumRows,
                              UINT NumSlices) {
    for (UINT z = 0; z < NumSlices; ++z) {
        BYTE* pDestSlice =
            reinterpret_cast<BYTE*>(pDest->pData) + pDest->SlicePitch * z;
        const BYTE* pSrcSlice =
            reinterpret_cast<const BYTE*>(pSrc->pData) + pSrc->SlicePitch * z;
        for (UINT y = 0; y < NumRows; ++y) {
            memcpy(pDestSlice + pDest->RowPitch * y,
                   pSrcSlice + pSrc->RowPitch * y, RowSizeInBytes);
        }
    }
}

// ******************************************
// following utility code copied from
// d3dx12.h
// Code is custimized by Tim Chen on 2017/12/20
// MIT license
// Original licensed as following:

//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

//------------------------------------------------------------------------------------------------
// Returns required size of a buffer to be used for data upload
inline UINT64 GetRequiredIntermediateSize(
    _In_ ID3D12Resource* pDestinationResource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource)
        UINT NumSubresources) {
    D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
    UINT64 RequiredSize = 0;

    ID3D12Device* pDevice;
    pDestinationResource->GetDevice(__uuidof(*pDevice),
                                    reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, 0,
                                   nullptr, nullptr, nullptr, &RequiredSize);
    pDevice->Release();

    return RequiredSize;
}

//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource)
        UINT NumSubresources,
    UINT64 RequiredSize,
    _In_reads_(NumSubresources)
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
    _In_reads_(NumSubresources) const UINT* pNumRows,
    _In_reads_(NumSubresources) const UINT64* pRowSizesInBytes,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData) {
    // Minor validation
    D3D12_RESOURCE_DESC IntermediateDesc = pIntermediate->GetDesc();
    D3D12_RESOURCE_DESC DestinationDesc = pDestinationResource->GetDesc();
    if (IntermediateDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        IntermediateDesc.Width < RequiredSize + pLayouts[0].Offset ||
        RequiredSize > (SIZE_T)-1 ||
        (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
         (FirstSubresource != 0 || NumSubresources != 1))) {
        return 0;
    }

    BYTE* pData;
    HRESULT hr = pIntermediate->Map(0, NULL, reinterpret_cast<void**>(&pData));
    if (FAILED(hr)) {
        return 0;
    }

    for (UINT i = 0; i < NumSubresources; ++i) {
        if (pRowSizesInBytes[i] > (SIZE_T)-1) return 0;
        D3D12_MEMCPY_DEST DestData = {
            pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch,
            pLayouts[i].Footprint.RowPitch * pNumRows[i]};
        MemcpySubresource(&DestData, &pSrcData[i], (SIZE_T)pRowSizesInBytes[i],
                          pNumRows[i], pLayouts[i].Footprint.Depth);
    }
    pIntermediate->Unmap(0, NULL);

    if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        pCmdList->CopyBufferRegion(pDestinationResource, 0, pIntermediate,
                                   pLayouts[0].Offset,
                                   pLayouts[0].Footprint.Width);
    } else {
        for (UINT i = 0; i < NumSubresources; ++i) {
            D3D12_TEXTURE_COPY_LOCATION Dst = {
                pDestinationResource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                i + FirstSubresource};
            D3D12_TEXTURE_COPY_LOCATION Src = {
                pIntermediate, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                pLayouts[i]};
            pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
        }
    }

    return RequiredSize;
}

//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate, UINT64 IntermediateOffset,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource)
        UINT NumSubresources,
    _In_reads_(NumSubresources) D3D12_SUBRESOURCE_DATA* pSrcData) {
    UINT64 RequiredSize = 0;
    UINT64 MemToAlloc =
        static_cast<UINT64>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) +
                            sizeof(UINT) + sizeof(UINT64)) *
        NumSubresources;
    if (MemToAlloc > SIZE_MAX) {
        return 0;
    }
    void* pMem =
        HeapAlloc(GetProcessHeap(), 0, static_cast<SIZE_T>(MemToAlloc));
    if (pMem == NULL) {
        return 0;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts =
        reinterpret_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(pMem);
    UINT64* pRowSizesInBytes =
        reinterpret_cast<UINT64*>(pLayouts + NumSubresources);
    UINT* pNumRows =
        reinterpret_cast<UINT*>(pRowSizesInBytes + NumSubresources);

    D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
    ID3D12Device* pDevice;
    pDestinationResource->GetDevice(__uuidof(*pDevice),
                                    reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources,
                                   IntermediateOffset, pLayouts, pNumRows,
                                   pRowSizesInBytes, &RequiredSize);
    SafeRelease(&pDevice);

    UINT64 Result =
        UpdateSubresources(pCmdList, pDestinationResource, pIntermediate,
                           FirstSubresource, NumSubresources, RequiredSize,
                           pLayouts, pNumRows, pRowSizesInBytes, pSrcData);
    HeapFree(GetProcessHeap(), 0, pMem);
    return Result;
}

//------------------------------------------------------------------------------------------------
// Stack-allocating UpdateSubresources implementation
template <UINT MaxSubresources>
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate, UINT64 IntermediateOffset,
    _In_range_(0, MaxSubresources) UINT FirstSubresource,
    _In_range_(1, MaxSubresources - FirstSubresource) UINT NumSubresources,
    _In_reads_(NumSubresources) D3D12_SUBRESOURCE_DATA* pSrcData) {
    UINT64 RequiredSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT Layouts[MaxSubresources];
    UINT NumRows[MaxSubresources];
    UINT64 RowSizesInBytes[MaxSubresources];

    D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
    ID3D12Device* pDevice;
    pDestinationResource->GetDevice(__uuidof(*pDevice),
                                    reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources,
                                   IntermediateOffset, Layouts, NumRows,
                                   RowSizesInBytes, &RequiredSize);
    SafeRelease(&pDevice);

    return UpdateSubresources(pCmdList, pDestinationResource, pIntermediate,
                              FirstSubresource, NumSubresources, RequiredSize,
                              Layouts, NumRows, RowSizesInBytes, pSrcData);
}

// ************************************************
// Code copied from d3dx12.h finished here
// ************************************************
}  // namespace My

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

    return result;
}

void D3d12GraphicsManager::Finalize() {
    GraphicsManager::Finalize();

    for (int i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        SafeRelease(&m_pFence[i]);
    }

    SafeRelease(&m_pRtvHeap);
    SafeRelease(&m_pDsvHeap);
    SafeRelease(&m_pSamplerHeap);
    SafeRelease(&m_pSrvHeap);
    SafeRelease(&m_pCbvHeap);
    SafeRelease(&m_pPerBatchSrvRingHeap);
    SafeRelease(&m_pRootSignature);
    SafeRelease(&m_pDepthStencilBuffer);
    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        SafeRelease(&m_pCommandList[i]);
        SafeRelease(&m_pCommandAllocator[i]);
        SafeRelease(&m_pPerFrameConstantUploadBuffer[i]);
        SafeRelease(&m_pLightDataUploadBuffer[i]);
        SafeRelease(&m_pRenderTargets[i << 1]);
        SafeRelease(&m_pRenderTargets[(i << 1) | 1]);
    }
    SafeRelease(&m_pCommandQueue);
    for (vector<ID3D12PipelineState*>::iterator it = m_pPipelineStates.begin();
         it != m_pPipelineStates.end(); it++) {
        SafeRelease(&*it);
    }
    m_pPipelineStates.clear();
    SafeRelease(&m_pSwapChain);

    SafeRelease(&m_pDev);

#if defined(D3D12_DEBUG_LAYER)
    if (m_pDebugDev) {
        m_pDebugDev->ReportLiveDeviceObjects(D3D12_RLDO_DETAIL);
    }

    SafeRelease(&m_pDebugDev);
    SafeRelease(&m_pDebugController);
#endif
}

HRESULT D3d12GraphicsManager::WaitForPreviousFrame() {
    // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
    // This is code implemented as such for simplicity. More advanced samples
    // illustrate how to use fences for efficient resource usage.

    // Signal and increment the fence value.
    HRESULT hr;
    const uint64_t fence = m_nFenceValue[m_nFrameIndex];
    if (FAILED(hr = m_pCommandQueue->Signal(m_pFence[m_nFrameIndex], fence))) {
        return hr;
    }

    m_nFenceValue[m_nFrameIndex]++;

    // Wait until the previous frame is finished.
    if (m_pFence[m_nFrameIndex]->GetCompletedValue() < fence) {
        if (FAILED(hr = m_pFence[m_nFrameIndex]->SetEventOnCompletion(
                       fence, m_hFenceEvent))) {
            return hr;
        }
        WaitForSingleObject(m_hFenceEvent, INFINITE);
    }

    m_nFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDescriptorHeaps() {
    HRESULT hr;

    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors =
        2 * GfxConfiguration::kMaxInFlightFrameCount;  // 1 for present + 1 for
                                                       // MSAA Resolver
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(hr = m_pDev->CreateDescriptorHeap(&rtvHeapDesc,
                                                 IID_PPV_ARGS(&m_pRtvHeap)))) {
        return hr;
    }
    m_pRtvHeap->SetName(L"RTV Heap");

    m_nRtvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Describe and create a depth stencil view (DSV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(hr = m_pDev->CreateDescriptorHeap(&dsvHeapDesc,
                                                 IID_PPV_ARGS(&m_pDsvHeap)))) {
        return hr;
    }
    m_pDsvHeap->SetName(L"DSV Heap");

    // Describe and create a Constant Buffer View (CBV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc = {};
    cbvHeapDesc.NumDescriptors = GfxConfiguration::kMaxInFlightFrameCount;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(hr = m_pDev->CreateDescriptorHeap(&cbvHeapDesc,
                                                 IID_PPV_ARGS(&m_pCbvHeap)))) {
        return hr;
    }
    m_pCbvHeap->SetName(L"CBV Heap");

    // Describe and create a shader-non-visible Shader Resource View (SRV)
    // descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = GfxConfiguration::kMaxTexturePerMaterialCount *
                                 GfxConfiguration::kMaxSceneObjectCount;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(hr = m_pDev->CreateDescriptorHeap(&srvHeapDesc,
                                                 IID_PPV_ARGS(&m_pSrvHeap)))) {
        return hr;
    }
    m_pSrvHeap->SetName(L"Offline SRV Heap");

    m_nCbvSrvUavDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Describe and create a sampler descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
    samplerHeapDesc.NumDescriptors =
        8;  // this is the max D3d12 HW support currently
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(hr = m_pDev->CreateDescriptorHeap(
                   &samplerHeapDesc, IID_PPV_ARGS(&m_pSamplerHeap)))) {
        return hr;
    }
    m_pSamplerHeap->SetName(L"Sample Heap");

    m_nSamplerDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    // Describe and create a ring descriptor heap for per batch SRV heap
    D3D12_DESCRIPTOR_HEAP_DESC perBatchSrvHeapDesc = {};
    m_nPerBatchSrvRingHeapSize = GfxConfiguration::kMaxTexturePerMaterialCount *
                                 GfxConfiguration::kMaxSceneObjectCount *
                                 GfxConfiguration::kMaxInFlightFrameCount;
    m_nPerBatchSrvRingHeapStart = m_nPerBatchSrvRingHeapEnd = 0;
    perBatchSrvHeapDesc.NumDescriptors = m_nPerBatchSrvRingHeapSize;
    perBatchSrvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    perBatchSrvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(
            hr = m_pDev->CreateDescriptorHeap(
                &perBatchSrvHeapDesc, IID_PPV_ARGS(&m_pPerBatchSrvRingHeap)))) {
        return hr;
    }
    m_pPerBatchSrvRingHeap->SetName(L"PerBatch SRV Heap");

    return hr;
}

HRESULT D3d12GraphicsManager::CreateRenderTarget() {
    HRESULT hr = S_OK;

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle =
        m_pRtvHeap->GetCPUDescriptorHandleForHeapStart();

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

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        // Create a RTV
        if (FAILED(hr = m_pSwapChain->GetBuffer(
                       i, IID_PPV_ARGS(&m_pRenderTargets[2 * i])))) {
            return hr;
        }
        m_pDev->CreateRenderTargetView(m_pRenderTargets[2 * i], nullptr,
                                       rtvHandle);
        m_pRenderTargets[2 * i]->SetName(L"Render Target");
        rtvHandle.ptr += m_nRtvDescriptorSize;

        // Create intermediate MSAA RT
        if (FAILED(hr = m_pDev->CreateCommittedResource(
                       &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
                       D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                       &optimizedClearValue,
                       IID_PPV_ARGS(&m_pRenderTargets[2 * i + 1])))) {
            return hr;
        }

        m_pRenderTargets[2 * i + 1]->SetName(L"MSAA Render Target");

        m_pDev->CreateRenderTargetView(m_pRenderTargets[2 * i + 1],
                                       &renderTargetDesc, rtvHandle);
        rtvHandle.ptr += m_nRtvDescriptorSize;
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDepthStencil() {
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

    if (FAILED(hr = m_pDev->CreateCommittedResource(
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthOptimizedClearValue,
                   IID_PPV_ARGS(&m_pDepthStencilBuffer)))) {
        return hr;
    }

    m_pDepthStencilBuffer->SetName(L"DepthStencilBuffer0");

    m_pDev->CreateDepthStencilView(
        m_pDepthStencilBuffer, &depthStencilDesc,
        m_pDsvHeap->GetCPUDescriptorHandleForHeapStart());

    return hr;
}

size_t D3d12GraphicsManager::CreateVertexBuffer(
    const SceneObjectVertexArray& v_property_array) {
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

    D3D12_SUBRESOURCE_DATA vertexData = {};
    vertexData.pData = v_property_array.GetData();

    UpdateSubresources<1>(m_pCommandList[m_nFrameIndex], pVertexBuffer,
                          pVertexBufferUploadHeap, 0, 0, 1, &vertexData);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pVertexBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    // initialize the vertex buffer view
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    vertexBufferView.BufferLocation = pVertexBuffer->GetGPUVirtualAddress();
    vertexBufferView.StrideInBytes = (UINT)(v_property_array.GetDataSize() /
                                            v_property_array.GetVertexCount());
    vertexBufferView.SizeInBytes = (UINT)v_property_array.GetDataSize();
    auto offset = m_VertexBufferView.size();
    m_VertexBufferView.push_back(vertexBufferView);

    m_Buffers.push_back(pVertexBuffer);
    m_Buffers.push_back(pVertexBufferUploadHeap);

    return offset;
}

size_t D3d12GraphicsManager::CreateIndexBuffer(
    const SceneObjectIndexArray& index_array) {
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

    D3D12_SUBRESOURCE_DATA indexData = {};
    indexData.pData = index_array.GetData();

    UpdateSubresources<1>(m_pCommandList[m_nFrameIndex], pIndexBuffer,
                          pIndexBufferUploadHeap, 0, 0, 1, &indexData);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pIndexBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
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

int32_t D3d12GraphicsManager::CreateTextureBuffer(SceneObjectTexture& texture) {
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
                   &prop, D3D12_HEAP_FLAG_NONE, &textureDesc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&pTextureBuffer)))) {
        return -1;
    }

    const UINT subresourceCount =
        textureDesc.DepthOrArraySize * textureDesc.MipLevels;
    const UINT64 uploadBufferSize =
        GetRequiredIntermediateSize(pTextureBuffer, 0, subresourceCount);

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
                   &prop, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&pTextureUploadHeap)))) {
        return -1;
    }

    // Copy data to the intermediate upload heap and then schedule a copy
    // from the upload heap to the Texture2D.
    D3D12_SUBRESOURCE_DATA textureData = {};
    textureData.pData = pImage->data;
    textureData.RowPitch = pImage->pitch;
    textureData.SlicePitch = static_cast<uint64_t>(pImage->pitch) *
                             static_cast<uint64_t>(pImage->Height);

    UpdateSubresources(m_pCommandList[m_nFrameIndex], pTextureBuffer,
                       pTextureUploadHeap, 0, 0, subresourceCount,
                       &textureData);
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pTextureBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    m_Buffers.push_back(pTextureUploadHeap);
    auto texture_id = m_Textures.size();
    m_Textures.push_back(pTextureBuffer);

    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle =
        m_pSrvHeap->GetCPUDescriptorHandleForHeapStart();
    srvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize * texture_id;

    m_pDev->CreateShaderResourceView(m_Textures[texture_id], NULL,
                                     srvCpuHandle);

    return static_cast<int32_t>(texture_id);
}

uint32_t D3d12GraphicsManager::CreateSamplerBuffer() {
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

    D3D12_RANGE readRange = {0, 0};

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
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
                       IID_PPV_ARGS(&m_pLightDataUploadBuffer[i])))) {
            return hr;
        }

        hr = m_pLightDataUploadBuffer[i]->Map(
            0, &readRange, reinterpret_cast<void**>(&m_pLightDataBegin[i]));
        m_pLightDataUploadBuffer[i]->SetName(L"Light Data Buffer");

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation =
            m_pLightDataUploadBuffer[i]->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes = kSizeLightInfo;

        D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
        cbvHandle.ptr = m_pCbvHeap->GetCPUDescriptorHandleForHeapStart().ptr +
                        static_cast<int64_t>(i) * m_nCbvSrvUavDescriptorSize;
        m_pDev->CreateConstantBufferView(&cbvDesc, cbvHandle);
    }

    return hr;
}

HRESULT D3d12GraphicsManager::CreateGraphicsResources() {
    HRESULT hr;

#if defined(D3D12_DEBUG_LAYER)
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

#if defined(D3D12_DEBUG_LAYER)
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
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    if (FAILED(hr = m_pDev->CreateCommandQueue(
                   &queueDesc, IID_PPV_ARGS(&m_pCommandQueue)))) {
        SafeRelease(&pFactory);
        return hr;
    }

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
                   m_pCommandQueue,  // Swap chain needs the queue so that it
                                     // can force a flush on it
                   hWnd, &scd, NULL, NULL, &pSwapChain))) {
        SafeRelease(&pFactory);
        return hr;
    }

    SafeRelease(&pFactory);

    m_pSwapChain = reinterpret_cast<IDXGISwapChain3*>(pSwapChain);

    m_nFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        if (FAILED(hr = m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                            IID_PPV_ARGS(&m_pFence[i])))) {
            return hr;
        }

        m_nFenceValue[i] = 1;
    }

    m_hFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (m_hFenceEvent == NULL) {
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

    cout << "Creating Root Signatures ...";
    if (FAILED(hr = CreateRootSignature())) {
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

HRESULT D3d12GraphicsManager::CreateRootSignature() {
    HRESULT hr = S_OK;

    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

    // This is the highest version the sample supports. If CheckFeatureSupport
    // succeeds, the HighestVersion returned will not be greater than this.
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

    if (FAILED(m_pDev->CheckFeatureSupport(
            D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))) {
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    // root signature for base pass
    {
        D3D12_DESCRIPTOR_RANGE1 ranges[] = {
            {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0},
            {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, UINT_MAX, 0, 0,
             D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
             D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
            {D3D12_DESCRIPTOR_RANGE_TYPE_CBV, UINT_MAX, 12, 0,
             D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
             D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND}};

        D3D12_ROOT_PARAMETER1 rootParameters[] = {
            {D3D12_ROOT_PARAMETER_TYPE_CBV,
             {},
             D3D12_SHADER_VISIBILITY_ALL},  // Per Frame Constant
            {D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
             {},
             D3D12_SHADER_VISIBILITY_VERTEX},  // Per Batch Constant
            {D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
             {1, &ranges[0]},
             D3D12_SHADER_VISIBILITY_PIXEL},  // Samplers
            {D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
             {1, &ranges[1]},
             D3D12_SHADER_VISIBILITY_PIXEL},  // SRVs
            {D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
             {1, &ranges[2]},
             D3D12_SHADER_VISIBILITY_ALL}  // CBVs
        };
        rootParameters[0].Descriptor.ShaderRegister = 10;
        rootParameters[0].Descriptor.RegisterSpace = 0;
        rootParameters[0].Descriptor.Flags =
            D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
        rootParameters[1].Constants.ShaderRegister = 11;
        rootParameters[1].Constants.RegisterSpace = 0;
        rootParameters[1].Constants.Num32BitValues = 16;

        // Allow input layout and deny uneccessary access to certain pipeline
        // stages.
        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        D3D12_ROOT_SIGNATURE_DESC1 rootSignatureDesc = {
            _countof(rootParameters), rootParameters, 0, nullptr,
            rootSignatureFlags};

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedRootSignatureDesc = {
            D3D_ROOT_SIGNATURE_VERSION_1_1,
        };

        versionedRootSignatureDesc.Desc_1_1 = rootSignatureDesc;

        ID3DBlob* signature = nullptr;
        ID3DBlob* error = nullptr;
        if (SUCCEEDED(hr = D3D12SerializeVersionedRootSignature(
                          &versionedRootSignatureDesc, &signature, &error))) {
            hr = m_pDev->CreateRootSignature(0, signature->GetBufferPointer(),
                                             signature->GetBufferSize(),
                                             IID_PPV_ARGS(&m_pRootSignature));
        }

        m_pRootSignature->SetName(L"RootSignature0");

        SafeRelease(&signature);
        SafeRelease(&error);
    }

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
    ID3D12PipelineState* pPipelineState;

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

    // create the input layout object
    D3D12_INPUT_ELEMENT_DESC ied_full[] = {
        {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 1, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 3, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

    D3D12_INPUT_ELEMENT_DESC ied_simple[] = {
        {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 2, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

    D3D12_INPUT_ELEMENT_DESC ied_cube[] = {
        {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, ::DXGI_FORMAT_R32G32_FLOAT, 3, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

    D3D12_INPUT_ELEMENT_DESC ied_pos_only[] = {
        {"POSITION", 0, ::DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

    // create rasterizer descriptor
    D3D12_RASTERIZER_DESC rsd = {D3D12_FILL_MODE_SOLID,
                                 D3D12_CULL_MODE_BACK,
                                 TRUE,
                                 D3D12_DEFAULT_DEPTH_BIAS,
                                 D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
                                 TRUE,
                                 FALSE,
                                 FALSE,
                                 0,
                                 D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF};

    const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlend = {
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

    D3D12_BLEND_DESC bld = {FALSE,
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

    const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp = {
        D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
        D3D12_COMPARISON_FUNC_ALWAYS};

    D3D12_DEPTH_STENCIL_DESC dsd = {TRUE,
                                    D3D12_DEPTH_WRITE_MASK_ALL,
                                    D3D12_COMPARISON_FUNC_LESS,
                                    FALSE,
                                    D3D12_DEFAULT_STENCIL_READ_MASK,
                                    D3D12_DEFAULT_STENCIL_WRITE_MASK,
                                    defaultStencilOp,
                                    defaultStencilOp};

    // describe and create the graphics pipeline state object (PSO)
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psod = {};
    psod.pRootSignature = m_pRootSignature;
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
    psod.NumRenderTargets = 1;
    psod.RTVFormats[0] = ::DXGI_FORMAT_R8G8B8A8_UNORM;
    psod.DSVFormat = ::DXGI_FORMAT_D32_FLOAT;
    psod.SampleDesc.Count = 4;  // 4X MSAA
    psod.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;

    if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(
                   &psod, IID_PPV_ARGS(&pPipelineState)))) {
        return false;
    }

    pPipelineState->SetName(s2ws(pipelineState.pipelineStateName).c_str());

    pipelineState.psoIndex = static_cast<int32_t>(m_pPipelineStates.size());
    m_pPipelineStates.push_back(pPipelineState);

    return hr;
}

HRESULT D3d12GraphicsManager::CreateCommandList() {
    HRESULT hr = S_OK;

    for (uint32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        if (FAILED(hr = m_pDev->CreateCommandAllocator(
                       D3D12_COMMAND_LIST_TYPE_DIRECT,
                       IID_PPV_ARGS(&m_pCommandAllocator[i])))) {
            return hr;
        }
        m_pCommandAllocator[i]->SetName(
            (wstring(L"Command Allocator") + to_wstring(i)).c_str());

        hr = m_pDev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       m_pCommandAllocator[i], NULL,
                                       IID_PPV_ARGS(&m_pCommandList[i]));

        if (SUCCEEDED(hr)) {
            m_pCommandList[i]->SetName(
                (wstring(L"Command List") + to_wstring(i)).c_str());
        }
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
            if (material) {
                if (auto& texture = material->GetBaseColor().ValueMap) {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.diffuseMap = texture_id;
                }

                if (auto& texture = material->GetNormal().ValueMap) {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.normalMap = texture_id;
                }

                if (auto& texture = material->GetMetallic().ValueMap) {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.metallicMap = texture_id;
                }

                if (auto& texture = material->GetRoughness().ValueMap) {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.roughnessMap = texture_id;
                }

                if (auto& texture = material->GetAO().ValueMap) {
                    int32_t texture_id;
                    const Image& image = *texture->GetTextureImage();
                    texture_id = CreateTextureBuffer(*texture);

                    dbc->material.aoMap = texture_id;
                }
            }

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
        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = pImage->data;
        textureData.RowPitch = pImage->pitch;
        textureData.SlicePitch = static_cast<uint64_t>(pImage->pitch) *
                                 static_cast<uint64_t>(pImage->Height);

        UpdateSubresources(m_pCommandList[m_nFrameIndex], pTextureBuffer,
                           pTextureUploadHeap, 0, i, 1, &textureData);
    }

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pTextureBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    m_Buffers.push_back(pTextureUploadHeap);

    for (int32_t i = 0; i < GfxConfiguration::kMaxInFlightFrameCount; i++) {
        m_Frames[i].skybox = m_Textures.size();
    }

    m_Textures.push_back(pTextureBuffer);
}

void D3d12GraphicsManager::initializeTerrain(const Scene& scene) {}

void D3d12GraphicsManager::EndScene() {
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

void D3d12GraphicsManager::BeginFrame(const Frame& frame) {
    GraphicsManager::BeginFrame(frame);

    // Indicate that the back buffer will be used as a resolve source.
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex + 1];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    // Set necessary state.
    m_pCommandList[m_nFrameIndex]->SetGraphicsRootSignature(m_pRootSignature);

    ID3D12DescriptorHeap* ppHeaps[] = {m_pPerBatchSrvRingHeap, m_pSamplerHeap};
    m_pCommandList[m_nFrameIndex]->SetDescriptorHeaps(_countof(ppHeaps),
                                                      ppHeaps);

    m_pCommandList[m_nFrameIndex]->RSSetViewports(1, &m_ViewPort);
    m_pCommandList[m_nFrameIndex]->RSSetScissorRects(1, &m_ScissorRect);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    // rtvHandle.ptr = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart().ptr +
    // m_nFrameIndex * m_nRtvDescriptorSize; bind the MSAA buffer
    rtvHandle.ptr = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart().ptr +
                    (2L * m_nFrameIndex + 1L) * m_nRtvDescriptorSize;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    dsvHandle = m_pDsvHeap->GetCPUDescriptorHandleForHeapStart();
    m_pCommandList[m_nFrameIndex]->OMSetRenderTargets(1, &rtvHandle, FALSE,
                                                      &dsvHandle);

    // clear the back buffer to a deep blue
    const FLOAT clearColor[] = {0.2f, 0.3f, 0.4f, 1.0f};
    m_pCommandList[m_nFrameIndex]->ClearRenderTargetView(rtvHandle, clearColor,
                                                         0, nullptr);
    m_pCommandList[m_nFrameIndex]->ClearDepthStencilView(
        m_pDsvHeap->GetCPUDescriptorHandleForHeapStart(),
        D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    SetPerFrameConstants(frame.frameContext);
    SetLightInfo(frame.lightInfo);

    // Per Frame CBV
    m_pCommandList[m_nFrameIndex]->SetGraphicsRootConstantBufferView(
        0,
        m_pPerFrameConstantUploadBuffer[m_nFrameIndex]->GetGPUVirtualAddress());
    D3D12_GPU_DESCRIPTOR_HANDLE cbvHandler;
    cbvHandler.ptr = m_pCbvHeap->GetGPUDescriptorHandleForHeapStart().ptr +
                     m_nFrameIndex * m_nCbvSrvUavDescriptorSize;
    m_pCommandList[m_nFrameIndex]->SetGraphicsRootDescriptorTable(4,
                                                                  cbvHandler);
}

void D3d12GraphicsManager::EndFrame(const Frame&) {
    HRESULT hr;

    if (SUCCEEDED(hr = m_pCommandList[m_nFrameIndex]->Close())) {
        ID3D12CommandList* ppCommandLists[] = {m_pCommandList[m_nFrameIndex]};
        m_pCommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                             ppCommandLists);

        WaitForPreviousFrame();
    }

    ResetCommandList();
}

void D3d12GraphicsManager::Draw() {
    GraphicsManager::Draw();

    MsaaResolve();
}

void D3d12GraphicsManager::DrawBatch(const Frame& frame) {
    for (const auto& pDbc : frame.batchContexts) {
        m_pCommandList[m_nFrameIndex]->SetGraphicsRoot32BitConstants(
            1, 16, &pDbc->modelMatrix, 0);

        const D3dDrawBatchContext& dbc =
            dynamic_cast<const D3dDrawBatchContext&>(*pDbc);

        // select which vertex buffer(s) to use
        for (uint32_t j = 0; j < dbc.property_count; j++) {
            m_pCommandList[m_nFrameIndex]->IASetVertexBuffers(
                j, 1, &m_VertexBufferView[dbc.property_offset + j]);
        }

        // set primitive topology
        m_pCommandList[m_nFrameIndex]->IASetPrimitiveTopology(
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        // select which index buffer to use
        m_pCommandList[m_nFrameIndex]->IASetIndexBuffer(
            &m_IndexBufferView[dbc.index_offset]);

        // CBV SRV
        D3D12_CPU_DESCRIPTOR_HANDLE cbvCpuHandle =
            m_pCbvHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle =
            m_pSrvHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE dstSrvCpuHandle =
            m_pPerBatchSrvRingHeap->GetCPUDescriptorHandleForHeapStart();
        if (m_nPerBatchSrvRingHeapSize - m_nPerBatchSrvRingHeapEnd <
            GfxConfiguration::kMaxTexturePerMaterialCount + 1) {
            m_nPerBatchSrvRingHeapEnd = 0;
        }
        m_nPerBatchSrvRingHeapStart = m_nPerBatchSrvRingHeapEnd;
        dstSrvCpuHandle.ptr +=
            m_nPerBatchSrvRingHeapStart * m_nCbvSrvUavDescriptorSize;

        m_pDev->CopyDescriptorsSimple(1, dstSrvCpuHandle, cbvCpuHandle,
                                      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        dstSrvCpuHandle.ptr += m_nCbvSrvUavDescriptorSize;

        if (dbc.material.diffuseMap >= 0) {
            srvCpuHandle.ptr +=
                dbc.material.diffuseMap * m_nCbvSrvUavDescriptorSize;
            m_pDev->CopyDescriptorsSimple(
                6, dstSrvCpuHandle, srvCpuHandle,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        // set skybox
        if (m_Frames[m_nFrameIndex].skybox >= 0) {
            D3D12_CPU_DESCRIPTOR_HANDLE srcSrvCpuHandle =
                m_pSrvHeap->GetCPUDescriptorHandleForHeapStart();
            srcSrvCpuHandle.ptr +=
                m_Frames[m_nFrameIndex].skybox * m_nCbvSrvUavDescriptorSize;
            dstSrvCpuHandle.ptr += 10 * m_nCbvSrvUavDescriptorSize;

            m_pDev->CopyDescriptorsSimple(
                1, dstSrvCpuHandle, srcSrvCpuHandle,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        m_nPerBatchSrvRingHeapEnd +=
            GfxConfiguration::kMaxTexturePerMaterialCount + 1;

        // Sampler
        m_pCommandList[m_nFrameIndex]->SetGraphicsRootDescriptorTable(
            2, m_pSamplerHeap->GetGPUDescriptorHandleForHeapStart());

        // CBV
        D3D12_GPU_DESCRIPTOR_HANDLE cbvGpuHandle =
            m_pPerBatchSrvRingHeap->GetGPUDescriptorHandleForHeapStart();
        cbvGpuHandle.ptr +=
            m_nPerBatchSrvRingHeapStart * m_nCbvSrvUavDescriptorSize;
        m_pCommandList[m_nFrameIndex]->SetGraphicsRootDescriptorTable(
            4, cbvGpuHandle);

        // SRV
        D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle;
        srvGpuHandle.ptr = cbvGpuHandle.ptr + m_nCbvSrvUavDescriptorSize;
        m_pCommandList[m_nFrameIndex]->SetGraphicsRootDescriptorTable(
            3, srvGpuHandle);

        // draw the vertex buffer to the back buffer
        m_pCommandList[m_nFrameIndex]->DrawIndexedInstanced(dbc.index_count, 1,
                                                            0, 0, 0);
    }
}

void D3d12GraphicsManager::SetPipelineState(
    const std::shared_ptr<PipelineState>& pipelineState, const Frame& frame) {
    if (pipelineState) {
        std::shared_ptr<D3d12PipelineState> state =
            dynamic_pointer_cast<D3d12PipelineState>(pipelineState);
        if (state->psoIndex == -1) {
            CreatePSO(*state);
        }

        m_pCommandList[m_nFrameIndex]->SetPipelineState(
            m_pPipelineStates[state->psoIndex]);
    }
}

void D3d12GraphicsManager::SetPerFrameConstants(
    const DrawFrameContext& context) {
    memcpy(m_pPerFrameCbvDataBegin[m_nFrameIndex],
           &static_cast<const PerFrameConstants&>(context),
           sizeof(PerFrameConstants));
}

void D3d12GraphicsManager::SetLightInfo(const LightInfo& lightInfo) {
    memcpy(m_pLightDataBegin[m_nFrameIndex], &lightInfo, sizeof(LightInfo));
}

HRESULT D3d12GraphicsManager::ResetCommandList() {
    HRESULT hr;

    // command list allocators can only be reset when the associated
    // command lists have finished execution on the GPU; apps should use
    // fences to determine GPU execution progress.
    if (SUCCEEDED(hr = m_pCommandAllocator[m_nFrameIndex]->Reset())) {
        // however, when ExecuteCommandList() is called on a particular command
        // list, that command list can then be reset at any time and must be
        // before re-recording.
        hr = m_pCommandList[m_nFrameIndex]->Reset(
            m_pCommandAllocator[m_nFrameIndex], NULL);
    }

    return hr;
}

HRESULT D3d12GraphicsManager::MsaaResolve() {
    D3D12_RESOURCE_BARRIER barrier = {};

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex + 1];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    m_pCommandList[m_nFrameIndex]->ResolveSubresource(
        m_pRenderTargets[2 * m_nFrameIndex], 0,
        m_pRenderTargets[2 * m_nFrameIndex + 1], 0,
        ::DXGI_FORMAT_R8G8B8A8_UNORM);

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = m_pRenderTargets[2 * m_nFrameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pCommandList[m_nFrameIndex]->ResourceBarrier(1, &barrier);

    return S_OK;
}

void D3d12GraphicsManager::Present() {
    HRESULT hr;

    // swap the back buffer and the front buffer
    hr = m_pSwapChain->Present(1, 0);

    WaitForPreviousFrame();

    (void)hr;
}

void D3d12GraphicsManager::DrawSkyBox() {
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

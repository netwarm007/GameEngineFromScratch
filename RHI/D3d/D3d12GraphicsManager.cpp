#include <iostream>
#include <objbase.h>
#include "D3d12GraphicsManager.hpp"
#include "WindowsApplication.hpp"
#include "SceneManager.hpp"
#include "AssetLoader.hpp"
#include "IPhysicsManager.hpp"
#include "D3dShaderManager.hpp"

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

	//------------------------------------------------------------------------------------------------
	// Row-by-row memcpy
	inline void MemcpySubresource(
		_In_ const D3D12_MEMCPY_DEST* pDest,
		_In_ const D3D12_SUBRESOURCE_DATA* pSrc,
		SIZE_T RowSizeInBytes,
		UINT NumRows,
		UINT NumSlices)
	{
		for (UINT z = 0; z < NumSlices; ++z)
		{
			BYTE* pDestSlice = reinterpret_cast<BYTE*>(pDest->pData) + pDest->SlicePitch * z;
			const BYTE* pSrcSlice = reinterpret_cast<const BYTE*>(pSrc->pData) + pSrc->SlicePitch * z;
			for (UINT y = 0; y < NumRows; ++y)
			{
				memcpy(pDestSlice + pDest->RowPitch * y,
					pSrcSlice + pSrc->RowPitch * y,
					RowSizeInBytes);
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
		_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) UINT NumSubresources)
	{
		D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
		UINT64 RequiredSize = 0;

		ID3D12Device* pDevice;
		pDestinationResource->GetDevice(__uuidof(*pDevice), reinterpret_cast<void**>(&pDevice));
		pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, 0, nullptr, nullptr, nullptr, &RequiredSize);
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
		_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) UINT NumSubresources,
		UINT64 RequiredSize,
		_In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
		_In_reads_(NumSubresources) const UINT* pNumRows,
		_In_reads_(NumSubresources) const UINT64* pRowSizesInBytes,
		_In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData)
	{
		// Minor validation
		D3D12_RESOURCE_DESC IntermediateDesc = pIntermediate->GetDesc();
		D3D12_RESOURCE_DESC DestinationDesc = pDestinationResource->GetDesc();
		if (IntermediateDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
			IntermediateDesc.Width < RequiredSize + pLayouts[0].Offset ||
			RequiredSize >(SIZE_T) - 1 ||
			(DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
			(FirstSubresource != 0 || NumSubresources != 1)))
		{
			return 0;
		}

		BYTE* pData;
		HRESULT hr = pIntermediate->Map(0, NULL, reinterpret_cast<void**>(&pData));
		if (FAILED(hr))
		{
			return 0;
		}

		for (UINT i = 0; i < NumSubresources; ++i)
		{
			if (pRowSizesInBytes[i] >(SIZE_T)-1) return 0;
			D3D12_MEMCPY_DEST DestData = { pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch, pLayouts[i].Footprint.RowPitch * pNumRows[i] };
			MemcpySubresource(&DestData, &pSrcData[i], (SIZE_T)pRowSizesInBytes[i], pNumRows[i], pLayouts[i].Footprint.Depth);
		}
		pIntermediate->Unmap(0, NULL);

		if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
		{
			pCmdList->CopyBufferRegion(
				pDestinationResource, 0, pIntermediate, pLayouts[0].Offset, pLayouts[0].Footprint.Width);
		}
		else
		{
			for (UINT i = 0; i < NumSubresources; ++i)
			{
				D3D12_TEXTURE_COPY_LOCATION Dst = { pDestinationResource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, i + FirstSubresource };
				D3D12_TEXTURE_COPY_LOCATION Src = { pIntermediate, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, pLayouts[i] };
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
		_In_ ID3D12Resource* pIntermediate,
		UINT64 IntermediateOffset,
		_In_range_(0, D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
		_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) UINT NumSubresources,
		_In_reads_(NumSubresources) D3D12_SUBRESOURCE_DATA* pSrcData)
	{
		UINT64 RequiredSize = 0;
		UINT64 MemToAlloc = static_cast<UINT64>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64)) * NumSubresources;
		if (MemToAlloc > SIZE_MAX)
		{
			return 0;
		}
		void* pMem = HeapAlloc(GetProcessHeap(), 0, static_cast<SIZE_T>(MemToAlloc));
		if (pMem == NULL)
		{
			return 0;
		}
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts = reinterpret_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(pMem);
		UINT64* pRowSizesInBytes = reinterpret_cast<UINT64*>(pLayouts + NumSubresources);
		UINT* pNumRows = reinterpret_cast<UINT*>(pRowSizesInBytes + NumSubresources);

		D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
		ID3D12Device* pDevice;
		pDestinationResource->GetDevice(__uuidof(*pDevice), reinterpret_cast<void**>(&pDevice));
		pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, pLayouts, pNumRows, pRowSizesInBytes, &RequiredSize);
		SafeRelease(&pDevice);

		UINT64 Result = UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, pLayouts, pNumRows, pRowSizesInBytes, pSrcData);
		HeapFree(GetProcessHeap(), 0, pMem);
		return Result;
	}

	//------------------------------------------------------------------------------------------------
	// Stack-allocating UpdateSubresources implementation
	template <UINT MaxSubresources>
	inline UINT64 UpdateSubresources(
		_In_ ID3D12GraphicsCommandList* pCmdList,
		_In_ ID3D12Resource* pDestinationResource,
		_In_ ID3D12Resource* pIntermediate,
		UINT64 IntermediateOffset,
		_In_range_(0, MaxSubresources) UINT FirstSubresource,
		_In_range_(1, MaxSubresources - FirstSubresource) UINT NumSubresources,
		_In_reads_(NumSubresources) D3D12_SUBRESOURCE_DATA* pSrcData)
	{
		UINT64 RequiredSize = 0;
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT Layouts[MaxSubresources];
		UINT NumRows[MaxSubresources];
		UINT64 RowSizesInBytes[MaxSubresources];

		D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
		ID3D12Device* pDevice;
		pDestinationResource->GetDevice(__uuidof(*pDevice), reinterpret_cast<void**>(&pDevice));
		pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, Layouts, NumRows, RowSizesInBytes, &RequiredSize);
		SafeRelease(&pDevice);

		return UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, Layouts, NumRows, RowSizesInBytes, pSrcData);
	}

	// ************************************************
	// Code copied from d3dx12.h finished here
	// ************************************************
}

HRESULT D3d12GraphicsManager::WaitForPreviousFrame() {
    // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
    // This is code implemented as such for simplicity. More advanced samples 
    // illustrate how to use fences for efficient resource usage.
    
    // Signal and increment the fence value.
    HRESULT hr;
    const uint64_t fence = m_nFenceValue;
    if(FAILED(hr = m_pCommandQueue->Signal(m_pFence, fence)))
    {
        return hr;
    }

    m_nFenceValue++;

    // Wait until the previous frame is finished.
    if (m_pFence->GetCompletedValue() < fence)
    {
        if(FAILED(hr = m_pFence->SetEventOnCompletion(fence, m_hFenceEvent)))
        {
            return hr;
        }
        WaitForSingleObject(m_hFenceEvent, INFINITE);
    }

    m_nFrameIndex = m_pSwapChain->GetCurrentBackBufferIndex();

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDescriptorHeaps() 
{
    HRESULT hr;

    // Describe and create a render target view (RTV) descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = m_kFrameCount + 1; // +1 for MSAA Resolver
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
        m_kFrameCount * (2 * m_kMaxObjectCount)                // 1 perFrame and 1 per DrawBatch
        + m_kMaxTextureCount;                                 // + m_kMaxTextureCount for the SRV(Texture).
    cbvSrvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvSrvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if(FAILED(hr = m_pDev->CreateDescriptorHeap(&cbvSrvUavHeapDesc, IID_PPV_ARGS(&m_pCbvHeap)))) {
        return hr;
    }

    m_nCbvSrvDescriptorSize = m_pDev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Describe and create a sampler descriptor heap.
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
    samplerHeapDesc.NumDescriptors = m_kMaxTextureCount; // this is the max D3d12 HW support currently
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

HRESULT D3d12GraphicsManager::CreateRenderTarget() 
{
    HRESULT hr = S_OK;

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart();

    // Create a RTV for each frame.
    for (uint32_t i = 0; i < m_kFrameCount; i++)
    {
        if (FAILED(hr = m_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&m_pRenderTargets[i])))) {
            return hr;
        }
        m_pDev->CreateRenderTargetView(m_pRenderTargets[i], nullptr, rtvHandle);
        rtvHandle.ptr += m_nRtvDescriptorSize;
    }

    // Create intermediate MSAA RT
    D3D12_RENDER_TARGET_VIEW_DESC renderTargetDesc;
    renderTargetDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    renderTargetDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;

    D3D12_CLEAR_VALUE optimizedClearValue = {};
    optimizedClearValue.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    optimizedClearValue.Color[0] = 0.0f;
    optimizedClearValue.Color[1] = 0.1f;
    optimizedClearValue.Color[2] = 0.2f;
    optimizedClearValue.Color[3] = 1.0f;

    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.Width = g_pApp->GetConfiguration().screenWidth;
    textureDesc.Height = g_pApp->GetConfiguration().screenHeight;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.SampleDesc.Count = 4;
    textureDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    if (FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        &optimizedClearValue,
        IID_PPV_ARGS(&m_pMsaaRenderTarget)
    )))
    {
        return hr;
    }

    m_pDev->CreateRenderTargetView(m_pMsaaRenderTarget, &renderTargetDesc, rtvHandle);

    return hr;
}

HRESULT D3d12GraphicsManager::CreateDepthStencil()
{
    HRESULT hr;

    // Create the depth stencil view.
    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
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
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_D32_FLOAT;
    resourceDesc.SampleDesc.Count = 4;
    resourceDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
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

HRESULT D3d12GraphicsManager::CreateInternalVertexBuffer()
{
    HRESULT hr;

	ID3D12Resource* pVertexBufferUploadHeap;

    float fullScreenQuad[] = {
        -1.0f, 1.0f, 0.0f,       0.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,       0.0f, 1.0f,
        1.0f, 1.0f, 0.0f,       1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,       1.0f, 1.0f
    };

    // create vertex GPU heap 
    D3D12_HEAP_PROPERTIES prop = {};
    prop.Type = D3D12_HEAP_TYPE_DEFAULT;
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    prop.CreationNodeMask = 1;
    prop.VisibleNodeMask = 1;

    auto size = sizeof(float) * (3 + 2) * 4;
    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = size;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
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
	vertexData.pData = fullScreenQuad;

	UpdateSubresources<1>(m_pCommandList, pVertexBuffer, pVertexBufferUploadHeap, 0, 0, 1, &vertexData);
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource = pVertexBuffer;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	m_pCommandList->ResourceBarrier(1, &barrier);

	// initialize the vertex buffer view
	m_VertexBufferViewResolve.BufferLocation = pVertexBuffer->GetGPUVirtualAddress();
	m_VertexBufferViewResolve.StrideInBytes = 20;
	m_VertexBufferViewResolve.SizeInBytes = static_cast<UINT>(size);

    m_Buffers.push_back(pVertexBuffer);
    m_Buffers.push_back(pVertexBufferUploadHeap);

    return hr;
}

HRESULT D3d12GraphicsManager::CreateVertexBuffer(const SceneObjectVertexArray& v_property_array)
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
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
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

	UpdateSubresources<1>(m_pCommandList, pVertexBuffer, pVertexBufferUploadHeap, 0, 0, 1, &vertexData);
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource = pVertexBuffer;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	m_pCommandList->ResourceBarrier(1, &barrier);

	// initialize the vertex buffer view
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
	vertexBufferView.BufferLocation = pVertexBuffer->GetGPUVirtualAddress();
	vertexBufferView.StrideInBytes = (UINT)(v_property_array.GetDataSize() / v_property_array.GetVertexCount());
	vertexBufferView.SizeInBytes = (UINT)v_property_array.GetDataSize();
	m_VertexBufferView.push_back(vertexBufferView);

    m_Buffers.push_back(pVertexBuffer);
    m_Buffers.push_back(pVertexBufferUploadHeap);

    return hr;
}


HRESULT D3d12GraphicsManager::CreateIndexBuffer(const SceneObjectIndexArray& index_array)
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
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
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
	
	UpdateSubresources<1>(m_pCommandList, pIndexBuffer, pIndexBufferUploadHeap, 0, 0, 1, &indexData);
	D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pIndexBuffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	m_pCommandList->ResourceBarrier(1, &barrier);

	// initialize the index buffer view
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	indexBufferView.BufferLocation = pIndexBuffer->GetGPUVirtualAddress();
	indexBufferView.Format = DXGI_FORMAT_R32_UINT;
	indexBufferView.SizeInBytes = (UINT)index_array.GetDataSize();
	m_IndexBufferView.push_back(indexBufferView);

    m_Buffers.push_back(pIndexBuffer);
    m_Buffers.push_back(pIndexBufferUploadHeap);

    return hr;
}

HRESULT D3d12GraphicsManager::CreateTextureBuffer()
{
    // Describe and create a SRV for the texture.
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
    size_t texture_id = static_cast<uint32_t>(m_TextureIndex.size());
    srvHandle.ptr = m_pCbvHeap->GetCPUDescriptorHandleForHeapStart().ptr + (m_kTextureDescStartIndex + texture_id) * m_nCbvSrvDescriptorSize;
    m_pDev->CreateShaderResourceView(m_pMsaaRenderTarget, &srvDesc, srvHandle);
    m_TextureIndex["MSAA"] = texture_id;

    return S_OK;
}

HRESULT D3d12GraphicsManager::CreateTextureBuffer(SceneObjectTexture& texture)
{
    HRESULT hr = S_OK;

    auto it = m_TextureIndex.find(texture.GetName());
	if (it == m_TextureIndex.end()) {
		auto image = texture.GetTextureImage();

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
			return hr;
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
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
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
			return hr;
		}

		// Copy data to the intermediate upload heap and then schedule a copy 
		// from the upload heap to the Texture2D.
		D3D12_SUBRESOURCE_DATA textureData = {};
		if (image.bitcount == 24)
		{
            // DXGI does not have 24bit formats so we have to extend it to 32bit
            uint32_t new_pitch = image.pitch / 3 * 4;
            size_t data_size = new_pitch * image.Height;
            void* data = new uint8_t[data_size];
            uint8_t* buf = reinterpret_cast<uint8_t*>(data);
            uint8_t* src = reinterpret_cast<uint8_t*>(image.data);
            for (uint32_t row = 0; row < image.Height; row++) {
                buf = reinterpret_cast<uint8_t*>(data) + row * new_pitch;
                src = reinterpret_cast<uint8_t*>(image.data) + row * image.pitch;
                for (uint32_t col = 0; col < image.Width; col++) {
                    *(uint32_t*)buf = *(uint32_t*)src;
                    buf[3] = 0;  // set alpha to 0
                    buf += 4;
                    src += 3;
                }
            }
            // we do not need to free the old data because the old data is still referenced by the
            // SceneObject
            // g_pMemoryManager->Free(image.data, image.data_size);
            image.data = data;
            image.data_size = data_size;
            image.pitch = new_pitch;
		}
    	textureData.pData = image.data;
		textureData.RowPitch = image.pitch;
		textureData.SlicePitch = image.pitch * image.Height;

		UpdateSubresources(m_pCommandList, pTextureBuffer, pTextureUploadHeap, 0, 0, subresourceCount, &textureData);
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier.Transition.pResource = pTextureBuffer;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_GENERIC_READ;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		m_pCommandList->ResourceBarrier(1, &barrier);

		// Describe and create a SRV for the texture.
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = -1;
        srvDesc.Texture2D.MostDetailedMip = 0;
		D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
		size_t texture_id = static_cast<uint32_t>(m_TextureIndex.size());
		srvHandle.ptr = m_pCbvHeap->GetCPUDescriptorHandleForHeapStart().ptr + (m_kTextureDescStartIndex + texture_id) * m_nCbvSrvDescriptorSize;
		m_pDev->CreateShaderResourceView(pTextureBuffer, &srvDesc, srvHandle);
		m_TextureIndex[texture.GetName()] = texture_id;

		m_Buffers.push_back(pTextureUploadHeap);
		m_Textures.push_back(pTextureBuffer);
	}

    return hr;
}

HRESULT D3d12GraphicsManager::CreateSamplerBuffer()
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

HRESULT D3d12GraphicsManager::CreateConstantBuffer()
{
    HRESULT hr;

    m_kSizePerFrameConstantBuffer = ALIGN(sizeof(DrawFrameContext) + sizeof(Light) * m_kMaxLightCount, 256); // CB size is required to be 256-byte aligned.
    m_kSizePerBatchConstantBuffer = ALIGN(sizeof(DrawBatchContext), 256); // CB size is required to be 256-byte aligned.
    m_kSizeConstantBufferPerFrame = m_kSizePerFrameConstantBuffer + m_kSizePerBatchConstantBuffer * m_kMaxObjectCount;

    D3D12_HEAP_PROPERTIES prop = { D3D12_HEAP_TYPE_UPLOAD, 
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN, 
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1 };

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = m_kSizeConstantBufferPerFrame * m_kFrameCount;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

	ID3D12Resource* pConstantUploadBuffer;
    if(FAILED(hr = m_pDev->CreateCommittedResource(
        &prop,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&pConstantUploadBuffer))))
    {
        return hr;
    }

    // populate descriptor table
    D3D12_CPU_DESCRIPTOR_HANDLE cbvHandle;
    cbvHandle.ptr = m_pCbvHeap->GetCPUDescriptorHandleForHeapStart().ptr;
    for (auto i = 0; i < m_kFrameCount; i++)
    {
        for (auto j = 0; j < m_kMaxObjectCount; j++)
        {
            // Describe and create constant buffer descriptors.
            // 1 per frame and 1 per batch descriptor per object
            D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};

            // Per frame constant buffer descriptor
            cbvDesc.BufferLocation = pConstantUploadBuffer->GetGPUVirtualAddress() 
                                        + i * m_kSizeConstantBufferPerFrame;
            cbvDesc.SizeInBytes = static_cast<UINT>(m_kSizePerFrameConstantBuffer);
            m_pDev->CreateConstantBufferView(&cbvDesc, cbvHandle);
            cbvHandle.ptr += m_nCbvSrvDescriptorSize;

            // Per batch constant buffer descriptor
            cbvDesc.BufferLocation += m_kSizePerFrameConstantBuffer + j * m_kSizePerBatchConstantBuffer;
            cbvDesc.SizeInBytes = static_cast<UINT>(m_kSizePerBatchConstantBuffer);
            m_pDev->CreateConstantBufferView(&cbvDesc, cbvHandle);
            cbvHandle.ptr += m_nCbvSrvDescriptorSize;
        }
    }

    D3D12_RANGE readRange = { 0, 0 };
    hr = pConstantUploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_pCbvDataBegin));

    m_Buffers.push_back(pConstantUploadBuffer);

    return hr;
}

HRESULT D3d12GraphicsManager::CreateGraphicsResources()
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
    scd.BufferCount = m_kFrameCount;                          // back buffer count
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

    cout << "Creating PSO ...";
    if (FAILED(hr = InitializePSO())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating Command List ...";
    if (FAILED(hr = CreateCommandList())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating PSO ...";
    if (FAILED(hr = InitializePSO())) {
        return hr;
    }
    cout << "Done!" << endl;

    cout << "Creating Command List ...";
    if (FAILED(hr = CreateCommandList())) {
        return hr;
    }
    cout << "Done!" << endl;

    return hr;
}

HRESULT D3d12GraphicsManager::CreateRootSignature()
{
    HRESULT hr = S_OK;

    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

    // This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

    if (FAILED(m_pDev->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
    {
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    // root signature for base pass
    {
        D3D12_DESCRIPTOR_RANGE1 ranges[3] = {
            { D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 2, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC },
            { D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 },
            { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0,D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC }
        };

        D3D12_ROOT_PARAMETER1 rootParameters[4] = {
            { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[0] }, D3D12_SHADER_VISIBILITY_ALL },
            { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[1] }, D3D12_SHADER_VISIBILITY_PIXEL },
            { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[2] }, D3D12_SHADER_VISIBILITY_PIXEL }
        };

        rootParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParameters[3].Constants = { 2, 0, 1 }; 
        rootParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

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

        SafeRelease(&signature);
        SafeRelease(&error);
    }

    // root signature for resolve pass
    {
        D3D12_DESCRIPTOR_RANGE1 ranges[1] = {
            { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0,D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC }
        };

        D3D12_ROOT_PARAMETER1 rootParameters[2];
        rootParameters[0] = 
            { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, { 1, &ranges[0] }, D3D12_SHADER_VISIBILITY_PIXEL };
        rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParameters[1].Constants = { 0, 0, 2 }; 
        rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

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
            hr = m_pDev->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_pRootSignatureResolve));
            if (FAILED(hr))
            {
                printf_s("failed to create root signature for reolve phase: %x", hr);
            }
        }

        SafeRelease(&signature);
        SafeRelease(&error);
    }

    return hr;
}


// this is the function that loads and prepares the pso 
HRESULT D3d12GraphicsManager::InitializePSO() {
    HRESULT hr = S_OK;

    // basic pass
    {
        const char* vsFilename = "Shaders/basic_vs.cso"; 
        const char* fsFilename = "Shaders/basic_ps.cso";

        // load the shaders
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
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 2, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

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

        // describe and create the graphics pipeline state object (PSO)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psod = {};
        psod.pRootSignature = m_pRootSignature;
        psod.VS             = vertexShaderByteCode;
        psod.PS             = pixelShaderByteCode;
        psod.BlendState     = bld;
        psod.SampleMask     = UINT_MAX;
        psod.RasterizerState= rsd;
        psod.DepthStencilState = dsd;
        psod.InputLayout    = { ied, _countof(ied) };
        psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psod.NumRenderTargets = 1;
        psod.RTVFormats[0]  = DXGI_FORMAT_R8G8B8A8_UNORM;
        psod.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psod.SampleDesc.Count = 4; // 4X MSAA
        psod.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;

        if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(&psod, IID_PPV_ARGS(&m_pPipelineState))))
        {
            return false;
        }

    }

    // resolve pass
    {
        const char* vsFilename = "Shaders/msaa_resolver_vs.cso"; 
        const char* fsFilename = "Shaders/msaa_resolver_ps.cso";

        // load the shaders
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
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

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

        // describe and create the graphics pipeline state object (PSO)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psod = {};
        psod.pRootSignature = m_pRootSignatureResolve;
        psod.VS             = vertexShaderByteCode;
        psod.PS             = pixelShaderByteCode;
        psod.BlendState     = bld;
        psod.SampleMask     = UINT_MAX;
        psod.RasterizerState= rsd;
        psod.DepthStencilState = dsd;
        psod.InputLayout    = { ied, _countof(ied) };
        psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psod.NumRenderTargets = 1;
        psod.RTVFormats[0]  = DXGI_FORMAT_R8G8B8A8_UNORM;
        psod.DSVFormat = DXGI_FORMAT_UNKNOWN;
        psod.SampleDesc.Count = 1;   // no MSAA
        psod.SampleDesc.Quality = 0; // no MSAA

        if (FAILED(hr = m_pDev->CreateGraphicsPipelineState(&psod, IID_PPV_ARGS(&m_pPipelineStateResolve))))
        {
            return false;
        }
    }

    return hr;
}

void D3d12GraphicsManager::InitializeBuffers(const Scene& scene)
{
    HRESULT hr;

	if (!m_pCommandList)
	{
		if (FAILED(hr = m_pDev->CreateCommandList(0, 
					D3D12_COMMAND_LIST_TYPE_DIRECT, 
					m_pCommandAllocator, 
					m_pPipelineState, 
					IID_PPV_ARGS(&m_pCommandList))))
		{
			return;
		}
	}

    cout << "Creating Vertex Buffer ...";
    for (auto _it : scene.GeometryNodes)
    {
	    auto pGeometryNode = _it.second.lock();

        if (pGeometryNode && pGeometryNode->Visible())
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
                const SceneObjectVertexArray& v_property_array = pMesh->GetVertexPropertyArray(i);

                CreateVertexBuffer(v_property_array);
            }

			// TODO: Implement LOD switching
            // auto indexGroupCount = pMesh->GetIndexGroupCount();

            const SceneObjectIndexArray& index_array      = pMesh->GetIndexArray(0);
            CreateIndexBuffer(index_array);

			auto material_index = index_array.GetMaterialIndex();
			auto material_key = pGeometryNode->GetMaterialRef(material_index);
			auto material = scene.GetMaterial(material_key);

			DrawBatchContext dbc;
			dbc.index_count = (UINT)index_array.GetIndexCount();
            dbc.property_count = vertexPropertiesCount;
			if (material) {
                dbc.material = material;
			}

            dbc.node = pGeometryNode;

            m_DrawBatchContext.push_back(dbc);
        }
    }
    cout << "Done!" << endl;

    cout << "Creating Internal Vertex Buffer ...";
	CreateInternalVertexBuffer();
    cout << "Done!" << endl;

    cout << "Creating Constant Buffer ...";
	if (FAILED(hr = CreateConstantBuffer())) {
		return;
	}

    cout << "Creating Sampler Buffer ...";
	if (FAILED(hr = CreateSamplerBuffer())) {
		return;
	}
    cout << "Done!" << endl;

    cout << "Creating Texture Buffer ...";
    if (FAILED(hr = CreateTextureBuffer())) {
        return;
    }

	for (auto _it : scene.Materials)
	{
		auto material = _it.second;
		if (material) {
			auto color = material->GetBaseColor();
			if (auto texture = color.ValueMap) {
				if (FAILED(hr = CreateTextureBuffer(*texture))) {
					return;
				}
			}
		}
	}

    cout << "Done!" << endl;

    if (SUCCEEDED(hr = m_pCommandList->Close()))
    {
        ID3D12CommandList* ppCommandLists[] = { m_pCommandList };
        m_pCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

        if (FAILED(hr = m_pDev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pFence))))
        {
            return;
        }

        m_nFenceValue = 1;

        m_hFenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        if (m_hFenceEvent == NULL)
        {
            hr = HRESULT_FROM_WIN32(GetLastError());
            if (FAILED(hr))
                return;
        }

        WaitForPreviousFrame();
    }

    hr = PopulateCommandList();

    return;
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

void D3d12GraphicsManager::ClearBuffers()
{
    SafeRelease(&m_pFence);
    for (auto p : m_Buffers) {
        SafeRelease(&p);
    }
    m_Buffers.clear();
    for (auto p : m_Textures) {
        SafeRelease(&p);
    }
    m_Textures.clear();
    m_TextureIndex.clear();
    m_VertexBufferView.clear();
    m_IndexBufferView.clear();
    m_DrawBatchContext.clear();
}

void D3d12GraphicsManager::Finalize()
{
    GraphicsManager::Finalize();

    SafeRelease(&m_pRtvHeap);
    SafeRelease(&m_pDsvHeap);
    SafeRelease(&m_pCbvHeap);
    SafeRelease(&m_pSamplerHeap);
    SafeRelease(&m_pRootSignature);
    SafeRelease(&m_pRootSignatureResolve);
    SafeRelease(&m_pCommandQueue);
    SafeRelease(&m_pCommandAllocator);
	SafeRelease(&m_pDepthStencilBuffer);
    SafeRelease(&m_pMsaaRenderTarget);
    for (uint32_t i = 0; i < m_kFrameCount; i++) {
        SafeRelease(&m_pRenderTargets[i]);
    }
    SafeRelease(&m_pSwapChain);
    SafeRelease(&m_pDev);
}

void D3d12GraphicsManager::Clear()
{
    GraphicsManager::Clear();
}

void D3d12GraphicsManager::Draw()
{
    PopulateCommandList();

    GraphicsManager::Draw();

    WaitForPreviousFrame();
}

HRESULT D3d12GraphicsManager::PopulateCommandList()
{
    HRESULT hr;

	// command list allocators can only be reset when the associated 
	// command lists have finished execution on the GPU; apps should use 
	// fences to determine GPU execution progress.
	if (FAILED(hr = m_pCommandAllocator->Reset()))
	{
		return hr;
	}

    // base pass + MSAA
    {
        // however, when ExecuteCommandList() is called on a particular command 
        // list, that command list can then be reset at any time and must be before 
        // re-recording.
        if (FAILED(hr = m_pCommandList->Reset(m_pCommandAllocator, m_pPipelineState)))
        {
            return hr;
        }

        // Indicate that the back buffer will be used as a resolve source.
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pMsaaRenderTarget;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
        // rtvHandle.ptr = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart().ptr + m_nFrameIndex * m_nRtvDescriptorSize;
        // bind the MSAA buffer
        rtvHandle.ptr = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart().ptr + m_kFrameCount * m_nRtvDescriptorSize;
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
        dsvHandle = m_pDsvHeap->GetCPUDescriptorHandleForHeapStart();
        m_pCommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

        // clear the back buffer to a deep blue
        const FLOAT clearColor[] = { 0.0f, 0.1f, 0.2f, 1.0f };
        m_pCommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        m_pCommandList->ClearDepthStencilView(m_pDsvHeap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

        // Set necessary state.
        m_pCommandList->SetGraphicsRootSignature(m_pRootSignature);

        ID3D12DescriptorHeap* ppHeaps[] = { m_pCbvHeap, m_pSamplerHeap };
        m_pCommandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

        // Sampler
        m_pCommandList->SetGraphicsRootDescriptorTable(1, m_pSamplerHeap->GetGPUDescriptorHandleForHeapStart());

        m_pCommandList->RSSetViewports(1, &m_ViewPort);
        m_pCommandList->RSSetScissorRects(1, &m_ScissorRect);
        m_pCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        // Set num of lights
        auto& DrawFrameContext = m_Frames[m_nFrameIndex].frameContext;
        m_pCommandList->SetGraphicsRoot32BitConstant(3, static_cast<UINT>(DrawFrameContext.m_lights.size()), 0);

        // do 3D rendering on the back buffer here
        int32_t i = 0;
        size_t vertex_buffer_view_offset = 0;
        for (auto dbc : m_DrawBatchContext)
        {
            // CBV Per Batch
            D3D12_GPU_DESCRIPTOR_HANDLE cbvSrvHandle;
            uint32_t nFrameResourceDescriptorOffset = m_nFrameIndex * (2 * m_kMaxObjectCount); // 2 descriptors for each draw call
            cbvSrvHandle.ptr = m_pCbvHeap->GetGPUDescriptorHandleForHeapStart().ptr 
                                    + (nFrameResourceDescriptorOffset + i * 2 /* 2 descriptors for each batch */) * m_nCbvSrvDescriptorSize;
            m_pCommandList->SetGraphicsRootDescriptorTable(0, cbvSrvHandle);

            // select which vertex buffer(s) to use
            for (uint32_t j = 0; j < dbc.property_count; j++)
            {
                m_pCommandList->IASetVertexBuffers(j, 1, &m_VertexBufferView[vertex_buffer_view_offset++]);
            }

			// set primitive topology
			m_pCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // select which index buffer to use
            m_pCommandList->IASetIndexBuffer(&m_IndexBufferView[i]);

            // Texture
            if(dbc.material)
            {
                if(auto texture = dbc.material->GetBaseColor().ValueMap)
                {
                    auto texture_index = m_TextureIndex[texture->GetName()];
                    D3D12_GPU_DESCRIPTOR_HANDLE srvHandle;
                    srvHandle.ptr = m_pCbvHeap->GetGPUDescriptorHandleForHeapStart().ptr + (m_kTextureDescStartIndex + texture_index) * m_nCbvSrvDescriptorSize;
                    m_pCommandList->SetGraphicsRootDescriptorTable(2, srvHandle);
                }
            }

            // draw the vertex buffer to the back buffer
            m_pCommandList->DrawIndexedInstanced(dbc.index_count, 1, 0, 0, 0);
            i++;
        }

#if 1
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pMsaaRenderTarget;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);

        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pRenderTargets[m_nFrameIndex];
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RESOLVE_DEST;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);

        m_pCommandList->ResolveSubresource(m_pRenderTargets[m_nFrameIndex], 0, m_pMsaaRenderTarget, 0, DXGI_FORMAT_R8G8B8A8_UNORM);

        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pMsaaRenderTarget;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);

        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pRenderTargets[m_nFrameIndex];
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);
#else
        // MSAA resolve pass
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pMsaaRenderTarget;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);

        m_pCommandList->SetPipelineState(m_pPipelineStateResolve);

        // Indicate that the back buffer will be used as a render target.
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pRenderTargets[m_nFrameIndex];
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);

        rtvHandle.ptr = m_pRtvHeap->GetCPUDescriptorHandleForHeapStart().ptr + m_nFrameIndex * m_nRtvDescriptorSize;
        m_pCommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

        // Set necessary state.
        m_pCommandList->SetGraphicsRootSignature(m_pRootSignatureResolve);

        // Set CBV
        m_pCommandList->SetGraphicsRoot32BitConstant(1, g_pApp->GetConfiguration().screenWidth, 0);
        m_pCommandList->SetGraphicsRoot32BitConstant(1, g_pApp->GetConfiguration().screenHeight, 1);

        // Set SRV
        auto texture_index = m_TextureIndex["MSAA"];
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle;
        srvHandle.ptr = m_pCbvHeap->GetGPUDescriptorHandleForHeapStart().ptr + (m_kTextureDescStartIndex + texture_index) * m_nCbvSrvDescriptorSize;
        m_pCommandList->SetGraphicsRootDescriptorTable(0, srvHandle);

        m_pCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        m_pCommandList->IASetVertexBuffers(0, 1, &m_VertexBufferViewResolve);
        m_pCommandList->DrawInstanced(4, 1, 0, 0);

        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_pRenderTargets[m_nFrameIndex];
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_pCommandList->ResourceBarrier(1, &barrier);
#endif

        hr = m_pCommandList->Close();
    }

    return hr;
}

void D3d12GraphicsManager::UpdateConstants()
{
    GraphicsManager::UpdateConstants();

	// CBV Per Frame
    SetPerFrameShaderParameters();
	int32_t i = 0;
    for (auto dbc : m_DrawBatchContext)
    {
	    SetPerBatchShaderParameters(i++);
    }
}

void D3d12GraphicsManager::RenderBuffers()
{
    HRESULT hr;

    // execute the command list
	ID3D12CommandList *ppCommandLists[] = { m_pCommandList };
    m_pCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // swap the back buffer and the front buffer
    hr = m_pSwapChain->Present(1, 0);

    (void)hr;
}

bool D3d12GraphicsManager::SetPerFrameShaderParameters()
{
    uint8_t* pHead = m_pCbvDataBegin + m_nFrameIndex * m_kSizeConstantBufferPerFrame;
    auto& DrawFrameContext = m_Frames[m_nFrameIndex].frameContext;
    size_t offset = (uint8_t *)&DrawFrameContext.m_lights - (uint8_t *)&DrawFrameContext;

    memcpy(pHead, 
        &DrawFrameContext, 
        offset); 

    pHead += ALIGN(offset, 16); // 16 bytes alignment

    for (auto light : DrawFrameContext.m_lights)
    {
        size_t size = ALIGN(sizeof(Light), 16); // 16 bytes alignment
        memcpy(pHead, &light, size);
        pHead += size;
    }

    return true;
}

bool D3d12GraphicsManager::SetPerBatchShaderParameters(int32_t index)
{
    PerBatchConstants pbc;
    memset(&pbc, 0x00, sizeof(pbc));

    Matrix4X4f trans;
    if (void* rigidBody = m_DrawBatchContext[index].node->RigidBody()) {
        // the geometry has rigid body bounded, we blend the simlation result here.
        Matrix4X4f simulated_result = g_pPhysicsManager->GetRigidBodyTransform(rigidBody);

        BuildIdentityMatrix(trans);

        // apply the rotation part of the simlation result
        memcpy(trans[0], simulated_result[0], sizeof(float) * 3);
        memcpy(trans[1], simulated_result[1], sizeof(float) * 3);
        memcpy(trans[2], simulated_result[2], sizeof(float) * 3);

        // replace the translation part of the matrix with simlation result directly
        memcpy(trans[3], simulated_result[3], sizeof(float) * 3);

    } else {
        trans = *m_DrawBatchContext[index].node->GetCalculatedTransform();
    }

    pbc.objectMatrix = trans;

    if (m_DrawBatchContext[index].material) {
        Color color = m_DrawBatchContext[index].material->GetBaseColor();
        if (color.ValueMap) {
            pbc.usingDiffuseMap = true;
        } else {
            pbc.diffuseColor = color.Value;
            pbc.usingDiffuseMap = false;
        }

        color = m_DrawBatchContext[index].material->GetSpecularColor();
        if (color.ValueMap) {
            pbc.specularColor = Vector4f(-1.0f);
        } else {
            pbc.specularColor = color.Value;
        }

        Parameter param = m_DrawBatchContext[index].material->GetSpecularPower();
        pbc.specularPower = param.Value;
    }

    memcpy(m_pCbvDataBegin + m_nFrameIndex * m_kSizeConstantBufferPerFrame              // offset by frame index
                + m_kSizePerFrameConstantBuffer                                         // offset by per frame buffer 
                + index * m_kSizePerBatchConstantBuffer,                                // offset by object index 
		&pbc, sizeof(pbc));
    return true;
}


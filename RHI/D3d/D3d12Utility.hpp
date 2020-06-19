#include <d3d12.h>
#include <DXGI1_4.h>

namespace My {
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
		*ppAdapter = nullptr;
		for (UINT adapterIndex = 0; ; ++adapterIndex)
		{
			IDXGIAdapter1* pAdapter = nullptr;
			if (DXGI_ERROR_NOT_FOUND == pFactory->EnumAdapters1(adapterIndex, &pAdapter))
			{
				// No more adapters to enumerate.
				break;
			}

			// Check to see if the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr)))
			{
				*ppAdapter = pAdapter;
				return;
			}
			pAdapter->Release();
		}
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
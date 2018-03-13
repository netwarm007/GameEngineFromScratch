#include "MemoryManager.hpp"
#include <cstdlib>

#ifndef ALIGN
#define ALIGN(x, a)         (((x) + ((a) - 1)) & ~((a) - 1))
#endif

using namespace My;

int MemoryManager::Initialize()
{
    if (!m_bInitialized) {
        // initialize block size lookup table
        m_pBlockSizeLookup = new size_t[kMaxBlockSize + 1];
        size_t j = 0;
        for (size_t i = 0; i <= kMaxBlockSize; i++) {
            if (i > kBlockSizes[j]) ++j;
            m_pBlockSizeLookup[i] = j;
        }

        // initialize the allocators
        m_pBlockAllocators = new BlockAllocator[kNumBlockSizes];
        for (size_t i = 0; i < kNumBlockSizes; i++) {
            m_pBlockAllocators[i].Reset(kBlockSizes[i], kPageSize, kAlignment);
        }

        m_bInitialized = true;
    }

    return 0;
}

void MemoryManager::Finalize()
{
    delete[] m_pBlockAllocators;
    delete[] m_pBlockSizeLookup;
    m_bInitialized = false;
}

void MemoryManager::Tick()
{
}

IAllocator* MemoryManager::LookUpAllocator(size_t size)
{
    // check eligibility for lookup
    if (size <= kMaxBlockSize)
        return m_pBlockAllocators + m_pBlockSizeLookup[size];
    else
        return nullptr;
}

void* MemoryManager::Allocate(size_t size)
{
    IAllocator* pAlloc = LookUpAllocator(size);
    if (pAlloc)
        return pAlloc->Allocate(size);
    else
        return malloc(size);
}

void* MemoryManager::Allocate(size_t size, size_t alignment)
{
    uint8_t* p;
    size += alignment;
    IAllocator* pAlloc = LookUpAllocator(size);
    if (pAlloc)
        p = reinterpret_cast<uint8_t*>(pAlloc->Allocate(size));
    else
        p = reinterpret_cast<uint8_t*>(malloc(size));

    p = reinterpret_cast<uint8_t*>(ALIGN(reinterpret_cast<size_t>(p), alignment));
    
    return static_cast<void*>(p);
}

void MemoryManager::Free(void* p, size_t size)
{
    if (m_bInitialized) {
        IAllocator* pAlloc = LookUpAllocator(size);
        if (pAlloc)
            pAlloc->Free(p);
        else
            free(p);
    }
}


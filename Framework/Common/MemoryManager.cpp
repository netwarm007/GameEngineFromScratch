#include "MemoryManager.hpp"

extern "C" void* malloc(size_t size);
extern "C" void  free(void* p);

#ifndef ALIGN
#define ALIGN(x, a)         (((x) + ((a) - 1)) & ~((a) - 1))
#endif

using namespace My;

namespace My {
    static const uint32_t kBlockSizes[] = {
        // 4-increments
        4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
        52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 

        // 32-increments
        128, 160, 192, 224, 256, 288, 320, 352, 384, 
        416, 448, 480, 512, 544, 576, 608, 640, 

        // 64-increments
        704, 768, 832, 896, 960, 1024
    };

    static const uint32_t kPageSize  = 8192;
    static const uint32_t kAlignment = 4;

    // number of elements in the block size array
    static const uint32_t kNumBlockSizes = 
        sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);

    // largest valid block size
    static const uint32_t kMaxBlockSize = 
        kBlockSizes[kNumBlockSizes - 1];

    size_t*        MemoryManager::m_pBlockSizeLookup;
    Allocator*     MemoryManager::m_pAllocators;
}

int My::MemoryManager::Initialize()
{
    // one-time initialization
    static bool s_bInitialized = false;
    if (!s_bInitialized) {
        // initialize block size lookup table
        m_pBlockSizeLookup = new size_t[kMaxBlockSize + 1];
        size_t j = 0;
        for (size_t i = 0; i <= kMaxBlockSize; i++) {
            if (i > kBlockSizes[j]) ++j;
            m_pBlockSizeLookup[i] = j;
        }

        // initialize the allocators
        m_pAllocators = new Allocator[kNumBlockSizes];
        for (size_t i = 0; i < kNumBlockSizes; i++) {
            m_pAllocators[i].Reset(kBlockSizes[i], kPageSize, kAlignment);
        }

        s_bInitialized = true;
    }

    return 0;
}

void My::MemoryManager::Finalize()
{
    delete[] m_pAllocators;
    delete[] m_pBlockSizeLookup;
}

void My::MemoryManager::Tick()
{
}

Allocator* My::MemoryManager::LookUpAllocator(size_t size)
{
    // check eligibility for lookup
    if (size <= kMaxBlockSize)
        return m_pAllocators + m_pBlockSizeLookup[size];
    else
        return nullptr;
}

void* My::MemoryManager::Allocate(size_t size)
{
    Allocator* pAlloc = LookUpAllocator(size);
    if (pAlloc)
        return pAlloc->Allocate();
    else
        return malloc(size);
}

void* My::MemoryManager::Allocate(size_t size, size_t alignment)
{
    uint8_t* p;
    size += alignment;
    Allocator* pAlloc = LookUpAllocator(size);
    if (pAlloc)
        p = reinterpret_cast<uint8_t*>(pAlloc->Allocate());
    else
        p = reinterpret_cast<uint8_t*>(malloc(size));

    p = reinterpret_cast<uint8_t*>(ALIGN(reinterpret_cast<size_t>(p), alignment));
    
    return static_cast<void*>(p);
}

void My::MemoryManager::Free(void* p, size_t size)
{
    Allocator* pAlloc = LookUpAllocator(size);
    if (pAlloc)
        pAlloc->Free(p);
    else
        free(p);
}


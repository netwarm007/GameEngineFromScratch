#pragma once
#include <new>
#include "IRuntimeModule.hpp"
#include "BlockAllocator.hpp"

namespace My {
    class MemoryManager : implements IRuntimeModule
    {
    public:
        template<class T, typename... Arguments>
        T* New(Arguments... parameters)
        {
            return new (Allocate(sizeof(T))) T(parameters...);
        }

        template<class T>
        void Delete(T* p)
        {
            p->~T();
            Free(p, sizeof(T));
        }

    public:
        virtual ~MemoryManager() {}

        virtual int Initialize();
        virtual void Finalize();
        virtual void Tick();

        void* Allocate(size_t size);
        void* Allocate(size_t size, size_t alignment);
        void  Free(void* p, size_t size);
    private:
        size_t*          m_pBlockSizeLookup;
        BlockAllocator*  m_pBlockAllocators;
        bool             m_bInitialized = false;
        
        const uint32_t kBlockSizes[47] = {
            // 4-increments
            4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
            52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 

            // 32-increments
            128, 160, 192, 224, 256, 288, 320, 352, 384, 
            416, 448, 480, 512, 544, 576, 608, 640, 

            // 64-increments
            704, 768, 832, 896, 960, 1024
        };

        const uint32_t kPageSize  = 8192;
        const uint32_t kAlignment = 4;

        // number of elements in the block size array
        const uint32_t kNumBlockSizes = 
            sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);

        // largest valid block size
        const uint32_t kMaxBlockSize = 
            kBlockSizes[kNumBlockSizes - 1];

    private:
        IAllocator* LookUpAllocator(size_t size);

    };

    extern MemoryManager*   g_pMemoryManager;
}


#pragma once
#include <map>
#include <new>
#include "IMemoryManager.hpp"
#include "portable.hpp"

namespace My {
    class MemoryManager : implements IMemoryManager
    {
    public:
        ~MemoryManager() {}

        int Initialize();
        void Finalize();
        void Tick();

        void* AllocatePage(size_t size);
        void  FreePage(void* p);

    protected:
        ENUM(MemoryType)
        {
            CPU = "CPU"_i32,
            GPU = "GPU"_i32
        };

        struct MemoryAllocationInfo 
        {
            size_t PageSize;
            MemoryType PageMemoryType;
        };

        std::map<void*, MemoryAllocationInfo> m_mapMemoryAllocationInfo;
    };
}


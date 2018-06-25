#pragma once
#include "IRuntimeModule.hpp"

namespace My {
    Interface IMemoryManager : implements IRuntimeModule
    {
    public:
        virtual int Initialize() = 0;
        virtual void Finalize() = 0;
        virtual void Tick() = 0;

        virtual void* AllocatePage(size_t size) = 0;
        virtual void  FreePage(void* p) = 0;
    };

    extern IMemoryManager*   g_pMemoryManager;
}
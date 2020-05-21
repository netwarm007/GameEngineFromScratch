#pragma once
#include <cstddef>

#include "IRuntimeModule.hpp"

namespace My {
Interface IMemoryManager : inherits IRuntimeModule {
   public:
    int Initialize() override = 0;
    void Finalize() override = 0;
    void Tick() override = 0;

    virtual void* AllocatePage(size_t size) = 0;
    virtual void FreePage(void* p) = 0;
};

extern IMemoryManager* g_pMemoryManager;
}  // namespace My
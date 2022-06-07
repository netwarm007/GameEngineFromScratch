#pragma once
#include <cstddef>

#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IMemoryManager : _inherits_ IRuntimeModule {
   public:
    IMemoryManager() = default;
    virtual ~IMemoryManager() = default;
    virtual void* AllocatePage(size_t size) = 0;
    virtual void FreePage(void* p) = 0;
};
}  // namespace My
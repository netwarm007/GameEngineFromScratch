#pragma once
#include <cstddef>

#include "Interface.hpp"

namespace My {
_Interface_ IMemoryManager {
   public:
    IMemoryManager() = default;
    virtual ~IMemoryManager() = default;
    virtual void* AllocatePage(size_t size) = 0;
    virtual void FreePage(void* p) = 0;
};
}  // namespace My
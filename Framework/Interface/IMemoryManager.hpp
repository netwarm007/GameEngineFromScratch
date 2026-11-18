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

    // Pages stricter than the default malloc alignment need this entry point.
    // Alignment must be a power of two. Pages are released with FreePage.
    // Returns nullptr when the request cannot be satisfied.
    virtual void* AllocateAlignedPage(size_t size, size_t alignment) {
        return alignment <= alignof(std::max_align_t) ? AllocatePage(size)
                                                      : nullptr;
    }
};
}  // namespace My
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#include "IAllocator.hpp"

namespace My {

// Process-wide hook that lets small pilot objects use the engine allocator
// without carrying a pointer to it. Every block gets a 16-byte header naming
// the allocator generation that produced it, so a block is always returned
// to its origin: the default heap, the hooked allocator, or (when that
// allocator is gone or replaced) nowhere. Such a block is leaked on purpose,
// which is safer than freeing it into the wrong allocator.
//
// SetEngineAllocator must only be called while no other thread allocates or
// frees through the hook (application startup and shutdown). The allocator
// must be thread-safe and 16-byte aligned, and must outlive its use unless it
// is cleared first.
void SetEngineAllocator(IAllocator* allocator);

// Alignment up to 16 bytes. Falls back to the default heap when no allocator
// is hooked or it is exhausted. Throws std::bad_alloc on failure.
void* EngineAllocate(size_t size);
void EngineFree(void* p);

// Blocks served by the hooked allocator that are still live (for tests).
size_t EngineAllocationCount();

template <typename T>
struct EngineStlAllocator {
    using value_type = T;
    static_assert(alignof(T) <= 16, "over-aligned types are not supported");

    EngineStlAllocator() = default;
    template <typename U>
    EngineStlAllocator(const EngineStlAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        if (n > SIZE_MAX / sizeof(T)) throw std::bad_array_new_length();
        return static_cast<T*>(EngineAllocate(n * sizeof(T)));
    }
    void deallocate(T* p, size_t) noexcept { EngineFree(p); }

    template <typename U>
    bool operator==(const EngineStlAllocator<U>&) const noexcept {
        return true;
    }
};

// std::make_shared whose control block and object come from the hook.
template <typename T, typename... Args>
std::shared_ptr<T> MakeEngineShared(Args&&... args) {
    return std::allocate_shared<T>(EngineStlAllocator<T>(),
                                   std::forward<Args>(args)...);
}
}  // namespace My

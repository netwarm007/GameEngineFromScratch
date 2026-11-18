#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

// Fixed-capacity bump (arena) allocator. Not thread-safe.
//
// Allocation is a pointer increment; individual frees are not supported
// (Free is a no-op) and memory is recycled all at once with FreeAll(), which
// keeps the backing memory. Release() or the destructor returns it to the
// memory manager. Allocate returns nullptr for zero sizes, invalid alignment,
// exhaustion, or an unconfigured allocator.
class LinearAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kDefaultAlignment = 16;

    explicit LinearAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    LinearAllocator(IMemoryManager* pMmgr, size_t capacity);
    ~LinearAllocator() override;
    // disable copy & assignment
    LinearAllocator(const LinearAllocator&) = delete;
    LinearAllocator& operator=(const LinearAllocator&) = delete;

    // Releases any current memory and acquires a new arena. Returns false
    // (leaving the allocator unconfigured) if capacity is zero or the memory
    // manager cannot provide it.
    bool Initialize(size_t capacity);
    // Returns the backing memory; the allocator becomes unconfigured.
    void Release();

    void* Allocate(size_t size) override {
        return Allocate(size, kDefaultAlignment);
    }
    // alignment must be a power of two
    void* Allocate(size_t size, size_t alignment);
    void Free(void*) override {}
    // Discards all allocations; capacity is kept.
    void FreeAll() override;

    bool IsConfigured() const { return m_pBase != nullptr; }
    size_t Capacity() const { return m_szCapacity; }
    size_t Used() const { return m_szUsed; }
    size_t Remaining() const { return m_szCapacity - m_szUsed; }
    size_t PeakUsed() const { return m_szPeak; }
    size_t AllocationCount() const { return m_nAllocations; }
    size_t FailedAllocationCount() const { return m_nFailures; }

   private:
    uint8_t* m_pBase = nullptr;
    size_t m_szCapacity = 0;
    size_t m_szUsed = 0;
    size_t m_szPeak = 0;
    size_t m_nAllocations = 0;
    size_t m_nFailures = 0;
};
}  // namespace My

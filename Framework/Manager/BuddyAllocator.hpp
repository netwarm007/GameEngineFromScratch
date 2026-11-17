#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

// Binary buddy allocator over a power-of-two arena. Not thread-safe.
//
// Blocks are power-of-two multiples of the minimum block size, each aligned to
// its own size relative to the arena base, so a block of size S is S-aligned
// (up to the arena's own alignment, kBaseAlignment). Freeing merges a block
// with its buddy repeatedly. Block state lives in a separate byte-per-unit
// table, so user memory carries no headers and Free needs no size.
//
// Allocate returns nullptr for zero sizes, invalid alignment, exhaustion, or
// an unconfigured allocator. Free(nullptr) is a no-op; pointers that are
// outside the arena, not at a block start, or already free are rejected and
// counted. Internal fragmentation is up to 50% per allocation.
class BuddyAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kBaseAlignment = 4096;
    static constexpr size_t kDefaultMinBlock = 64;
    static constexpr size_t kMaxOrders = 64;

    explicit BuddyAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    BuddyAllocator(IMemoryManager* pMmgr, size_t capacity,
                   size_t minBlock = kDefaultMinBlock);
    ~BuddyAllocator() override;
    BuddyAllocator(const BuddyAllocator&) = delete;
    BuddyAllocator& operator=(const BuddyAllocator&) = delete;

    // Replaces the arena. capacity and minBlock must be powers of two with
    // 16 <= minBlock <= capacity. Returns false (leaving the allocator
    // unconfigured) otherwise or when the memory manager fails.
    bool Initialize(size_t capacity, size_t minBlock = kDefaultMinBlock);
    void Release();

    void* Allocate(size_t size) override { return Allocate(size, 1); }
    // alignment up to min(capacity, kBaseAlignment) is honored by rounding the
    // block up to that size.
    void* Allocate(size_t size, size_t alignment);
    void Free(void* p) override;
    void FreeAll() override;

    // Size of the block a request of this size receives; 0 if it cannot fit.
    size_t BlockSizeFor(size_t size) const;
    // Walks the tables and free lists; true if all invariants hold.
    bool Check() const;

    bool IsConfigured() const { return m_pBase != nullptr; }
    size_t Capacity() const { return m_szCapacity; }
    size_t MinBlockSize() const { return m_szMinBlock; }
    // Bytes in live blocks (block sizes, not requested sizes).
    size_t Used() const { return m_szUsed; }
    size_t PeakUsed() const { return m_szPeak; }
    size_t LargestFreeBlock() const;
    size_t AllocationCount() const { return m_nAllocations; }
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidFreeCount() const { return m_nInvalidFrees; }

   private:
    struct FreeNode {
        FreeNode* next;
        FreeNode* prev;
    };

    size_t UnitOf(const void* p) const {
        return (static_cast<const uint8_t*>(p) - m_pBase) >> m_nMinShift;
    }
    uint8_t* AddressOf(size_t unit) const {
        return m_pBase + (unit << m_nMinShift);
    }
    void PushFree(size_t unit, size_t order);
    void RemoveFree(size_t unit, size_t order);

    uint8_t* m_pBase = nullptr;
    // 0: inside a block; order + 1: live block start; 0x80 | (order + 1):
    // free block start
    uint8_t* m_pState = nullptr;
    size_t m_szCapacity = 0;
    size_t m_szMinBlock = 0;
    size_t m_nMinShift = 0;
    size_t m_nOrders = 0;  // order count; the arena itself is order m_nOrders-1
    size_t m_szUsed = 0;
    size_t m_szPeak = 0;
    size_t m_nAllocations = 0;
    size_t m_nFailures = 0;
    size_t m_nInvalidFrees = 0;
    FreeNode* m_FreeLists[kMaxOrders] = {};
};
}  // namespace My

#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

// Two-Level Segregated Fit allocator: general-purpose, fixed-capacity arena
// with O(1) allocate and free and immediate coalescing. Not thread-safe.
//
// Free blocks are binned by size into a first level (power of two) and a
// second level (32 linear subdivisions); two bitmaps locate a suitable bin
// without searching. Every block carries a 16-byte header, and every result
// is at least 16-byte aligned. Larger alignments are honored by splitting off
// the leading gap as a free block.
//
// Lookup is good-fit: a request is rounded up to the next bin so any block in
// the bin fits. A request within about 1/32 of the only free block can
// therefore fail even though the block would hold it.
//
// Allocate returns nullptr for zero sizes, invalid alignment, exhaustion, or
// an unconfigured allocator. Free(nullptr) is a no-op; Free rejects and counts
// pointers it can recognise as invalid: outside the arena, misaligned, not at
// a block start, or already free. The checks inspect block headers, so a
// pointer into user data crafted to look like a header can fool them.
class TlsfAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kAlignment = 16;
    static constexpr size_t kHeaderSize = 16;
    static constexpr size_t kMinBlockSize = 32;
    static constexpr size_t kMinCapacity = 4 * kAlignment;

    explicit TlsfAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    TlsfAllocator(IMemoryManager* pMmgr, size_t capacity);
    ~TlsfAllocator() override;
    TlsfAllocator(const TlsfAllocator&) = delete;
    TlsfAllocator& operator=(const TlsfAllocator&) = delete;

    // Replaces the arena. Capacity is rounded down to a multiple of 16 and
    // must be at least kMinCapacity. Returns false (leaving the allocator
    // unconfigured) on a bad capacity or when the memory manager fails.
    bool Initialize(size_t capacity);
    void Release();

    void* Allocate(size_t size) override { return Allocate(size, kAlignment); }
    void* Allocate(size_t size, size_t alignment);
    void Free(void* p) override;
    // Drops every allocation; the arena becomes one free block.
    void FreeAll() override;

    // Walks the whole heap and the bins; true if all invariants hold.
    bool Check() const;

    bool IsConfigured() const { return m_pBase != nullptr; }
    size_t Capacity() const { return m_szCapacity; }
    // Bytes held by live blocks, headers included.
    size_t Used() const { return m_szUsed; }
    size_t PeakUsed() const { return m_szPeak; }
    size_t FreeBytes() const { return m_szCapacity - kHeaderSize - m_szUsed; }
    size_t LargestFreeBlock() const;
    size_t AllocationCount() const { return m_nAllocations; }
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidFreeCount() const { return m_nInvalidFrees; }

   private:
    struct Block;
    static constexpr int kSlLog2 = 5;
    static constexpr size_t kSlCount = size_t(1) << kSlLog2;
    static constexpr size_t kSmallLimit = 256;
    static constexpr size_t kFlCount = 64;

    static void Mapping(size_t size, size_t& fl, size_t& sl);
    Block* FindSuitable(size_t size);
    void InsertFree(Block* b);
    void RemoveFree(Block* b);
    Block* Next(const Block* b) const;
    bool IsValidUsedBlock(const Block* b) const;

    uint8_t* m_pBase = nullptr;
    size_t m_szCapacity = 0;
    size_t m_szUsed = 0;
    size_t m_szPeak = 0;
    size_t m_nAllocations = 0;
    size_t m_nFailures = 0;
    size_t m_nInvalidFrees = 0;

    uint64_t m_FlBitmap = 0;
    uint32_t m_SlBitmap[kFlCount] = {};
    Block* m_Bins[kFlCount][kSlCount] = {};
};
}  // namespace My

#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

// Fixed-capacity circular allocator with FIFO reclamation. Not thread-safe.
//
// Blocks are contiguous; a request that does not fit before the end of the
// buffer wraps to the start and the skipped tail counts as used. Memory is
// reclaimed in allocation order: capture GetHead() after a batch of
// allocations (for example once per frame) and pass it to Retire() when the
// consumer, such as the GPU, has finished with that batch. Allocate returns
// nullptr when the buffer is full; it never overwrites live data.
//
// Positions are monotonically increasing 64-bit logical offsets, so stale or
// out-of-range positions passed to Retire() are detected.
class RingAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kDefaultAlignment = 16;

    explicit RingAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    RingAllocator(IMemoryManager* pMmgr, size_t capacity);
    ~RingAllocator() override;
    // disable copy & assignment
    RingAllocator(const RingAllocator&) = delete;
    RingAllocator& operator=(const RingAllocator&) = delete;

    bool Initialize(size_t capacity);
    void Release();

    void* Allocate(size_t size) override {
        return Allocate(size, kDefaultAlignment);
    }
    // alignment must be a power of two
    void* Allocate(size_t size, size_t alignment);
    // Individual frees are not supported; use Retire().
    void Free(void*) override {}
    // Reclaims everything.
    void FreeAll() override;

    // Position after the newest allocation.
    uint64_t GetHead() const { return m_nHead; }
    // Reclaims all memory allocated before position. Returns false (and
    // changes nothing) if position is older than the last retired position or
    // newer than the head.
    bool Retire(uint64_t position);

    bool IsConfigured() const { return m_pBase != nullptr; }
    size_t Capacity() const { return m_szCapacity; }
    size_t Used() const { return static_cast<size_t>(m_nHead - m_nTail); }
    size_t Remaining() const { return m_szCapacity - Used(); }
    size_t PeakUsed() const { return m_szPeak; }
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidRetireCount() const { return m_nInvalidRetires; }

   private:
    uint8_t* m_pBase = nullptr;
    size_t m_szCapacity = 0;
    uint64_t m_nHead = 0;
    uint64_t m_nTail = 0;
    size_t m_szPeak = 0;
    size_t m_nFailures = 0;
    size_t m_nInvalidRetires = 0;
};
}  // namespace My

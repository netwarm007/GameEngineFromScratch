#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "IAllocator.hpp"
#include "ThreadCacheAllocator.hpp"
#include "TlsfAllocator.hpp"

namespace My {

// Thread-safe general-purpose facade: requests of 1..kSmallLimit bytes with at
// most 16-byte alignment go to a ThreadCacheAllocator, everything else goes to
// a TlsfAllocator arena guarded by a mutex. Free() routes by address: pointers
// inside the TLSF arena go to TLSF, all others to the thread cache, which
// rejects pointers it does not recognise.
//
// Allocate(size, alignment) accepts any power-of-two alignment; alignments
// above 16 are served by TLSF even for small sizes. When the small path
// cannot get memory (page source exhausted) the request falls back to TLSF.
//
// Limits: the TLSF arena has a fixed capacity; double frees of small objects
// are not detected (see ThreadCacheAllocator); FreeAll() must not run
// concurrently with other calls.
class RoutingAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kSmallLimit = ThreadCacheAllocator::kMaxSize;
    static constexpr size_t kAlignment = TlsfAllocator::kAlignment;

    RoutingAllocator(IMemoryManager* pMmgr, size_t largeCapacity,
                     size_t slabSize = SizeClassAllocator::kDefaultSlabSize,
                     uint32_t maxCachedPerClass =
                         ThreadCacheAllocator::kDefaultMaxCached);
    ~RoutingAllocator() override = default;
    RoutingAllocator(const RoutingAllocator&) = delete;
    RoutingAllocator& operator=(const RoutingAllocator&) = delete;

    bool IsConfigured() const {
        return m_Small.IsConfigured() && m_Large.IsConfigured();
    }

    void* Allocate(size_t size) override { return Allocate(size, kAlignment); }
    void* Allocate(size_t size, size_t alignment);
    void Free(void* p) override;
    void FreeAll() override;

    // Returns the calling thread's small-object cache to the backing store.
    void FlushThreadCache() { m_Small.FlushThreadCache(); }
    // Flushes the calling thread's cache and releases empty slabs.
    size_t Trim() { return m_Small.Trim(); }

    // Statistics. Large* are read under the lock.
    size_t SmallLiveCount() const { return m_Small.BackingLiveCount(); }
    size_t LargeAllocationCount() const;
    size_t LargeUsed() const;
    size_t LargeCapacity() const;
    size_t InvalidFreeCount() const;
    // Requests that no path could satisfy (valid size and alignment only).
    size_t FailedAllocationCount() const { return m_nFailures; }
    uint64_t SmallRouteCount() const { return m_nSmall; }
    uint64_t LargeRouteCount() const { return m_nLarge; }

   private:
    ThreadCacheAllocator m_Small;
    TlsfAllocator m_Large;
    mutable std::mutex m_LargeMutex;
    std::atomic<uint64_t> m_nSmall{0};
    std::atomic<uint64_t> m_nLarge{0};
    std::atomic<size_t> m_nFailures{0};
};
}  // namespace My

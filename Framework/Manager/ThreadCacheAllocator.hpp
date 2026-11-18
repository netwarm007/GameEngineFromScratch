#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "IAllocator.hpp"
#include "SizeClassAllocator.hpp"

namespace My {

// Thread-safe small-object allocator: a SizeClassAllocator shared by all
// threads behind a mutex, fronted by a per-thread cache for each size class.
// The common Allocate/Free path touches only the calling thread's cache and
// takes no lock; the lock is taken to move a batch of objects between a
// cache and the shared backing allocator.
//
// Requests of 1..SizeClassAllocator::kMaxSize bytes are served (16-byte
// aligned); anything else returns nullptr. Objects may be freed by a thread
// other than the allocating one; they join that thread's cache.
//
// Lifetime rules:
//  - A thread's cache is returned to the backing allocator when the thread
//    exits, or on FlushThreadCache(). Caches of threads still running when
//    the allocator is destroyed are discarded together with the backing
//    memory, so no object may be used after destruction.
//  - FreeAll() invalidates every object and every thread's cache; it must not
//    run concurrently with other calls on this allocator.
//  - Free() only accepts live objects of this allocator. Pointers from other
//    allocators are rejected when their slab header is recognised, and
//    double frees are NOT detected (an object in a cache still looks live);
//    wrap the allocator in a TrackingAllocator to catch both.
class ThreadCacheAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kMaxSize = SizeClassAllocator::kMaxSize;
    static constexpr uint32_t kDefaultMaxCached = 64;

    // maxCachedPerClass bounds each thread's cache per size class; when it is
    // exceeded half of the bound is flushed to the backing allocator.
    explicit ThreadCacheAllocator(
        IMemoryManager* pMmgr,
        size_t slabSize = SizeClassAllocator::kDefaultSlabSize,
        uint32_t maxCachedPerClass = kDefaultMaxCached);
    ~ThreadCacheAllocator() override;
    ThreadCacheAllocator(const ThreadCacheAllocator&) = delete;
    ThreadCacheAllocator& operator=(const ThreadCacheAllocator&) = delete;

    bool IsConfigured() const { return m_bConfigured; }

    void* Allocate(size_t size) override;
    void Free(void* p) override;
    void FreeAll() override;

    // Returns the calling thread's cached objects to the backing allocator.
    void FlushThreadCache();
    // Flushes the calling thread's cache, then gives empty slabs back to the
    // memory manager; returns the slab count released. Other threads' cached
    // objects keep their slabs alive.
    size_t Trim();

    // Objects held by the backing allocator as live, including objects parked
    // in thread caches.
    size_t BackingLiveCount() const;
    size_t BackingSlabCount() const;
    // Objects cached by the calling thread.
    size_t CachedCount() const;
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidFreeCount() const;
    // Batches moved from the backing allocator into a cache, and back.
    uint64_t RefillCount() const { return m_nRefills; }
    uint64_t FlushCount() const { return m_nFlushes; }

   private:
    struct ThreadCache;
    struct ThreadCacheList;
    friend struct ThreadCacheList;

    ThreadCache* GetCache();
    void Sync(ThreadCache& cache);
    void FlushClass(ThreadCache& cache, size_t index, uint32_t keep);
    void FlushAllClasses(ThreadCache& cache);

    SizeClassAllocator m_Backing;
    mutable std::mutex m_Mutex;
    uint64_t m_nId;
    uint32_t m_nMaxCached;
    uint32_t m_nBatch;
    bool m_bConfigured = false;
    std::atomic<uint64_t> m_nGeneration{0};
    std::atomic<size_t> m_nFailures{0};
    std::atomic<size_t> m_nInvalidFrees{0};
    std::atomic<uint64_t> m_nRefills{0};
    std::atomic<uint64_t> m_nFlushes{0};
};
}  // namespace My

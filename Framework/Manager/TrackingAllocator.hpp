#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "IAllocator.hpp"

namespace My {

// Debug and statistics decorator for any IAllocator.
//
// Every live allocation is recorded, so the decorator can
//  - reject frees of unknown or already freed pointers without forwarding
//    them to the wrapped allocator (protecting allocators whose own checks
//    are weak or disabled),
//  - detect writes past the end of a block with a guard band behind each
//    allocation, checked on Free and by CheckGuards(),
//  - keep statistics and produce a leak report.
// All decorator state is guarded by a mutex that is also held while calling
// the wrapped allocator, which therefore becomes safe to share between
// threads as long as it is only used through the decorator.
//
// The wrapped allocator must outlive the decorator and is not owned. Results
// have the wrapped allocator's alignment (guard bytes are placed after the
// user bytes only). The decorator allocates its bookkeeping from the C++
// heap, so it is meant for debug and tooling builds.
class TrackingAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kGuardSize = 16;
    static constexpr uint8_t kGuardByte = 0xFB;

    struct Stats {
        size_t allocations = 0;      // successful, cumulative
        size_t frees = 0;            // successful, cumulative
        size_t failedAllocations = 0;
        size_t invalidFrees = 0;     // unknown or double frees
        size_t guardViolations = 0;  // corrupted guard bands found
        size_t liveCount = 0;
        size_t liveBytes = 0;        // requested bytes of live blocks
        size_t peakLiveBytes = 0;
        size_t peakLiveCount = 0;
        uint64_t totalBytes = 0;     // requested bytes, cumulative
    };

    struct Leak {
        const void* address;
        size_t size;
        const char* tag;  // may be nullptr
        uint64_t sequence;  // allocation order, starting at 1
    };

    using LeakHandler = std::function<void(const std::string&)>;

    // name labels the report; it is copied.
    TrackingAllocator(IMemoryManager* pMmgr, IAllocator& inner,
                      std::string name = "allocator");
    // Reports leaks to the leak handler, if one is set and blocks are live.
    ~TrackingAllocator() override;
    TrackingAllocator(const TrackingAllocator&) = delete;
    TrackingAllocator& operator=(const TrackingAllocator&) = delete;

    void* Allocate(size_t size) override { return AllocateTagged(size, nullptr); }
    // tag must outlive the allocation (typically a string literal).
    void* AllocateTagged(size_t size, const char* tag);
    // Free(nullptr) is a no-op.
    void Free(void* p) override;
    // Forgets every record and forwards to the wrapped allocator.
    void FreeAll() override;

    // Number of live blocks whose guard band was overwritten.
    size_t CheckGuards() const;

    Stats GetStats() const;
    // Live blocks ordered by allocation sequence.
    std::vector<Leak> Leaks() const;
    // Human readable summary; the first line states the leak count.
    std::string LeakReport() const;
    void SetLeakHandler(LeakHandler handler);

   private:
    struct Record {
        size_t size;
        const char* tag;
        uint64_t sequence;
    };

    bool GuardIntact(const void* p, size_t size) const;

    IAllocator& m_Inner;
    std::string m_Name;
    mutable std::mutex m_Mutex;
    std::unordered_map<const void*, Record> m_Live;
    Stats m_Stats;
    uint64_t m_nSequence = 0;
    LeakHandler m_LeakHandler;
};
}  // namespace My

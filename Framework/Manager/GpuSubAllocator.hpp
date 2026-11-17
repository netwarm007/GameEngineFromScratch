#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

namespace My {

// Handle to a range of a GPU heap or buffer. The offset is relative to the
// heap; the allocator never touches the memory itself.
struct GpuAllocation {
    uint64_t offset = 0;
    uint64_t size = 0;  // requested size
    uint64_t id = 0;    // 0 means invalid; detects stale handles
    bool IsValid() const { return id != 0; }
};

// Offset allocator for GPU memory (a Vulkan/D3D12/Metal heap or a large
// buffer): all bookkeeping lives in CPU memory, so it works for memory that
// the CPU cannot address. Not thread-safe.
//
// Best-fit over free ranges kept in an ordered map (neighbours coalesce on
// free) and a size-ordered set. Operations are O(log n) in the number of free
// ranges; an aligned request may inspect a few larger candidates. Sizes are
// rounded up to the granularity, and any alignment (power of two) is honoured
// by splitting the leading gap off as a free range.
//
// Fence-based reclamation: FreeDeferred(a, fence) keeps the range reserved
// until Retire(completed) is called with completed >= fence, so memory still
// referenced by in-flight GPU work is never reused. Fence values are caller
// defined (frame numbers or timeline-semaphore values) and need not be
// monotonic across FreeDeferred calls.
class GpuSubAllocator {
   public:
    static constexpr uint64_t kDefaultGranularity = 256;

    // capacity and granularity must be non-zero; otherwise IsConfigured() is
    // false and every allocation fails.
    explicit GpuSubAllocator(uint64_t capacity,
                             uint64_t granularity = kDefaultGranularity);

    bool IsConfigured() const { return m_bConfigured; }

    // alignment must be a power of two. On failure out is invalid.
    bool Allocate(uint64_t size, uint64_t alignment, GpuAllocation& out);
    // Immediate release. Stale, foreign, double-freed or already deferred
    // handles are rejected and counted.
    bool Free(const GpuAllocation& a);
    // Releases once Retire() has been called with a value >= fence. A fence
    // already completed releases immediately.
    bool FreeDeferred(const GpuAllocation& a, uint64_t fence);
    // Releases every deferred free whose fence <= completed; returns how many.
    size_t Retire(uint64_t completed);
    // Drops every allocation, live or deferred. Handles issued before stay
    // invalid.
    void Reset();

    uint64_t Capacity() const { return m_nCapacity; }
    uint64_t Granularity() const { return m_nGranularity; }
    uint64_t UsedBytes() const { return m_nUsed; }
    uint64_t FreeBytes() const { return m_nCapacity - m_nUsed; }
    uint64_t PeakUsedBytes() const { return m_nPeakUsed; }
    uint64_t PendingBytes() const { return m_nPendingBytes; }
    uint64_t LargestFreeRange() const;
    size_t LiveCount() const { return m_Live.size(); }
    size_t PendingCount() const { return m_Pending.size(); }
    size_t FreeRangeCount() const { return m_Free.size(); }
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidFreeCount() const { return m_nInvalidFrees; }

    // Verifies internal invariants (coverage, coalescing, index agreement).
    bool Check() const;

   private:
    struct Live {
        uint64_t reserved;
        uint64_t requested;
        uint64_t id;
        bool pending;
    };

    Live* Find(const GpuAllocation& a);
    void AddFree(uint64_t offset, uint64_t size);
    void Release(uint64_t offset);

    std::map<uint64_t, uint64_t> m_Free;             // offset -> size
    std::set<std::pair<uint64_t, uint64_t>> m_BySize;  // (size, offset)
    std::unordered_map<uint64_t, Live> m_Live;       // offset -> allocation
    std::multimap<uint64_t, uint64_t> m_Pending;     // fence -> offset

    uint64_t m_nCapacity = 0;
    uint64_t m_nGranularity = 0;
    uint64_t m_nUsed = 0;
    uint64_t m_nPeakUsed = 0;
    uint64_t m_nPendingBytes = 0;
    uint64_t m_nNextId = 1;
    uint64_t m_nCompleted = 0;
    size_t m_nFailures = 0;
    size_t m_nInvalidFrees = 0;
    bool m_bHasCompleted = false;
    bool m_bConfigured = false;
};
}  // namespace My

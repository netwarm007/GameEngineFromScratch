#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

// Fixed-capacity stack allocator with LIFO release. Not thread-safe.
//
// Each allocation is preceded by a small header, so the most recent
// allocation can be popped with Free(p). GetMarker()/FreeToMarker() release
// everything allocated after a point in one step. Out-of-order Free calls,
// foreign pointers and markers newer than the current top are rejected and
// counted instead of corrupting the stack. A marker is only meaningful while
// the stack has not been unwound below it.
class StackAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kDefaultAlignment = 16;

    struct Marker {
        size_t top = 0;
        size_t last = SIZE_MAX;
    };

    explicit StackAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    StackAllocator(IMemoryManager* pMmgr, size_t capacity);
    ~StackAllocator() override;
    // disable copy & assignment
    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;

    bool Initialize(size_t capacity);
    void Release();

    void* Allocate(size_t size) override {
        return Allocate(size, kDefaultAlignment);
    }
    // alignment must be a power of two
    void* Allocate(size_t size, size_t alignment);
    // Pops the most recent allocation; Free(nullptr) is a no-op.
    void Free(void* p) override;
    // Empties the stack; capacity is kept.
    void FreeAll() override;

    Marker GetMarker() const { return {m_szTop, m_szLast}; }
    // Returns false (and changes nothing) if the marker is newer than the
    // current top.
    bool FreeToMarker(const Marker& marker);

    bool IsConfigured() const { return m_pBase != nullptr; }
    size_t Capacity() const { return m_szCapacity; }
    size_t Used() const { return m_szTop; }
    size_t Remaining() const { return m_szCapacity - m_szTop; }
    size_t PeakUsed() const { return m_szPeak; }
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidFreeCount() const { return m_nInvalidFrees; }

   private:
    struct Header {
        size_t prevTop;
        size_t prevLast;
    };

    void Poison(size_t from, size_t to);

    uint8_t* m_pBase = nullptr;
    size_t m_szCapacity = 0;
    size_t m_szTop = 0;
    // offset of the most recent allocation, SIZE_MAX when empty
    size_t m_szLast = SIZE_MAX;
    size_t m_szPeak = 0;
    size_t m_nFailures = 0;
    size_t m_nInvalidFrees = 0;
};
}  // namespace My

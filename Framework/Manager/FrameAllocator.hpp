#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "IAllocator.hpp"
#include "LinearAllocator.hpp"

namespace My {

// Per-frame scratch allocator with N rotating arenas (default 2). Not
// thread-safe; BeginFrame() must not run concurrently with allocation.
//
// BeginFrame() moves to the next arena and recycles it, so memory allocated
// during the previous N-1 frames stays valid while the current frame
// allocates. Free is a no-op; FreeAll recycles every arena.
class FrameAllocator : _implements_ IAllocator {
   public:
    explicit FrameAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    FrameAllocator(IMemoryManager* pMmgr, size_t capacityPerFrame,
                   size_t frameCount = 2);
    ~FrameAllocator() override;
    // disable copy & assignment
    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;

    // Releases current memory and acquires frameCount arenas of
    // capacityPerFrame bytes each. frameCount must be at least 2. On failure
    // nothing is held and false is returned.
    bool Initialize(size_t capacityPerFrame, size_t frameCount = 2);
    void Release();

    // Advances to the next arena and discards its previous contents.
    void BeginFrame();

    void* Allocate(size_t size) override {
        return Allocate(size, LinearAllocator::kDefaultAlignment);
    }
    void* Allocate(size_t size, size_t alignment);
    void Free(void*) override {}
    void FreeAll() override;

    bool IsConfigured() const { return !m_Frames.empty(); }
    size_t FrameCount() const { return m_Frames.size(); }
    // number of BeginFrame calls since Initialize
    uint64_t FrameNumber() const { return m_nFrameNumber; }
    size_t CapacityPerFrame() const;
    size_t Used() const;
    size_t Remaining() const;
    size_t FailedAllocationCount() const;

   private:
    std::vector<std::unique_ptr<LinearAllocator>> m_Frames;
    size_t m_nCurrent = 0;
    uint64_t m_nFrameNumber = 0;
};
}  // namespace My

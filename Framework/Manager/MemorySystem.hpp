#pragma once
#include <cstddef>
#include <memory>
#include <memory_resource>

#include "FrameAllocator.hpp"
#include "IAllocator.hpp"
#include "PmrAdapter.hpp"
#include "RoutingAllocator.hpp"
#include "TrackingAllocator.hpp"

namespace My {

// Engine-wide allocator stack owned by the application: a RoutingAllocator
// over the page source, optionally wrapped by a TrackingAllocator that reports
// leaks on destruction. Must be destroyed before the IMemoryManager it draws
// pages from is finalized.
class MemorySystem {
   public:
    struct Config {
        size_t largeCapacity = 16 * 1024 * 1024;
        size_t frameCapacity = 1024 * 1024;  // per frame arena
        bool tracking = false;
    };

    MemorySystem(IMemoryManager* pMmgr, const Config& config);
    MemorySystem(const MemorySystem&) = delete;
    MemorySystem& operator=(const MemorySystem&) = delete;

    bool IsConfigured() const {
        return m_Routing.IsConfigured() && m_Frame.IsConfigured();
    }

    // Recycles the oldest frame arena; call once per frame on the main
    // thread, before anything allocates scratch memory for the frame.
    void BeginFrame() { m_Frame.BeginFrame(); }
    // Per-frame scratch memory for pmr containers. Never freed individually;
    // everything is valid only until the arena is recycled (frameCount-1
    // BeginFrame calls later). Main thread only. Throws std::bad_alloc when
    // the arena is exhausted.
    std::pmr::memory_resource& FrameResource() { return m_FrameResource; }
    FrameAllocator& Frame() { return m_Frame; }

    // The allocator clients should use: tracked when tracking is on.
    IAllocator& Allocator();
    RoutingAllocator& Routing() { return m_Routing; }
    // nullptr unless tracking is on.
    TrackingAllocator* Tracking() { return m_Tracking.get(); }

   private:
    RoutingAllocator m_Routing;
    FrameAllocator m_Frame;
    PmrAdapter<FrameAllocator> m_FrameResource;
    std::unique_ptr<TrackingAllocator> m_Tracking;  // destroyed first
};
}  // namespace My

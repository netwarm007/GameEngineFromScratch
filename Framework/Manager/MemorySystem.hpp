#pragma once
#include <cstddef>
#include <memory>

#include "IAllocator.hpp"
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
        bool tracking = false;
    };

    MemorySystem(IMemoryManager* pMmgr, const Config& config);
    MemorySystem(const MemorySystem&) = delete;
    MemorySystem& operator=(const MemorySystem&) = delete;

    bool IsConfigured() const { return m_Routing.IsConfigured(); }

    // The allocator clients should use: tracked when tracking is on.
    IAllocator& Allocator();
    RoutingAllocator& Routing() { return m_Routing; }
    // nullptr unless tracking is on.
    TrackingAllocator* Tracking() { return m_Tracking.get(); }

   private:
    RoutingAllocator m_Routing;
    std::unique_ptr<TrackingAllocator> m_Tracking;  // destroyed first
};
}  // namespace My

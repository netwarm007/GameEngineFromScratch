#include "MemorySystem.hpp"

#include <iostream>
#include <string>

namespace My {

MemorySystem::MemorySystem(IMemoryManager* pMmgr, const Config& config)
    : m_Routing(pMmgr, config.largeCapacity),
      m_Frame(pMmgr, config.frameCapacity),
      m_FrameResource(m_Frame) {
    if (config.tracking && IsConfigured()) {
        m_Tracking =
            std::make_unique<TrackingAllocator>(pMmgr, m_Routing, "engine");
        m_Tracking->SetLeakHandler(
            [](const std::string& report) { std::cerr << report << '\n'; });
    }
}

IAllocator& MemorySystem::Allocator() {
    if (m_Tracking) return *m_Tracking;
    return m_Routing;
}

}  // namespace My

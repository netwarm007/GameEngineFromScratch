#include "MemorySystem.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace My {

MemorySystem::MemorySystem(const Config& config)
    : MemorySystem(std::make_unique<MemoryManager>(), config) {}

MemorySystem::MemorySystem(std::unique_ptr<MemoryManager> pages,
                           const Config& config)
    : MemorySystem(pages.get(), config) {
    m_OwnedPages = std::move(pages);
}

MemorySystem::MemorySystem(IMemoryManager* pMmgr, const Config& config)
    : m_Routing(pMmgr, config.largeCapacity),
      m_Frame(pMmgr, config.frameCapacity),
      m_FrameResource(m_Frame),
      m_Report(config.reportHandler
                   ? config.reportHandler
                   : [](const std::string& text) { std::cerr << text << '\n'; }) {
    if (config.tracking && IsConfigured()) {
        m_Tracking =
            std::make_unique<TrackingAllocator>(pMmgr, m_Routing, "engine");
        m_Tracking->SetLeakHandler(m_Report);
    }
}

MemorySystem::~MemorySystem() {
    if (IsConfigured()) m_Report(StatsReport());
}

std::string MemorySystem::StatsReport() const {
    std::ostringstream out;
    out << "[MemorySystem] small: live " << m_Routing.SmallLiveCount()
        << ", routed " << m_Routing.SmallRouteCount()
        << "; large: live " << m_Routing.LargeAllocationCount() << ", used "
        << m_Routing.LargeUsed() << '/' << m_Routing.LargeCapacity()
        << ", routed " << m_Routing.LargeRouteCount()
        << "; failed " << m_Routing.FailedAllocationCount()
        << ", invalid frees " << m_Routing.InvalidFreeCount()
        << "\n[MemorySystem] frame: " << m_Frame.FrameCount() << " x "
        << m_Frame.CapacityPerFrame() << " B, used " << m_Frame.Used()
        << ", failed " << m_Frame.FailedAllocationCount();
    if (m_Tracking) {
        const auto st = m_Tracking->GetStats();
        out << "\n[MemorySystem] tracking: allocs " << st.allocations
            << ", frees " << st.frees << ", live " << st.liveCount << " ("
            << st.liveBytes << " B), peak " << st.peakLiveCount << " ("
            << st.peakLiveBytes << " B), invalid frees " << st.invalidFrees
            << ", guard violations " << st.guardViolations;
    }
    return out.str();
}

IAllocator& MemorySystem::Allocator() {
    if (m_Tracking) return *m_Tracking;
    return m_Routing;
}

}  // namespace My

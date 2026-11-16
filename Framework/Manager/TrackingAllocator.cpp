#include "TrackingAllocator.hpp"

#include <algorithm>
#include <cstring>

namespace My {

TrackingAllocator::TrackingAllocator(IMemoryManager* pMmgr, IAllocator& inner,
                                     std::string name)
    : IAllocator(pMmgr), m_Inner(inner), m_Name(std::move(name)) {}

TrackingAllocator::~TrackingAllocator() {
    if (m_LeakHandler && !m_Live.empty()) m_LeakHandler(LeakReport());
}

void* TrackingAllocator::AllocateTagged(size_t size, const char* tag) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (size == 0 || size > SIZE_MAX - kGuardSize) {
        ++m_Stats.failedAllocations;
        return nullptr;
    }
    auto* p = static_cast<uint8_t*>(m_Inner.Allocate(size + kGuardSize));
    if (!p) {
        ++m_Stats.failedAllocations;
        return nullptr;
    }
    std::memset(p + size, kGuardByte, kGuardSize);

    m_Live[p] = Record{size, tag, ++m_nSequence};
    ++m_Stats.allocations;
    m_Stats.totalBytes += size;
    m_Stats.liveBytes += size;
    m_Stats.liveCount = m_Live.size();
    m_Stats.peakLiveBytes = (std::max)(m_Stats.peakLiveBytes, m_Stats.liveBytes);
    m_Stats.peakLiveCount = (std::max)(m_Stats.peakLiveCount, m_Stats.liveCount);
    return p;
}

bool TrackingAllocator::GuardIntact(const void* p, size_t size) const {
    const auto* g = static_cast<const uint8_t*>(p) + size;
    for (size_t i = 0; i < kGuardSize; i++) {
        if (g[i] != kGuardByte) return false;
    }
    return true;
}

void TrackingAllocator::Free(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(m_Mutex);
    auto it = m_Live.find(p);
    if (it == m_Live.end()) {
        ++m_Stats.invalidFrees;
        return;
    }
    if (!GuardIntact(p, it->second.size)) ++m_Stats.guardViolations;

    m_Stats.liveBytes -= it->second.size;
    m_Live.erase(it);
    m_Stats.liveCount = m_Live.size();
    ++m_Stats.frees;
    m_Inner.Free(p);
}

void TrackingAllocator::FreeAll() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Live.clear();
    m_Stats.liveCount = 0;
    m_Stats.liveBytes = 0;
    m_Inner.FreeAll();
}

size_t TrackingAllocator::CheckGuards() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    size_t bad = 0;
    for (const auto& [p, r] : m_Live) {
        if (!GuardIntact(p, r.size)) ++bad;
    }
    return bad;
}

TrackingAllocator::Stats TrackingAllocator::GetStats() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Stats;
}

std::vector<TrackingAllocator::Leak> TrackingAllocator::Leaks() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    std::vector<Leak> leaks;
    leaks.reserve(m_Live.size());
    for (const auto& [p, r] : m_Live) {
        leaks.push_back({p, r.size, r.tag, r.sequence});
    }
    std::sort(leaks.begin(), leaks.end(),
              [](const Leak& a, const Leak& b) { return a.sequence < b.sequence; });
    return leaks;
}

std::string TrackingAllocator::LeakReport() const {
    const auto leaks = Leaks();
    size_t bytes = 0;
    for (const auto& l : leaks) bytes += l.size;

    std::string out = m_Name + ": " + std::to_string(leaks.size()) +
                      " leaked block(s), " + std::to_string(bytes) + " byte(s)\n";
    for (const auto& l : leaks) {
        out += "  #" + std::to_string(l.sequence) + " " +
               std::to_string(l.size) + " bytes";
        if (l.tag) out += std::string(" [") + l.tag + "]";
        out += "\n";
    }
    return out;
}

void TrackingAllocator::SetLeakHandler(LeakHandler handler) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_LeakHandler = std::move(handler);
}
}  // namespace My

#include "GpuSubAllocator.hpp"

#include <algorithm>
#include <limits>

namespace My {

namespace {
constexpr uint64_t kMax = (std::numeric_limits<uint64_t>::max)();

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& out) {
    const uint64_t mask = alignment - 1;
    if (value > kMax - mask) return false;
    out = (value + mask) & ~mask;
    return true;
}
}  // namespace

GpuSubAllocator::GpuSubAllocator(uint64_t capacity, uint64_t granularity)
    : m_nCapacity(capacity), m_nGranularity(granularity) {
    if (capacity == 0 || granularity == 0) return;
    m_bConfigured = true;
    AddFree(0, capacity);
}

void GpuSubAllocator::AddFree(uint64_t offset, uint64_t size) {
    auto next = m_Free.lower_bound(offset);
    if (next != m_Free.end() && offset + size == next->first) {
        size += next->second;
        m_BySize.erase({next->second, next->first});
        next = m_Free.erase(next);
    }
    if (next != m_Free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == offset) {
            offset = prev->first;
            size += prev->second;
            m_BySize.erase({prev->second, prev->first});
            m_Free.erase(prev);
        }
    }
    m_Free.emplace(offset, size);
    m_BySize.emplace(size, offset);
}

bool GpuSubAllocator::Allocate(uint64_t size, uint64_t alignment,
                               GpuAllocation& out) {
    out = {};
    uint64_t reserved = 0;
    if (!m_bConfigured || size == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0 ||
        !AlignUp(size, m_nGranularity, reserved) || reserved > m_nCapacity) {
        ++m_nFailures;
        return false;
    }

    for (auto it = m_BySize.lower_bound({reserved, 0}); it != m_BySize.end();
         ++it) {
        const uint64_t blockSize = it->first;
        const uint64_t blockOffset = it->second;
        uint64_t aligned = 0;
        if (!AlignUp(blockOffset, alignment, aligned)) continue;
        const uint64_t gap = aligned - blockOffset;
        if (gap > blockSize || blockSize - gap < reserved) continue;

        m_Free.erase(blockOffset);
        m_BySize.erase(it);
        if (gap > 0) AddFree(blockOffset, gap);
        const uint64_t tail = blockSize - gap - reserved;
        if (tail > 0) AddFree(aligned + reserved, tail);

        const uint64_t id = m_nNextId++;
        m_Live.emplace(aligned, Live{reserved, size, id, false});
        m_nUsed += reserved;
        m_nPeakUsed = (std::max)(m_nPeakUsed, m_nUsed);
        out = {aligned, size, id};
        return true;
    }
    ++m_nFailures;
    return false;
}

GpuSubAllocator::Live* GpuSubAllocator::Find(const GpuAllocation& a) {
    if (!a.IsValid()) return nullptr;
    auto it = m_Live.find(a.offset);
    if (it == m_Live.end() || it->second.id != a.id ||
        it->second.requested != a.size) {
        return nullptr;
    }
    return &it->second;
}

void GpuSubAllocator::Release(uint64_t offset) {
    auto it = m_Live.find(offset);
    const uint64_t reserved = it->second.reserved;
    if (it->second.pending) m_nPendingBytes -= reserved;
    m_Live.erase(it);
    m_nUsed -= reserved;
    AddFree(offset, reserved);
}

bool GpuSubAllocator::Free(const GpuAllocation& a) {
    Live* live = Find(a);
    if (!live || live->pending) {
        ++m_nInvalidFrees;
        return false;
    }
    Release(a.offset);
    return true;
}

bool GpuSubAllocator::FreeDeferred(const GpuAllocation& a, uint64_t fence) {
    Live* live = Find(a);
    if (!live || live->pending) {
        ++m_nInvalidFrees;
        return false;
    }
    if (m_bHasCompleted && fence <= m_nCompleted) {
        Release(a.offset);
        return true;
    }
    live->pending = true;
    m_nPendingBytes += live->reserved;
    m_Pending.emplace(fence, a.offset);
    return true;
}

size_t GpuSubAllocator::Retire(uint64_t completed) {
    if (!m_bHasCompleted || completed > m_nCompleted) {
        m_nCompleted = completed;
        m_bHasCompleted = true;
    }
    size_t released = 0;
    while (!m_Pending.empty() && m_Pending.begin()->first <= completed) {
        Release(m_Pending.begin()->second);
        m_Pending.erase(m_Pending.begin());
        ++released;
    }
    return released;
}

void GpuSubAllocator::Reset() {
    m_Free.clear();
    m_BySize.clear();
    m_Live.clear();
    m_Pending.clear();
    m_nUsed = 0;
    m_nPendingBytes = 0;
    m_bHasCompleted = false;
    m_nCompleted = 0;
    if (m_bConfigured) AddFree(0, m_nCapacity);
}

uint64_t GpuSubAllocator::LargestFreeRange() const {
    return m_BySize.empty() ? 0 : m_BySize.rbegin()->first;
}

bool GpuSubAllocator::Check() const {
    if (!m_bConfigured) return m_Free.empty() && m_Live.empty();
    if (m_Free.size() != m_BySize.size()) return false;

    // free ranges and live ranges together must tile [0, capacity)
    std::map<uint64_t, uint64_t> all;
    uint64_t freeBytes = 0;
    uint64_t prevEnd = 0;
    bool havePrev = false;
    for (const auto& [off, size] : m_Free) {
        if (size == 0 || !m_BySize.count({size, off})) return false;
        if (havePrev && prevEnd >= off) return false;  // overlap or uncoalesced
        prevEnd = off + size;
        havePrev = true;
        freeBytes += size;
        all.emplace(off, size);
    }
    uint64_t usedBytes = 0;
    uint64_t pendingBytes = 0;
    for (const auto& [off, live] : m_Live) {
        usedBytes += live.reserved;
        if (live.pending) pendingBytes += live.reserved;
        if (!all.emplace(off, live.reserved).second) return false;
    }
    if (usedBytes != m_nUsed || pendingBytes != m_nPendingBytes ||
        freeBytes + usedBytes != m_nCapacity) {
        return false;
    }
    uint64_t cursor = 0;
    for (const auto& [off, size] : all) {
        if (off != cursor) return false;
        cursor = off + size;
    }
    if (cursor != m_nCapacity) return false;

    size_t pending = 0;
    for (const auto& entry : m_Pending) {
        auto it = m_Live.find(entry.second);
        if (it == m_Live.end() || !it->second.pending) return false;
        ++pending;
    }
    return pending == m_Pending.size();
}
}  // namespace My

#include "RingAllocator.hpp"

#include <cstring>

#include "AllocatorUtils.hpp"

using namespace My;

namespace {
constexpr size_t kArenaAlignment = 64;

// Offset within the buffer of the first aligned address at or after pos.
bool AlignedOffset(const uint8_t* base, size_t pos, size_t align,
                   size_t& out) {
    uintptr_t aligned = 0;
    if (!AlignAddress(reinterpret_cast<uintptr_t>(base) + pos, align,
                      aligned)) {
        return false;
    }
    out = aligned - reinterpret_cast<uintptr_t>(base);
    return true;
}
}  // namespace

RingAllocator::RingAllocator(IMemoryManager* pMmgr, size_t capacity)
    : IAllocator(pMmgr) {
    Initialize(capacity);
}

RingAllocator::~RingAllocator() { Release(); }

bool RingAllocator::Initialize(size_t capacity) {
    Release();
    if (capacity == 0) return false;

    m_pBase = static_cast<uint8_t*>(
        m_pMemoryManager->AllocateAlignedPage(capacity, kArenaAlignment));
    if (!m_pBase) return false;

    m_szCapacity = capacity;
    return true;
}

void RingAllocator::Release() {
    if (m_pBase) m_pMemoryManager->FreePage(m_pBase);
    m_pBase = nullptr;
    m_szCapacity = 0;
    m_nHead = 0;
    m_nTail = 0;
    m_szPeak = 0;
}

void* RingAllocator::Allocate(size_t size, size_t alignment) {
    size_t off = 0;
    if (!m_pBase || size == 0 || size > m_szCapacity ||
        !IsPowerOfTwo(alignment)) {
        ++m_nFailures;
        return nullptr;
    }

    const size_t pos = static_cast<size_t>(m_nHead % m_szCapacity);
    uint64_t start = 0;
    bool ok = AlignedOffset(m_pBase, pos, alignment, off);
    if (ok && off <= m_szCapacity && size <= m_szCapacity - off) {
        start = m_nHead + (off - pos);
    } else {
        // wrap: skip the rest of the buffer and start from the beginning
        ok = AlignedOffset(m_pBase, 0, alignment, off) && off <= m_szCapacity &&
             size <= m_szCapacity - off;
        if (ok) start = m_nHead + (m_szCapacity - pos) + off;
    }

    if (!ok || start + size - m_nTail > m_szCapacity) {
        ++m_nFailures;
        return nullptr;
    }

    m_nHead = start + size;
    if (Used() > m_szPeak) m_szPeak = Used();

    void* p = m_pBase + off;
#if defined(_DEBUG)
    std::memset(p, PATTERN_ALLOC, size);
#endif
    return p;
}

bool RingAllocator::Retire(uint64_t position) {
    if (position < m_nTail || position > m_nHead) {
        ++m_nInvalidRetires;
        return false;
    }
    m_nTail = position;
    return true;
}

void RingAllocator::FreeAll() { m_nTail = m_nHead; }

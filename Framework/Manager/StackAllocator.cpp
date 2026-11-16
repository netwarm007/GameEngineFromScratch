#include "StackAllocator.hpp"

#include <cstring>

#include "AllocatorUtils.hpp"

using namespace My;

namespace {
constexpr size_t kArenaAlignment = 64;
}  // namespace

StackAllocator::StackAllocator(IMemoryManager* pMmgr, size_t capacity)
    : IAllocator(pMmgr) {
    Initialize(capacity);
}

StackAllocator::~StackAllocator() { Release(); }

bool StackAllocator::Initialize(size_t capacity) {
    Release();
    if (capacity == 0) return false;

    m_pBase = static_cast<uint8_t*>(
        m_pMemoryManager->AllocateAlignedPage(capacity, kArenaAlignment));
    if (!m_pBase) return false;

    m_szCapacity = capacity;
    return true;
}

void StackAllocator::Release() {
    if (m_pBase) m_pMemoryManager->FreePage(m_pBase);
    m_pBase = nullptr;
    m_szCapacity = 0;
    m_szTop = 0;
    m_szLast = SIZE_MAX;
    m_szPeak = 0;
}

void* StackAllocator::Allocate(size_t size, size_t alignment) {
    uintptr_t user = 0;
    if (!m_pBase || size == 0 || !IsPowerOfTwo(alignment)) {
        ++m_nFailures;
        return nullptr;
    }
    // the header sits right below the user pointer, so keep it aligned too
    if (alignment < alignof(Header)) alignment = alignof(Header);

    const auto base = reinterpret_cast<uintptr_t>(m_pBase);
    if (!AlignAddress(base + m_szTop + sizeof(Header), alignment, user)) {
        ++m_nFailures;
        return nullptr;
    }

    const size_t offset = user - base;
    if (offset > m_szCapacity || size > m_szCapacity - offset) {
        ++m_nFailures;
        return nullptr;
    }

    Header header{m_szTop, m_szLast};
    std::memcpy(m_pBase + offset - sizeof(Header), &header, sizeof(Header));

    m_szTop = offset + size;
    m_szLast = offset;
    if (m_szTop > m_szPeak) m_szPeak = m_szTop;

#if defined(_DEBUG)
    std::memset(m_pBase + offset, PATTERN_ALLOC, size);
#endif
    return m_pBase + offset;
}

void StackAllocator::Poison(size_t from, size_t to) {
#if defined(_DEBUG)
    if (m_pBase && from < to) std::memset(m_pBase + from, PATTERN_FREE, to - from);
#else
    (void)from;
    (void)to;
#endif
}

void StackAllocator::Free(void* p) {
    if (!p) return;

    const auto addr = reinterpret_cast<uintptr_t>(p);
    const auto base = reinterpret_cast<uintptr_t>(m_pBase);
    // only the most recent allocation can be popped
    if (!m_pBase || m_szLast == SIZE_MAX || addr != base + m_szLast) {
        ++m_nInvalidFrees;
        return;
    }

    Header header;
    std::memcpy(&header, m_pBase + m_szLast - sizeof(Header), sizeof(Header));

    Poison(header.prevTop, m_szTop);
    m_szTop = header.prevTop;
    m_szLast = header.prevLast;
}

bool StackAllocator::FreeToMarker(const Marker& marker) {
    if (marker.top > m_szTop) {
        ++m_nInvalidFrees;
        return false;
    }
    Poison(marker.top, m_szTop);
    m_szTop = marker.top;
    m_szLast = marker.last;
    return true;
}

void StackAllocator::FreeAll() {
    Poison(0, m_szTop);
    m_szTop = 0;
    m_szLast = SIZE_MAX;
}

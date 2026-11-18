#include "LinearAllocator.hpp"

#include <cstring>

#include "AllocatorUtils.hpp"

using namespace My;

namespace {
// large enough for typical SIMD and cache-line alignment of the arena start
constexpr size_t kArenaAlignment = 64;
}  // namespace

LinearAllocator::LinearAllocator(IMemoryManager* pMmgr, size_t capacity)
    : IAllocator(pMmgr) {
    Initialize(capacity);
}

LinearAllocator::~LinearAllocator() { Release(); }

bool LinearAllocator::Initialize(size_t capacity) {
    Release();
    if (capacity == 0) return false;

    m_pBase = static_cast<uint8_t*>(
        m_pMemoryManager->AllocateAlignedPage(capacity, kArenaAlignment));
    if (!m_pBase) return false;

    m_szCapacity = capacity;
    return true;
}

void LinearAllocator::Release() {
    if (m_pBase) m_pMemoryManager->FreePage(m_pBase);
    m_pBase = nullptr;
    m_szCapacity = 0;
    m_szUsed = 0;
    m_szPeak = 0;
    m_nAllocations = 0;
}

void* LinearAllocator::Allocate(size_t size, size_t alignment) {
    uintptr_t aligned = 0;
    if (!m_pBase || size == 0 || !IsPowerOfTwo(alignment) ||
        !AlignAddress(reinterpret_cast<uintptr_t>(m_pBase) + m_szUsed,
                      alignment, aligned)) {
        ++m_nFailures;
        return nullptr;
    }

    const size_t offset = aligned - reinterpret_cast<uintptr_t>(m_pBase);
    if (offset > m_szCapacity || size > m_szCapacity - offset) {
        ++m_nFailures;
        return nullptr;
    }

    m_szUsed = offset + size;
    if (m_szUsed > m_szPeak) m_szPeak = m_szUsed;
    ++m_nAllocations;

    void* p = m_pBase + offset;
#if defined(_DEBUG)
    std::memset(p, PATTERN_ALLOC, size);
#endif
    return p;
}

void LinearAllocator::FreeAll() {
#if defined(_DEBUG)
    if (m_pBase) std::memset(m_pBase, PATTERN_FREE, m_szUsed);
#endif
    m_szUsed = 0;
    m_nAllocations = 0;
}

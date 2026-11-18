#include "BuddyAllocator.hpp"

#include <bit>
#include <cstring>

#include "AllocatorUtils.hpp"

namespace My {

namespace {
constexpr uint8_t kFreeBit = 0x80;
}

BuddyAllocator::BuddyAllocator(IMemoryManager* pMmgr, size_t capacity,
                               size_t minBlock)
    : IAllocator(pMmgr) {
    Initialize(capacity, minBlock);
}

BuddyAllocator::~BuddyAllocator() { Release(); }

void BuddyAllocator::Release() {
    if (m_pBase) m_pMemoryManager->FreePage(m_pBase);
    if (m_pState) m_pMemoryManager->FreePage(m_pState);
    m_pBase = nullptr;
    m_pState = nullptr;
    m_szCapacity = m_szMinBlock = m_nMinShift = m_nOrders = 0;
    m_szUsed = m_szPeak = m_nAllocations = 0;
    std::memset(m_FreeLists, 0, sizeof(m_FreeLists));
}

bool BuddyAllocator::Initialize(size_t capacity, size_t minBlock) {
    Release();
    if (!IsPowerOfTwo(capacity) || !IsPowerOfTwo(minBlock) || minBlock < 16 ||
        minBlock > capacity) {
        return false;
    }
    const size_t orders = std::countr_zero(capacity / minBlock) + 1;
    if (orders > kMaxOrders) return false;

    const size_t units = capacity / minBlock;
    m_pBase = static_cast<uint8_t*>(
        m_pMemoryManager->AllocateAlignedPage(capacity, kBaseAlignment));
    if (!m_pBase) return false;
    m_pState = static_cast<uint8_t*>(m_pMemoryManager->AllocatePage(units));
    if (!m_pState) {
        m_pMemoryManager->FreePage(m_pBase);
        m_pBase = nullptr;
        return false;
    }

    m_szCapacity = capacity;
    m_szMinBlock = minBlock;
    m_nMinShift = std::countr_zero(minBlock);
    m_nOrders = orders;
    m_nFailures = m_nInvalidFrees = 0;
    FreeAll();
    return true;
}

void BuddyAllocator::PushFree(size_t unit, size_t order) {
    auto* node = reinterpret_cast<FreeNode*>(AddressOf(unit));
    node->prev = nullptr;
    node->next = m_FreeLists[order];
    if (node->next) node->next->prev = node;
    m_FreeLists[order] = node;
    m_pState[unit] = kFreeBit | static_cast<uint8_t>(order + 1);
}

void BuddyAllocator::RemoveFree(size_t unit, size_t order) {
    auto* node = reinterpret_cast<FreeNode*>(AddressOf(unit));
    if (node->next) node->next->prev = node->prev;
    if (node->prev) {
        node->prev->next = node->next;
    } else {
        m_FreeLists[order] = node->next;
    }
    m_pState[unit] = 0;
}

void BuddyAllocator::FreeAll() {
    if (!m_pBase) return;
    std::memset(m_pState, 0, m_szCapacity >> m_nMinShift);
    std::memset(m_FreeLists, 0, sizeof(m_FreeLists));
    m_szUsed = 0;
    m_nAllocations = 0;
    PushFree(0, m_nOrders - 1);
}

size_t BuddyAllocator::BlockSizeFor(size_t size) const {
    if (!m_pBase || size == 0 || size > m_szCapacity) return 0;
    const size_t units = (size + m_szMinBlock - 1) >> m_nMinShift;
    return m_szMinBlock << (units == 1 ? 0 : std::bit_width(units - 1));
}

void* BuddyAllocator::Allocate(size_t size, size_t alignment) {
    if (!m_pBase || size == 0 || !IsPowerOfTwo(alignment) ||
        alignment > kBaseAlignment || size > m_szCapacity) {
        ++m_nFailures;
        return nullptr;
    }
    const size_t blockSize = BlockSizeFor(size > alignment ? size : alignment);
    if (blockSize == 0) {
        ++m_nFailures;
        return nullptr;
    }
    const size_t order = std::countr_zero(blockSize / m_szMinBlock);

    size_t o = order;
    while (o < m_nOrders && !m_FreeLists[o]) ++o;
    if (o >= m_nOrders) {
        ++m_nFailures;
        return nullptr;
    }

    const size_t unit = UnitOf(m_FreeLists[o]);
    RemoveFree(unit, o);
    while (o > order) {
        --o;
        PushFree(unit + (size_t(1) << o), o);
    }
    m_pState[unit] = static_cast<uint8_t>(order + 1);

    m_szUsed += blockSize;
    if (m_szUsed > m_szPeak) m_szPeak = m_szUsed;
    ++m_nAllocations;

    void* p = AddressOf(unit);
#if defined(_DEBUG)
    std::memset(p, PATTERN_ALLOC, blockSize);
#endif
    return p;
}

void BuddyAllocator::Free(void* p) {
    if (!p) return;
    const auto* raw = static_cast<const uint8_t*>(p);
    if (!m_pBase || raw < m_pBase || raw >= m_pBase + m_szCapacity ||
        static_cast<size_t>(raw - m_pBase) % m_szMinBlock != 0) {
        ++m_nInvalidFrees;
        return;
    }
    size_t unit = UnitOf(p);
    const uint8_t state = m_pState[unit];
    if (state == 0 || (state & kFreeBit)) {
        ++m_nInvalidFrees;
        return;
    }

    size_t order = state - 1;
    m_szUsed -= m_szMinBlock << order;
    --m_nAllocations;
#if defined(_DEBUG)
    std::memset(p, PATTERN_FREE, m_szMinBlock << order);
#endif

    m_pState[unit] = 0;
    while (order + 1 < m_nOrders) {
        const size_t buddy = unit ^ (size_t(1) << order);
        if (m_pState[buddy] != (kFreeBit | static_cast<uint8_t>(order + 1))) break;
        RemoveFree(buddy, order);
        if (buddy < unit) unit = buddy;
        m_pState[unit] = 0;
        ++order;
    }
    PushFree(unit, order);
}

size_t BuddyAllocator::LargestFreeBlock() const {
    for (size_t o = m_nOrders; o-- > 0;) {
        if (m_FreeLists[o]) return m_szMinBlock << o;
    }
    return 0;
}

bool BuddyAllocator::Check() const {
    if (!m_pBase) return true;
    const size_t units = m_szCapacity >> m_nMinShift;
    size_t used = 0, live = 0, freeBlocks = 0;

    for (size_t u = 0; u < units;) {
        const uint8_t s = m_pState[u];
        if (s == 0) return false;
        const size_t order = (s & ~kFreeBit) - 1;
        const size_t span = size_t(1) << order;
        if (order >= m_nOrders || (u & (span - 1)) != 0 || u + span > units) {
            return false;
        }
        for (size_t i = 1; i < span; i++) {
            if (m_pState[u + i] != 0) return false;
        }
        if (s & kFreeBit) {
            ++freeBlocks;
            if (order + 1 < m_nOrders && m_pState[u ^ span] == s) return false;
        } else {
            used += span << m_nMinShift;
            ++live;
        }
        u += span;
    }
    if (used != m_szUsed || live != m_nAllocations) return false;

    size_t listed = 0;
    for (size_t o = 0; o < m_nOrders; o++) {
        const FreeNode* prev = nullptr;
        for (const FreeNode* n = m_FreeLists[o]; n; n = n->next) {
            const size_t unit = UnitOf(n);
            if (m_pState[unit] != (kFreeBit | (o + 1)) || n->prev != prev) {
                return false;
            }
            prev = n;
            if (++listed > freeBlocks) return false;
        }
    }
    return listed == freeBlocks;
}
}  // namespace My

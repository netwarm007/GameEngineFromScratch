#include "RoutingAllocator.hpp"

#include "AllocatorUtils.hpp"

namespace My {

RoutingAllocator::RoutingAllocator(IMemoryManager* pMmgr, size_t largeCapacity,
                                   size_t slabSize, uint32_t maxCachedPerClass)
    : IAllocator(pMmgr),
      m_Small(pMmgr, slabSize, maxCachedPerClass),
      m_Large(pMmgr, largeCapacity) {}

void* RoutingAllocator::Allocate(size_t size, size_t alignment) {
    if (size == 0 || !IsPowerOfTwo(alignment) || !IsConfigured()) {
        return nullptr;
    }

    if (size <= kSmallLimit && alignment <= kAlignment) {
        if (void* p = m_Small.Allocate(size)) {
            ++m_nSmall;
            return p;
        }
    }

    void* p;
    {
        std::lock_guard<std::mutex> lock(m_LargeMutex);
        p = m_Large.Allocate(size, alignment);
    }
    if (p == nullptr) {
        ++m_nFailures;
        return nullptr;
    }
    ++m_nLarge;
    return p;
}

void RoutingAllocator::Free(void* p) {
    if (p == nullptr) return;
    if (m_Large.Owns(p)) {
        std::lock_guard<std::mutex> lock(m_LargeMutex);
        m_Large.Free(p);
    } else {
        m_Small.Free(p);
    }
}

void RoutingAllocator::FreeAll() {
    m_Small.FreeAll();
    std::lock_guard<std::mutex> lock(m_LargeMutex);
    m_Large.FreeAll();
}

size_t RoutingAllocator::LargeAllocationCount() const {
    std::lock_guard<std::mutex> lock(m_LargeMutex);
    return m_Large.AllocationCount();
}

size_t RoutingAllocator::LargeUsed() const {
    std::lock_guard<std::mutex> lock(m_LargeMutex);
    return m_Large.Used();
}

size_t RoutingAllocator::LargeCapacity() const { return m_Large.Capacity(); }

size_t RoutingAllocator::InvalidFreeCount() const {
    std::lock_guard<std::mutex> lock(m_LargeMutex);
    return m_Small.InvalidFreeCount() + m_Large.InvalidFreeCount();
}

}  // namespace My

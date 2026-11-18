#include "MemoryManager.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace My;
using namespace std;

namespace My {
std::ostream& operator<<(std::ostream& out, MemoryType type) {
    return out << (type == MemoryType::GPU ? "GPU" : "CPU");
}
}  // namespace My

int MemoryManager::Initialize() { return 0; }

void MemoryManager::Finalize() {
    lock_guard<mutex> lock(m_Mutex);
    if (!m_mapMemoryAllocationInfo.empty()) {
        cerr << "MemoryManager: " << m_mapMemoryAllocationInfo.size()
             << " page(s) leaked, " << m_nAllocatedBytes << " bytes\n";
        for (const auto& info : m_mapMemoryAllocationInfo) {
            cerr << "  " << info.first << '\t' << info.second.PageMemoryType
                 << '\t' << info.second.PageSize << '\n';
        }
    }
    assert(m_mapMemoryAllocationInfo.empty());
}

void MemoryManager::Tick() {}

void* MemoryManager::Register(void* p, size_t size, bool aligned) {
    if (p) {
        lock_guard<mutex> lock(m_Mutex);
        m_mapMemoryAllocationInfo.insert({p, {size, MemoryType::CPU, aligned}});
        m_nAllocatedBytes += size;
        if (m_nAllocatedBytes > m_nPeakAllocatedBytes) {
            m_nPeakAllocatedBytes = m_nAllocatedBytes;
        }
    }
    return p;
}

void* MemoryManager::AllocatePage(size_t size) {
    if (size == 0) return nullptr;
    return Register(malloc(size), size, false);
}

void* MemoryManager::AllocateAlignedPage(size_t size, size_t alignment) {
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return nullptr;
    }
    if (alignment <= alignof(max_align_t)) {
        return AllocatePage(size);
    }

#if defined(_WIN32)
    void* p = _aligned_malloc(size, alignment);
#else
    // aligned_alloc requires the size to be a multiple of the alignment
    size_t padded = (size + alignment - 1) & ~(alignment - 1);
    if (padded < size) return nullptr;
    void* p = aligned_alloc(alignment, padded);
#endif
    return Register(p, size, true);
}

void MemoryManager::FreePage(void* p) {
    bool aligned = false;
    {
        lock_guard<mutex> lock(m_Mutex);
        auto it = m_mapMemoryAllocationInfo.find(p);
        if (it == m_mapMemoryAllocationInfo.end()) {
            ++m_nInvalidFrees;
            return;
        }
        aligned = it->second.Aligned;
        m_nAllocatedBytes -= it->second.PageSize;
        m_mapMemoryAllocationInfo.erase(it);
    }

#if defined(_WIN32)
    if (aligned) {
        _aligned_free(p);
        return;
    }
#else
    (void)aligned;
#endif
    free(p);
}

size_t MemoryManager::PageCount() const {
    lock_guard<mutex> lock(m_Mutex);
    return m_mapMemoryAllocationInfo.size();
}

size_t MemoryManager::AllocatedBytes() const {
    lock_guard<mutex> lock(m_Mutex);
    return m_nAllocatedBytes;
}

size_t MemoryManager::PeakAllocatedBytes() const {
    lock_guard<mutex> lock(m_Mutex);
    return m_nPeakAllocatedBytes;
}

size_t MemoryManager::InvalidFreeCount() const {
    lock_guard<mutex> lock(m_Mutex);
    return m_nInvalidFrees;
}

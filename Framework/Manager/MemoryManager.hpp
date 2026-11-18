#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <ostream>
#include <unordered_map>

#include "IMemoryManager.hpp"

namespace My {
enum class MemoryType : int32_t { CPU, GPU };

std::ostream& operator<<(std::ostream& out, MemoryType type);

// Page-level memory source for allocators. Thread-safe.
class MemoryManager : _implements_ IMemoryManager {
   public:
    ~MemoryManager() override = default;
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void* AllocatePage(size_t size) override;
    void* AllocateAlignedPage(size_t size, size_t alignment) override;
    void FreePage(void* p) override;

    // statistics
    size_t PageCount() const;
    size_t AllocatedBytes() const;
    size_t PeakAllocatedBytes() const;
    // frees of null, unknown, or already released pointers
    size_t InvalidFreeCount() const;

   protected:
    struct MemoryAllocationInfo {
        size_t PageSize;
        MemoryType PageMemoryType;
        bool Aligned;
    };

    void* Register(void* p, size_t size, bool aligned);

    mutable std::mutex m_Mutex;
    std::unordered_map<void*, MemoryAllocationInfo> m_mapMemoryAllocationInfo;
    size_t m_nAllocatedBytes = 0;
    size_t m_nPeakAllocatedBytes = 0;
    size_t m_nInvalidFrees = 0;
};
}  // namespace My

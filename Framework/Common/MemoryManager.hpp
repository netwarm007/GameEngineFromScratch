#pragma once
#include <map>
#include <new>
#include <ostream>

#include "IMemoryManager.hpp"
#include "portable.hpp"

namespace My {
ENUM(MemoryType){CPU = "CPU"_i32, GPU = "GPU"_i32};

std::ostream& operator<<(std::ostream& out, MemoryType type);

class MemoryManager : _implements_ IMemoryManager {
   public:
    ~MemoryManager() override = default;
    int Initialize() override;
    void Finalize() override;
    void Tick() override;

    void* AllocatePage(size_t size) override;
    void FreePage(void* p) override;

   protected:
    struct MemoryAllocationInfo {
        size_t PageSize;
        MemoryType PageMemoryType;
    };

    std::map<void*, MemoryAllocationInfo> m_mapMemoryAllocationInfo;
};
}  // namespace My

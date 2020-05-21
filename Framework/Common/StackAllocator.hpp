#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>

#include "IAllocator.hpp"

namespace My {
class StackAllocator : _implements_ IAllocator {
   public:
    StackAllocator();
    StackAllocator(size_t page_size, size_t alignment);
    ~StackAllocator() override;
    // disable copy & assignment
    StackAllocator(const StackAllocator& clone) = delete;
    StackAllocator& operator=(const StackAllocator& rhs) = delete;

    // alloc and free blocks
    void* Allocate(size_t size) override;
    void Free(void* p) override;
    void FreeAll() override;

   protected:
    std::list<uint8_t*> m_pPages;
    std::list<std::shared_ptr<void>> m_pAllocatedPointers;
    int64_t m_StackTop;
    size_t m_MaxSize;
};
}  // namespace My

#pragma once
#include <cstddef>
#include <memory_resource>
#include <new>

#include "IAllocator.hpp"

namespace My {

// Adapts an engine allocator to std::pmr::memory_resource so standard
// containers can allocate from it. The adapter borrows the allocator, which
// must outlive the resource and everything allocated through it.
//
// Allocators providing Allocate(size, alignment) honor any power-of-two
// alignment. Others (such as BlockAllocator) only receive the size; a request
// whose result is not suitably aligned is returned to the allocator and
// reported as failure. Failure throws std::bad_alloc, as pmr requires.
// Deallocation forwards to Free(), so for LIFO or arena allocators the
// container's release pattern determines whether the free takes effect.
template <typename Allocator>
class PmrAdapter final : public std::pmr::memory_resource {
   public:
    explicit PmrAdapter(Allocator& allocator) : m_Allocator(allocator) {}

   private:
    void* do_allocate(size_t bytes, size_t alignment) override {
        if (bytes == 0) bytes = 1;

        void* p = nullptr;
        if constexpr (requires { m_Allocator.Allocate(bytes, alignment); }) {
            p = m_Allocator.Allocate(bytes, alignment);
        } else {
            p = m_Allocator.Allocate(bytes);
            if (p && reinterpret_cast<uintptr_t>(p) % alignment != 0) {
                m_Allocator.Free(p);
                p = nullptr;
            }
        }

        if (!p) throw std::bad_alloc();
        return p;
    }

    void do_deallocate(void* p, size_t, size_t) override {
        m_Allocator.Free(p);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    Allocator& m_Allocator;
};
}  // namespace My

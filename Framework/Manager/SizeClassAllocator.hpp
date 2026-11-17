#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include "IAllocator.hpp"
#include "SlabAllocator.hpp"

namespace My {

// General small-object allocator: requests up to kMaxSize bytes are served
// from one SlabAllocator per size class (16-byte steps up to 128 bytes, then
// 32- and 64-byte steps). Not thread-safe.
//
// All objects are 16-byte aligned. Free() needs no size: the owning slab is
// found by masking the pointer. Requests larger than kMaxSize (or zero) fail
// with nullptr, so callers route large blocks elsewhere. Pointers that were
// not returned by this allocator are rejected only when validation is on
// (default: debug builds); without it they must lie in a slab of this
// allocator, as with BlockAllocator.
class SizeClassAllocator : _implements_ IAllocator {
   public:
    static constexpr size_t kMaxSize = 512;
    static constexpr size_t kAlignment = 16;
    static constexpr size_t kClassCount = 16;
    static constexpr size_t kDefaultSlabSize = 16 * 1024;

    explicit SizeClassAllocator(IMemoryManager* pMmgr,
                                size_t slabSize = kDefaultSlabSize);
    ~SizeClassAllocator() override = default;
    SizeClassAllocator(const SizeClassAllocator&) = delete;
    SizeClassAllocator& operator=(const SizeClassAllocator&) = delete;

    bool IsConfigured() const { return m_bConfigured; }

    void* Allocate(size_t size) override;
    void Free(void* p) override;
    // Returns every object to the allocator; slabs are kept.
    void FreeAll() override;
    // Gives empty slabs back to the memory manager; returns the count.
    size_t Trim();

    void SetValidation(bool enable);

    // Size of the class serving this request, or 0 if it is out of range.
    static size_t ClassSize(size_t size);
    static size_t ClassIndex(size_t size);
    static size_t ClassSizeAt(size_t index);

    // Class index of a pointer returned by Allocate(), or kClassCount if the
    // slab header does not belong to this allocator. Reads only immutable
    // slab header fields, so it is safe to call while other threads allocate
    // from other objects; p must be a live object of this allocator.
    size_t ClassIndexOf(const void* p) const;

    size_t SlabCount() const;
    size_t LiveCount() const;
    size_t FailedAllocationCount() const { return m_nFailures + ClassFailures(); }
    size_t InvalidFreeCount() const;
    const SlabAllocator& Class(size_t index) const { return *m_Classes[index]; }

   private:
    size_t ClassFailures() const;

    std::unique_ptr<SlabAllocator> m_Classes[kClassCount];
    size_t m_szSlabSize;
    size_t m_nFailures = 0;
    size_t m_nInvalidFrees = 0;
    bool m_bConfigured = false;
    bool m_bValidate;
};
}  // namespace My

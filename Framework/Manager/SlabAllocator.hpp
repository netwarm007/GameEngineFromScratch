#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

// Object cache in the style of a slab allocator. Not thread-safe.
//
// Memory comes from slabs of slabSize bytes (a power of two) that are aligned
// to slabSize, so Free() finds the owning slab by masking the pointer. Slab
// bookkeeping (free-index list and in-use flags) lives in the slab header,
// never inside objects, so free objects keep their contents.
//
// With a constructor/destructor pair the cache keeps objects constructed:
// every object is constructed once when its slab is created and destroyed
// only when the slab is released (Trim() or destruction). Allocate() hands out
// an already constructed object and Free() takes it back without destroying
// it; the caller is responsible for returning it to a reusable state.
//
// Slabs with no live objects are kept for reuse until Trim() gives them back
// to the memory manager.
class SlabAllocator : _implements_ IAllocator {
   public:
    using ObjectFunction = void (*)(void*);

    explicit SlabAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    SlabAllocator(IMemoryManager* pMmgr, size_t objectSize, size_t alignment,
                  size_t slabSize, ObjectFunction construct = nullptr,
                  ObjectFunction destruct = nullptr);
    ~SlabAllocator() override;
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    // Releases every slab and applies a new configuration. Returns false (and
    // leaves the cache unusable) if alignment or slabSize is not a power of
    // two, objectSize is zero, or a slab cannot hold one object.
    bool Reset(size_t objectSize, size_t alignment, size_t slabSize,
               ObjectFunction construct = nullptr,
               ObjectFunction destruct = nullptr);

    void* Allocate();
    // nullptr when size is zero or exceeds the object size.
    void* Allocate(size_t size) override;
    // Free(nullptr) is a no-op. Pointers that are not live objects of this
    // cache (misaligned, out of range, already free) are rejected and counted.
    // With validation enabled a pointer from a foreign slab is also rejected;
    // without it the pointer must at least lie inside a slab of some cache.
    void Free(void* p) override;
    // Returns every object to the cache. Object contents are not touched.
    void FreeAll() override;

    // Gives all slabs without live objects back to the memory manager and
    // returns how many were released.
    size_t Trim();

    // Cache owning the slab that contains p, or nullptr if the slab header is
    // not valid. p must lie in memory the caller may read.
    static SlabAllocator* OwnerOf(const void* p, size_t slabSize);

    // True if p points into a slab of this cache (walks the slab lists).
    bool Owns(const void* p) const;

    // Validation walks the slab lists on every Free. Default: on in debug
    // builds only.
    void SetValidation(bool enable) { m_bValidate = enable; }

    bool IsConfigured() const { return m_bConfigured; }
    size_t ObjectSize() const { return m_szObjectSize; }
    size_t ObjectsPerSlab() const { return m_nObjectsPerSlab; }
    size_t SlabSize() const { return m_szSlabSize; }
    size_t SlabCount() const { return m_nSlabs; }
    size_t EmptySlabCount() const { return m_nEmptySlabs; }
    size_t LiveCount() const { return m_nLive; }
    size_t PeakLiveCount() const { return m_nPeakLive; }
    size_t FailedAllocationCount() const { return m_nFailures; }
    size_t InvalidFreeCount() const { return m_nInvalidFrees; }

   private:
    struct Slab;
    enum List : uint8_t { kPartial, kFull, kEmpty, kListCount };

    Slab* NewSlab();
    void DestroySlab(Slab* pSlab);
    void ResetSlab(Slab* pSlab);
    void Link(Slab* pSlab, List list);
    void Unlink(Slab* pSlab);
    uint8_t* Objects(Slab* pSlab) const;
    void ReleaseAll();

    Slab* m_pLists[kListCount] = {};

    ObjectFunction m_Construct = nullptr;
    ObjectFunction m_Destruct = nullptr;
    size_t m_szObjectSize = 0;
    size_t m_szAlignment = 0;
    size_t m_szSlabSize = 0;
    size_t m_szObjectsOffset = 0;
    size_t m_nObjectsPerSlab = 0;
    bool m_bConfigured = false;
#if defined(_DEBUG)
    bool m_bValidate = true;
#else
    bool m_bValidate = false;
#endif

    size_t m_nSlabs = 0;
    size_t m_nEmptySlabs = 0;
    size_t m_nLive = 0;
    size_t m_nPeakLive = 0;
    size_t m_nFailures = 0;
    size_t m_nInvalidFrees = 0;
};
}  // namespace My

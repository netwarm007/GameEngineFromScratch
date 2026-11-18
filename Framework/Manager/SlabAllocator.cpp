#include "SlabAllocator.hpp"

#include <cstring>
#include <new>

#include "AllocatorUtils.hpp"

namespace My {

namespace {
constexpr uint32_t kSlabMagic = 0x534C4142;  // "SLAB"
constexpr uint32_t kEndOfList = UINT32_MAX;
}  // namespace

struct SlabAllocator::Slab {
    uint32_t magic;
    uint32_t freeHead;
    uint32_t inUse;
    uint32_t total;
    SlabAllocator* owner;
    Slab* next;
    Slab* prev;
    List list;

    // Followed by uint32_t links[total] and uint8_t used[total].
    uint32_t* Links() { return reinterpret_cast<uint32_t*>(this + 1); }
    uint8_t* Used() { return reinterpret_cast<uint8_t*>(Links() + total); }
};

SlabAllocator::SlabAllocator(IMemoryManager* pMmgr, size_t objectSize,
                             size_t alignment, size_t slabSize,
                             ObjectFunction construct,
                             ObjectFunction destruct)
    : IAllocator(pMmgr) {
    Reset(objectSize, alignment, slabSize, construct, destruct);
}

SlabAllocator::~SlabAllocator() { ReleaseAll(); }

bool SlabAllocator::Reset(size_t objectSize, size_t alignment, size_t slabSize,
                          ObjectFunction construct, ObjectFunction destruct) {
    ReleaseAll();
    m_bConfigured = false;
    m_nObjectsPerSlab = 0;
    m_nPeakLive = 0;

    if (objectSize == 0 || !IsPowerOfTwo(alignment) ||
        !IsPowerOfTwo(slabSize) || alignment > slabSize ||
        objectSize > slabSize) {
        return false;
    }

    size_t stride = 0;
    if (objectSize > SIZE_MAX - (alignment - 1)) return false;
    stride = (objectSize + alignment - 1) & ~(alignment - 1);

    // largest object count whose header and objects fit in one slab
    size_t n = slabSize / stride;
    size_t offset = 0;
    for (; n >= 1; --n) {
        const size_t meta = sizeof(Slab) + n * (sizeof(uint32_t) + 1);
        offset = (meta + alignment - 1) & ~(alignment - 1);
        if (offset + n * stride <= slabSize) break;
    }
    if (n == 0 || n >= kEndOfList) return false;

    m_szObjectSize = stride;
    m_szAlignment = alignment;
    m_szSlabSize = slabSize;
    m_szObjectsOffset = offset;
    m_nObjectsPerSlab = n;
    m_Construct = construct;
    m_Destruct = destruct;
    m_bConfigured = true;
    return true;
}

uint8_t* SlabAllocator::Objects(Slab* pSlab) const {
    return reinterpret_cast<uint8_t*>(pSlab) + m_szObjectsOffset;
}

void SlabAllocator::Link(Slab* pSlab, List list) {
    pSlab->list = list;
    pSlab->prev = nullptr;
    pSlab->next = m_pLists[list];
    if (pSlab->next) pSlab->next->prev = pSlab;
    m_pLists[list] = pSlab;
    if (list == kEmpty) ++m_nEmptySlabs;
}

void SlabAllocator::Unlink(Slab* pSlab) {
    if (pSlab->prev) {
        pSlab->prev->next = pSlab->next;
    } else {
        m_pLists[pSlab->list] = pSlab->next;
    }
    if (pSlab->next) pSlab->next->prev = pSlab->prev;
    pSlab->next = pSlab->prev = nullptr;
    if (pSlab->list == kEmpty) --m_nEmptySlabs;
}

void SlabAllocator::ResetSlab(Slab* pSlab) {
    const uint32_t n = pSlab->total;
    uint32_t* links = pSlab->Links();
    for (uint32_t i = 0; i < n; i++) links[i] = i + 1 < n ? i + 1 : kEndOfList;
    std::memset(pSlab->Used(), 0, n);
    pSlab->freeHead = 0;
    pSlab->inUse = 0;
}

SlabAllocator::Slab* SlabAllocator::NewSlab() {
    auto* pSlab = static_cast<Slab*>(
        m_pMemoryManager->AllocateAlignedPage(m_szSlabSize, m_szSlabSize));
    if (!pSlab) return nullptr;

    pSlab->magic = kSlabMagic;
    pSlab->total = static_cast<uint32_t>(m_nObjectsPerSlab);
    pSlab->owner = this;
    pSlab->next = pSlab->prev = nullptr;
    ResetSlab(pSlab);

    if (m_Construct) {
        uint8_t* objects = Objects(pSlab);
        for (uint32_t i = 0; i < pSlab->total; i++) {
            m_Construct(objects + i * m_szObjectSize);
        }
    }
    ++m_nSlabs;
    Link(pSlab, kEmpty);
    return pSlab;
}

void SlabAllocator::DestroySlab(Slab* pSlab) {
    Unlink(pSlab);
    if (m_Destruct) {
        uint8_t* objects = Objects(pSlab);
        for (uint32_t i = 0; i < pSlab->total; i++) {
            m_Destruct(objects + i * m_szObjectSize);
        }
    }
    pSlab->magic = 0;
    --m_nSlabs;
    m_pMemoryManager->FreePage(pSlab);
}

void SlabAllocator::ReleaseAll() {
    for (int list = 0; list < kListCount; list++) {
        while (m_pLists[list]) DestroySlab(m_pLists[list]);
    }
    m_nLive = 0;
}

void* SlabAllocator::Allocate() { return Allocate(m_szObjectSize); }

void* SlabAllocator::Allocate(size_t size) {
    if (!m_bConfigured || size == 0 || size > m_szObjectSize) {
        ++m_nFailures;
        return nullptr;
    }

    Slab* pSlab = m_pLists[kPartial];
    if (!pSlab) pSlab = m_pLists[kEmpty];
    if (!pSlab) pSlab = NewSlab();
    if (!pSlab) {
        ++m_nFailures;
        return nullptr;
    }

    const uint32_t index = pSlab->freeHead;
    pSlab->freeHead = pSlab->Links()[index];
    pSlab->Used()[index] = 1;
    ++pSlab->inUse;

    const List target = pSlab->inUse == pSlab->total ? kFull : kPartial;
    if (pSlab->list != target) {
        Unlink(pSlab);
        Link(pSlab, target);
    }

    ++m_nLive;
    if (m_nLive > m_nPeakLive) m_nPeakLive = m_nLive;

    void* p = Objects(pSlab) + static_cast<size_t>(index) * m_szObjectSize;
#if defined(_DEBUG)
    if (!m_Construct) std::memset(p, PATTERN_ALLOC, m_szObjectSize);
#endif
    return p;
}

SlabAllocator* SlabAllocator::OwnerOf(const void* p, size_t slabSize) {
    const auto* pSlab = reinterpret_cast<const Slab*>(
        reinterpret_cast<uintptr_t>(p) & ~(slabSize - 1));
    return pSlab->magic == kSlabMagic ? pSlab->owner : nullptr;
}

bool SlabAllocator::Owns(const void* p) const {
    const auto base = reinterpret_cast<uintptr_t>(p) & ~(m_szSlabSize - 1);
    for (int list = 0; list < kListCount; list++) {
        for (Slab* s = m_pLists[list]; s; s = s->next) {
            if (reinterpret_cast<uintptr_t>(s) == base) return true;
        }
    }
    return false;
}

void SlabAllocator::Free(void* p) {
    if (!p) return;
    if (!m_bConfigured || (m_bValidate && !Owns(p))) {
        ++m_nInvalidFrees;
        return;
    }

    auto* pSlab = reinterpret_cast<Slab*>(reinterpret_cast<uintptr_t>(p) &
                                          ~(m_szSlabSize - 1));
    if (pSlab->magic != kSlabMagic || pSlab->owner != this) {
        ++m_nInvalidFrees;
        return;
    }

    const auto addr = reinterpret_cast<uintptr_t>(p);
    const auto first = reinterpret_cast<uintptr_t>(Objects(pSlab));
    if (addr < first || (addr - first) % m_szObjectSize != 0) {
        ++m_nInvalidFrees;
        return;
    }
    const size_t index = (addr - first) / m_szObjectSize;
    if (index >= pSlab->total || !pSlab->Used()[index]) {
        ++m_nInvalidFrees;
        return;
    }

#if defined(_DEBUG)
    if (!m_Construct) std::memset(p, PATTERN_FREE, m_szObjectSize);
#endif
    pSlab->Used()[index] = 0;
    pSlab->Links()[index] = pSlab->freeHead;
    pSlab->freeHead = static_cast<uint32_t>(index);
    --pSlab->inUse;
    --m_nLive;

    const List target = pSlab->inUse == 0 ? kEmpty : kPartial;
    if (pSlab->list != target) {
        Unlink(pSlab);
        Link(pSlab, target);
    }
}

void SlabAllocator::FreeAll() {
    for (int list = 0; list < kListCount; list++) {
        Slab* s = m_pLists[list];
        while (s) {
            Slab* next = s->next;
            ResetSlab(s);
            if (s->list != kEmpty) {
                Unlink(s);
                Link(s, kEmpty);
            }
            s = next;
        }
    }
    m_nLive = 0;
}

size_t SlabAllocator::Trim() {
    size_t released = 0;
    while (m_pLists[kEmpty]) {
        DestroySlab(m_pLists[kEmpty]);
        ++released;
    }
    return released;
}
}  // namespace My

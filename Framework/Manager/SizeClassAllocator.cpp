#include "SizeClassAllocator.hpp"

#include "AllocatorUtils.hpp"

namespace My {

namespace {
constexpr size_t kSizes[SizeClassAllocator::kClassCount] = {
    16,  32,  48,  64,  80,  96,  112, 128,
    160, 192, 224, 256, 320, 384, 448, 512};
}

size_t SizeClassAllocator::ClassSizeAt(size_t index) {
    return index < kClassCount ? kSizes[index] : 0;
}

size_t SizeClassAllocator::ClassIndex(size_t size) {
    if (size == 0 || size > kMaxSize) return kClassCount;
    size_t i = 0;
    while (kSizes[i] < size) ++i;
    return i;
}

size_t SizeClassAllocator::ClassSize(size_t size) {
    const size_t i = ClassIndex(size);
    return i < kClassCount ? kSizes[i] : 0;
}

SizeClassAllocator::SizeClassAllocator(IMemoryManager* pMmgr, size_t slabSize)
    : IAllocator(pMmgr), m_szSlabSize(slabSize) {
#if defined(_DEBUG)
    m_bValidate = true;
#else
    m_bValidate = false;
#endif
    for (size_t i = 0; i < kClassCount; i++) {
        m_Classes[i] = std::make_unique<SlabAllocator>(
            pMmgr, kSizes[i], kAlignment, slabSize);
        if (!m_Classes[i]->IsConfigured()) return;
    }
    SetValidation(m_bValidate);
    m_bConfigured = true;
}

void SizeClassAllocator::SetValidation(bool enable) {
    m_bValidate = enable;
    for (auto& c : m_Classes) {
        if (c) c->SetValidation(enable);
    }
}

void* SizeClassAllocator::Allocate(size_t size) {
    const size_t i = ClassIndex(size);
    if (!m_bConfigured || i >= kClassCount) {
        ++m_nFailures;
        return nullptr;
    }
    return m_Classes[i]->Allocate();
}

void SizeClassAllocator::Free(void* p) {
    if (!p) return;
    if (!m_bConfigured) {
        ++m_nInvalidFrees;
        return;
    }

    if (m_bValidate) {
        for (auto& c : m_Classes) {
            if (c->Owns(p)) {
                c->Free(p);
                return;
            }
        }
        ++m_nInvalidFrees;
        return;
    }

    SlabAllocator* owner = SlabAllocator::OwnerOf(p, m_szSlabSize);
    if (owner) {
        owner->Free(p);  // rejects owners that are not ours via its own checks
    } else {
        ++m_nInvalidFrees;
    }
}

void SizeClassAllocator::FreeAll() {
    for (auto& c : m_Classes) {
        if (c) c->FreeAll();
    }
}

size_t SizeClassAllocator::Trim() {
    size_t released = 0;
    for (auto& c : m_Classes) {
        if (c) released += c->Trim();
    }
    return released;
}

size_t SizeClassAllocator::SlabCount() const {
    size_t n = 0;
    for (auto& c : m_Classes) {
        if (c) n += c->SlabCount();
    }
    return n;
}

size_t SizeClassAllocator::LiveCount() const {
    size_t n = 0;
    for (auto& c : m_Classes) {
        if (c) n += c->LiveCount();
    }
    return n;
}

size_t SizeClassAllocator::ClassFailures() const {
    size_t n = 0;
    for (auto& c : m_Classes) {
        if (c) n += c->FailedAllocationCount();
    }
    return n;
}

size_t SizeClassAllocator::InvalidFreeCount() const {
    size_t n = m_nInvalidFrees;
    for (auto& c : m_Classes) {
        if (c) n += c->InvalidFreeCount();
    }
    return n;
}
}  // namespace My

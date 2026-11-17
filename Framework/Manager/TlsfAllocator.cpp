#include "TlsfAllocator.hpp"

#include <bit>
#include <cstring>

#include "AllocatorUtils.hpp"

namespace My {

namespace {
constexpr size_t kFreeFlag = 1;
constexpr size_t kSizeMask = ~size_t(15);
constexpr size_t kArenaAlignment = 64;
}  // namespace

struct alignas(16) TlsfAllocator::Block {
    Block* prev;  // previous physical block, nullptr for the first
    size_t sizeFlags;

    size_t Size() const { return sizeFlags & kSizeMask; }
    bool IsFree() const { return (sizeFlags & kFreeFlag) != 0; }
    void* Payload() { return reinterpret_cast<uint8_t*>(this) + kHeaderSize; }
    // Free-list links overlay the payload of free blocks.
    Block** Links() { return static_cast<Block**>(Payload()); }
    Block*& NextFree() { return Links()[0]; }
    Block*& PrevFree() { return Links()[1]; }
};

TlsfAllocator::TlsfAllocator(IMemoryManager* pMmgr, size_t capacity)
    : IAllocator(pMmgr) {
    Initialize(capacity);
}

TlsfAllocator::~TlsfAllocator() { Release(); }

void TlsfAllocator::Release() {
    if (m_pBase) m_pMemoryManager->FreePage(m_pBase);
    m_pBase = nullptr;
    m_szCapacity = 0;
    m_szUsed = m_szPeak = 0;
    m_nAllocations = 0;
}

bool TlsfAllocator::Initialize(size_t capacity) {
    Release();
    capacity &= ~(kAlignment - 1);
    if (capacity < kMinCapacity) return false;

    m_pBase = static_cast<uint8_t*>(
        m_pMemoryManager->AllocateAlignedPage(capacity, kArenaAlignment));
    if (!m_pBase) return false;
    m_szCapacity = capacity;
    m_nFailures = m_nInvalidFrees = 0;
    FreeAll();
    return true;
}

TlsfAllocator::Block* TlsfAllocator::Next(const Block* b) const {
    return reinterpret_cast<Block*>(
        reinterpret_cast<uint8_t*>(const_cast<Block*>(b)) + b->Size());
}

void TlsfAllocator::Mapping(size_t size, size_t& fl, size_t& sl) {
    if (size < kSmallLimit) {
        fl = 0;
        sl = size >> 3;
    } else {
        const size_t log2 = std::bit_width(size) - 1;
        fl = log2 - 7;
        sl = (size >> (log2 - kSlLog2)) - kSlCount;
    }
}

void TlsfAllocator::InsertFree(Block* b) {
    size_t fl, sl;
    Mapping(b->Size(), fl, sl);
    b->sizeFlags |= kFreeFlag;
    b->PrevFree() = nullptr;
    b->NextFree() = m_Bins[fl][sl];
    if (b->NextFree()) b->NextFree()->PrevFree() = b;
    m_Bins[fl][sl] = b;
    m_FlBitmap |= uint64_t(1) << fl;
    m_SlBitmap[fl] |= uint32_t(1) << sl;
}

void TlsfAllocator::RemoveFree(Block* b) {
    size_t fl, sl;
    Mapping(b->Size(), fl, sl);
    if (b->NextFree()) b->NextFree()->PrevFree() = b->PrevFree();
    if (b->PrevFree()) {
        b->PrevFree()->NextFree() = b->NextFree();
    } else {
        m_Bins[fl][sl] = b->NextFree();
        if (!m_Bins[fl][sl]) {
            m_SlBitmap[fl] &= ~(uint32_t(1) << sl);
            if (!m_SlBitmap[fl]) m_FlBitmap &= ~(uint64_t(1) << fl);
        }
    }
    b->sizeFlags &= ~kFreeFlag;
}

TlsfAllocator::Block* TlsfAllocator::FindSuitable(size_t size) {
    // round up so that every block in the chosen bin is large enough
    if (size >= kSmallLimit) {
        const size_t log2 = std::bit_width(size) - 1;
        size += (size_t(1) << (log2 - kSlLog2)) - 1;
    }
    size_t fl, sl;
    Mapping(size, fl, sl);

    uint32_t slMap = m_SlBitmap[fl] & (~uint32_t(0) << sl);
    if (!slMap) {
        const uint64_t flMap =
            fl + 1 < kFlCount ? m_FlBitmap & (~uint64_t(0) << (fl + 1)) : 0;
        if (!flMap) return nullptr;
        fl = std::countr_zero(flMap);
        slMap = m_SlBitmap[fl];
    }
    sl = std::countr_zero(slMap);
    return m_Bins[fl][sl];
}

void* TlsfAllocator::Allocate(size_t size, size_t alignment) {
    if (!m_pBase || size == 0 || !IsPowerOfTwo(alignment) ||
        size > m_szCapacity || alignment > m_szCapacity) {
        ++m_nFailures;
        return nullptr;
    }

    size_t payload = (size + kAlignment - 1) & ~(kAlignment - 1);
    const size_t need = payload + kHeaderSize;
    const size_t slack = alignment > kAlignment ? alignment + kMinBlockSize : 0;
    if (need > m_szCapacity || slack > m_szCapacity - need) {
        ++m_nFailures;
        return nullptr;
    }

    Block* b = FindSuitable(need + slack);
    if (!b) {
        ++m_nFailures;
        return nullptr;
    }
    RemoveFree(b);

    if (alignment > kAlignment) {
        const auto first = reinterpret_cast<uintptr_t>(b->Payload());
        uintptr_t aligned = (first + alignment - 1) & ~(alignment - 1);
        size_t gap = aligned - first;
        if (gap != 0 && gap < kMinBlockSize) {
            aligned += alignment;
            gap += alignment;
        }
        if (gap != 0) {
            auto* c = reinterpret_cast<Block*>(reinterpret_cast<uint8_t*>(b) + gap);
            c->prev = b;
            c->sizeFlags = b->Size() - gap;
            b->sizeFlags = gap;
            Next(c)->prev = c;
            InsertFree(b);
            b = c;
        }
    }

    const size_t remainder = b->Size() - need;
    if (remainder >= kMinBlockSize) {
        auto* r = reinterpret_cast<Block*>(reinterpret_cast<uint8_t*>(b) + need);
        r->prev = b;
        r->sizeFlags = remainder;
        b->sizeFlags = need;
        Next(r)->prev = r;
        InsertFree(r);
    }

    m_szUsed += b->Size();
    if (m_szUsed > m_szPeak) m_szPeak = m_szUsed;
    ++m_nAllocations;

    void* p = b->Payload();
#if defined(_DEBUG)
    std::memset(p, PATTERN_ALLOC, b->Size() - kHeaderSize);
#endif
    return p;
}

bool TlsfAllocator::IsValidUsedBlock(const Block* b) const {
    const auto* lo = m_pBase;
    const auto* sentinel = m_pBase + m_szCapacity - kHeaderSize;
    const auto* raw = reinterpret_cast<const uint8_t*>(b);
    if (raw < lo || raw >= sentinel) return false;
    if (b->IsFree()) return false;
    const size_t size = b->Size();
    if (size < kMinBlockSize || size > static_cast<size_t>(sentinel - raw)) {
        return false;
    }
    if (Next(b)->prev != b) return false;
    if (raw == lo) return b->prev == nullptr;
    const auto* prev = reinterpret_cast<const uint8_t*>(b->prev);
    return prev >= lo && prev < raw &&
           reinterpret_cast<const uint8_t*>(Next(b->prev)) == raw;
}

void TlsfAllocator::Free(void* p) {
    if (!p) return;
    const auto* raw = static_cast<const uint8_t*>(p);
    if (!m_pBase || raw < m_pBase + kHeaderSize ||
        raw >= m_pBase + m_szCapacity - kHeaderSize ||
        reinterpret_cast<uintptr_t>(p) % kAlignment != 0) {
        ++m_nInvalidFrees;
        return;
    }
    auto* b = reinterpret_cast<Block*>(static_cast<uint8_t*>(p) - kHeaderSize);
    if (!IsValidUsedBlock(b)) {
        ++m_nInvalidFrees;
        return;
    }

    m_szUsed -= b->Size();
    --m_nAllocations;
#if defined(_DEBUG)
    std::memset(p, PATTERN_FREE, b->Size() - kHeaderSize);
#endif

    Block* next = Next(b);
    if (next->IsFree()) {
        RemoveFree(next);
        b->sizeFlags += next->Size();
    }
    if (b->prev && b->prev->IsFree()) {
        Block* prev = b->prev;
        RemoveFree(prev);
        prev->sizeFlags += b->Size();
        b = prev;
    }
    Next(b)->prev = b;
    InsertFree(b);
}

void TlsfAllocator::FreeAll() {
    if (!m_pBase) return;
    m_FlBitmap = 0;
    std::memset(m_SlBitmap, 0, sizeof(m_SlBitmap));
    std::memset(m_Bins, 0, sizeof(m_Bins));
    m_szUsed = 0;
    m_nAllocations = 0;

    auto* first = reinterpret_cast<Block*>(m_pBase);
    first->prev = nullptr;
    first->sizeFlags = m_szCapacity - kHeaderSize;
    auto* sentinel = Next(first);
    sentinel->prev = first;
    sentinel->sizeFlags = 0;  // zero size, marked used
    InsertFree(first);
}

size_t TlsfAllocator::LargestFreeBlock() const {
    if (!m_pBase) return 0;
    size_t largest = 0;
    for (const Block* b = reinterpret_cast<const Block*>(m_pBase); b->Size();
         b = Next(b)) {
        if (b->IsFree() && b->Size() > largest) largest = b->Size();
    }
    return largest;
}

bool TlsfAllocator::Check() const {
    if (!m_pBase) return true;
    size_t total = 0, used = 0, freeBlocks = 0, live = 0;
    const Block* prev = nullptr;
    const Block* b = reinterpret_cast<const Block*>(m_pBase);
    for (; b->Size(); b = Next(b)) {
        if (b->prev != prev || b->Size() < kMinBlockSize) return false;
        if (prev && prev->IsFree() && b->IsFree()) return false;
        total += b->Size();
        if (total > m_szCapacity - kHeaderSize) return false;
        if (b->IsFree()) {
            ++freeBlocks;
        } else {
            used += b->Size();
            ++live;
        }
        prev = b;
    }
    // b is now the sentinel
    if (total != m_szCapacity - kHeaderSize || b->prev != prev || b->IsFree()) {
        return false;
    }
    if (used != m_szUsed || live != m_nAllocations) return false;

    size_t listed = 0;
    for (size_t fl = 0; fl < kFlCount; fl++) {
        if (((m_FlBitmap >> fl) & 1) != (m_SlBitmap[fl] != 0)) return false;
        for (size_t sl = 0; sl < kSlCount; sl++) {
            const Block* head = m_Bins[fl][sl];
            if (((m_SlBitmap[fl] >> sl) & 1) != (head != nullptr)) return false;
            const Block* prevFree = nullptr;
            for (const Block* f = head; f; f = const_cast<Block*>(f)->NextFree()) {
                size_t efl, esl;
                Mapping(f->Size(), efl, esl);
                if (!f->IsFree() || efl != fl || esl != sl) return false;
                if (const_cast<Block*>(f)->PrevFree() != prevFree) return false;
                prevFree = f;
                if (++listed > freeBlocks) return false;
            }
        }
    }
    return listed == freeBlocks;
}
}  // namespace My

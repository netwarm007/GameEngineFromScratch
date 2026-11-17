#include "BlockAllocator.hpp"

#include <cassert>
#include <cstring>

using namespace My;

namespace {
constexpr size_t AlignUp(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }
}  // namespace

BlockAllocator::BlockAllocator(IMemoryManager* pMmgr, size_t data_size,
                               size_t page_size, size_t alignment)
    : IAllocator(pMmgr) {
    Reset(data_size, page_size, alignment);
}

BlockAllocator::~BlockAllocator() { FreeAll(); }

bool BlockAllocator::Reset(size_t data_size, size_t page_size,
                           size_t alignment) {
    FreeAll();
    m_bConfigured = false;
    m_nBlocksPerPage = 0;

    // the masks below only work for power-of-two alignments
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;

    size_t minimal_size =
        (sizeof(BlockHeader) > data_size) ? sizeof(BlockHeader) : data_size;
    size_t block_size = AlignUp(minimal_size, alignment);
    // overflow guard
    if (block_size < minimal_size) return false;

    // page memory is aligned to the block alignment and the header is padded
    // so the first block is aligned as well
    size_t page_align =
        alignment > alignof(PageHeader) ? alignment : alignof(PageHeader);
    size_t header_size = AlignUp(sizeof(PageHeader), page_align);
    if (page_size <= header_size) return false;

    size_t blocks_per_page = (page_size - header_size) / block_size;
    if (blocks_per_page == 0) return false;

    m_szPageSize = page_size;
    m_szAlignment = page_align;
    m_szHeaderSize = header_size;
    m_szBlockSize = block_size;
    m_szAlignmentSize = block_size - minimal_size;
    m_nBlocksPerPage = blocks_per_page;
    m_bConfigured = true;
    return true;
}

void* BlockAllocator::Allocate(size_t size) {
    if (size > m_szBlockSize) return nullptr;
    return Allocate();
}

void* BlockAllocator::Allocate() {
    if (!m_bConfigured) return nullptr;

    BlockHeader* block = nullptr;

    if (m_pFreeList) {
        block = m_pFreeList;
        m_pFreeList = block->pNext;
    } else {
        if (m_pBump == m_pBumpEnd) {
            auto* pNewPage = static_cast<PageHeader*>(
                m_pMemoryManager->AllocateAlignedPage(m_szPageSize,
                                                      m_szAlignment));
            if (!pNewPage) return nullptr;

            pNewPage->pNext = m_pPageList;
            m_pPageList = pNewPage;
            m_pBump = PageBlocks(pNewPage);
            m_pBumpEnd = m_pBump + m_nBlocksPerPage * m_szBlockSize;

            ++m_nPages;
            m_nBlocks += m_nBlocksPerPage;
            m_nFreeBlocks += m_nBlocksPerPage;
        }

        block = reinterpret_cast<BlockHeader*>(m_pBump);
        m_pBump += m_szBlockSize;
    }

    --m_nFreeBlocks;

#if defined(_DEBUG)
    FillAllocatedBlock(block);
#endif

    return block;
}

bool BlockAllocator::OwnsBlock(const void* p) const {
    auto addr = reinterpret_cast<uintptr_t>(p);
    for (PageHeader* page = m_pPageList; page; page = page->pNext) {
        auto first = reinterpret_cast<uintptr_t>(PageBlocks(page));
        auto end = first + m_nBlocksPerPage * m_szBlockSize;
        if (addr >= first && addr < end) {
            return (addr - first) % m_szBlockSize == 0;
        }
    }
    return false;
}

bool BlockAllocator::IsOnFreeList(const BlockHeader* pBlock) const {
    for (const BlockHeader* b = m_pFreeList; b; b = b->pNext) {
        if (b == pBlock) return true;
    }
    return false;
}

void BlockAllocator::Free(void* p) {
    if (!p) return;

    auto* block = static_cast<BlockHeader*>(p);

    if (m_bValidate) {
        auto bump = reinterpret_cast<uintptr_t>(m_pBump);
        auto addr = reinterpret_cast<uintptr_t>(p);
        bool inUntouched = bump != 0 &&
                           addr >= bump && addr < reinterpret_cast<uintptr_t>(m_pBumpEnd);
        if (!OwnsBlock(p) || inUntouched || IsOnFreeList(block)) {
            ++m_nInvalidFrees;
            return;
        }
    }

#if defined(_DEBUG)
    FillFreeBlock(block);
#endif

    block->pNext = m_pFreeList;
    m_pFreeList = block;
    ++m_nFreeBlocks;
}

void BlockAllocator::FreeAll() {
    PageHeader* pPage = m_pPageList;
    while (pPage) {
        PageHeader* _p = pPage;
        pPage = pPage->pNext;

        m_pMemoryManager->FreePage(_p);
    }

    m_pPageList = nullptr;
    m_pFreeList = nullptr;
    m_pBump = nullptr;
    m_pBumpEnd = nullptr;

    m_nPages = 0;
    m_nBlocks = 0;
    m_nFreeBlocks = 0;
}

#if defined(_DEBUG)
void BlockAllocator::FillFreeBlock(BlockHeader* pBlock) {
    // block header + data
    memset(pBlock, PATTERN_FREE, m_szBlockSize - m_szAlignmentSize);

    // alignment
    memset(
        reinterpret_cast<uint8_t*>(pBlock) + m_szBlockSize - m_szAlignmentSize,
        PATTERN_ALIGN, m_szAlignmentSize);
}

void BlockAllocator::FillAllocatedBlock(BlockHeader* pBlock) {
    // block header + data
    memset(pBlock, PATTERN_ALLOC, m_szBlockSize - m_szAlignmentSize);

    // alignment
    memset(
        reinterpret_cast<uint8_t*>(pBlock) + m_szBlockSize - m_szAlignmentSize,
        PATTERN_ALIGN, m_szAlignmentSize);
}
#endif

#pragma once
#include <cstddef>
#include <cstdint>

#include "IAllocator.hpp"

namespace My {

struct BlockHeader {
    // union-ed with data
    BlockHeader* pNext;
};

struct PageHeader {
    PageHeader* pNext;
};

// Fixed-size block (pool) allocator. Not thread-safe.
//
// Every block address is aligned to the configured alignment. Blocks are
// carved lazily from the newest page, so adding a page is O(1). Allocate
// returns nullptr when the request is too large, the allocator is not
// configured, or the memory manager is out of memory.
class BlockAllocator : _implements_ IAllocator {
   public:
    explicit BlockAllocator(IMemoryManager* pMmgr) : IAllocator(pMmgr) {}
    BlockAllocator(IMemoryManager* pMmgr, size_t data_size, size_t page_size,
                   size_t alignment);
    ~BlockAllocator() override;
    // disable copy & assignment
    BlockAllocator(const BlockAllocator& clone) = delete;
    BlockAllocator& operator=(const BlockAllocator& rhs) = delete;

    // Releases all pages and applies a new configuration. Returns false (and
    // leaves the allocator unusable) if alignment is not a power of two or the
    // page cannot hold a single block.
    bool Reset(size_t data_size, size_t page_size, size_t alignment);

    // alloc and free blocks
    void* Allocate();
    void* Allocate(size_t size) override;
    // Free(nullptr) is a no-op. With validation enabled, foreign, misaligned
    // and double-freed pointers are rejected and counted instead of corrupting
    // the free list.
    void Free(void* p) override;
    void FreeAll() override;

    // Validation walks the page and free lists on every Free. Default: on in
    // debug builds only.
    void SetValidation(bool enable) { m_bValidate = enable; }

    bool IsConfigured() const { return m_bConfigured; }
    size_t BlockSize() const { return m_szBlockSize; }
    size_t BlocksPerPage() const { return m_nBlocksPerPage; }
    size_t PageCount() const { return m_nPages; }
    size_t BlockCount() const { return m_nBlocks; }
    size_t FreeBlockCount() const { return m_nFreeBlocks; }
    size_t InvalidFreeCount() const { return m_nInvalidFrees; }

   private:
#if defined(_DEBUG)
    // fill a block with debug patterns
    void FillFreeBlock(BlockHeader* pBlock);
    void FillAllocatedBlock(BlockHeader* pBlock);
#endif

    bool OwnsBlock(const void* p) const;
    bool IsOnFreeList(const BlockHeader* pBlock) const;
    uint8_t* PageBlocks(PageHeader* pPage) const {
        return reinterpret_cast<uint8_t*>(pPage) + m_szHeaderSize;
    }

    // the page list
    PageHeader* m_pPageList = nullptr;

    // the free block list
    BlockHeader* m_pFreeList = nullptr;

    // untouched blocks at the end of the newest page
    uint8_t* m_pBump = nullptr;
    uint8_t* m_pBumpEnd = nullptr;

    size_t m_szPageSize = 0;
    size_t m_szHeaderSize = 0;
    size_t m_szAlignment = 0;
    size_t m_szAlignmentSize = 0;
    size_t m_szBlockSize = 0;
    size_t m_nBlocksPerPage = 0;
    bool m_bConfigured = false;
#if defined(_DEBUG)
    bool m_bValidate = true;
#else
    bool m_bValidate = false;
#endif

    // statistics
    size_t m_nPages = 0;
    size_t m_nBlocks = 0;
    size_t m_nFreeBlocks = 0;
    size_t m_nInvalidFrees = 0;
};
}  // namespace My

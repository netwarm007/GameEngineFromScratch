#include <stddef.h>
#include <stdint.h>

namespace My {

    struct BlockHeader {
        // union-ed with data
        BlockHeader* pNext;
    };

    struct PageHeader {
        PageHeader* pNext;
        BlockHeader* Blocks() {
                return reinterpret_cast<BlockHeader*>(this + 1);
        }
    };

    class Allocator {
        public:
                // debug patterns
                static const uint8_t PATTERN_ALIGN = 0xFC;
                static const uint8_t PATTERN_ALLOC = 0xFD;
                static const uint8_t PATTERN_FREE  = 0xFE;

                Allocator();
                Allocator(size_t data_size, size_t page_size, size_t alignment);
                ~Allocator();

                // resets the allocator to a new configuration
                void Reset(size_t data_size, size_t page_size, size_t alignment);

                // alloc and free blocks
                void* Allocate();
                void  Free(void* p);
                void  FreeAll();
        private:
#if defined(_DEBUG)
                // fill a free page with debug patterns
                void FillFreePage(PageHeader* pPage);

                // fill a block with debug patterns
                void FillFreeBlock(BlockHeader* pBlock);

                // fill an allocated block with debug patterns
                void FillAllocatedBlock(BlockHeader* pBlock);
#endif

                // gets the next block
                BlockHeader* NextBlock(BlockHeader* pBlock);

                // the page list
                PageHeader* m_pPageList;

                // the free block list
                BlockHeader* m_pFreeList;

                size_t      m_szDataSize;
                size_t      m_szPageSize;
                size_t      m_szAlignmentSize;
                size_t      m_szBlockSize;
                size_t      m_nBlocksPerPage;

                // statistics
                uint32_t    m_nPages;
                uint32_t    m_nBlocks;
                uint32_t    m_nFreeBlocks;

                // disable copy & assignment
                Allocator(const Allocator& clone);
                Allocator &operator=(const Allocator &rhs);
    };
}


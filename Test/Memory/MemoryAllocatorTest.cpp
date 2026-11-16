#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "BlockAllocator.hpp"
#include "MemoryManager.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool Aligned(const void* p, size_t a) {
    return reinterpret_cast<uintptr_t>(p) % a == 0;
}

// Memory source that can be made to fail.
class FailingMemoryManager : public MemoryManager {
   public:
    void* AllocateAlignedPage(size_t size, size_t alignment) override {
        if (fail) return nullptr;
        return MemoryManager::AllocateAlignedPage(size, alignment);
    }
    bool fail = false;
};

void TestMemoryManager() {
    MemoryManager mm;
    Expect(mm.Initialize() == 0, "initialize succeeds");

    Expect(mm.AllocatePage(0) == nullptr, "zero-size page rejected");
    Expect(mm.AllocateAlignedPage(64, 0) == nullptr, "zero alignment rejected");
    Expect(mm.AllocateAlignedPage(64, 48) == nullptr,
           "non power-of-two alignment rejected");
    Expect(mm.PageCount() == 0, "rejected requests allocate nothing");

    void* a = mm.AllocatePage(100);
    Expect(a != nullptr, "plain page allocated");
    Expect(mm.PageCount() == 1 && mm.AllocatedBytes() == 100,
           "plain page tracked");

    for (size_t align : {16u, 64u, 256u, 4096u}) {
        void* p = mm.AllocateAlignedPage(100, align);
        Expect(p != nullptr && Aligned(p, align), "aligned page is aligned");
        mm.FreePage(p);
    }
    Expect(mm.PageCount() == 1, "aligned pages released");
    Expect(mm.PeakAllocatedBytes() >= 100 + 100, "peak tracked");

    void* b = mm.AllocateAlignedPage(5000, 4096);
    Expect(mm.AllocatedBytes() == 5100, "bytes counted by requested size");
    mm.FreePage(b);
    Expect(mm.AllocatedBytes() == 100, "bytes decrease on free");
    Expect(mm.PeakAllocatedBytes() >= 5100, "peak keeps high-water mark");

    mm.FreePage(nullptr);
    Expect(mm.InvalidFreeCount() == 1, "null free counted");
    int local;
    mm.FreePage(&local);
    Expect(mm.InvalidFreeCount() == 2, "foreign free counted, not released");
    mm.FreePage(a);
    mm.FreePage(a);
    Expect(mm.InvalidFreeCount() == 3, "double free counted");
    Expect(mm.PageCount() == 0 && mm.AllocatedBytes() == 0,
           "all pages released");

    mm.Finalize();
}

void TestConfiguration() {
    MemoryManager mm;
    {
        BlockAllocator unconfigured(&mm);
        Expect(!unconfigured.IsConfigured(), "default ctor unconfigured");
        Expect(unconfigured.Allocate() == nullptr,
               "unconfigured allocate returns null");
        Expect(unconfigured.PageCount() == 0 && unconfigured.BlockCount() == 0 &&
                   unconfigured.FreeBlockCount() == 0,
               "unconfigured statistics are zero");
    }

    BlockAllocator a(&mm);
    Expect(!a.Reset(16, 4096, 0), "zero alignment rejected");
    Expect(!a.Reset(16, 4096, 24), "non power-of-two alignment rejected");
    Expect(!a.Reset(16, 0, 16), "zero page size rejected");
    Expect(!a.Reset(128, 64, 16), "page smaller than a block rejected");
    Expect(!a.Reset(64, 16 + 64 - 1, 16), "page one byte short rejected");
    Expect(!a.Reset(~size_t(0) - 2, 4096, 16), "size overflow rejected");
    Expect(a.Allocate() == nullptr, "rejected configuration cannot allocate");
    Expect(mm.PageCount() == 0, "rejected configuration allocates no pages");

    Expect(a.Reset(1, 4096, 1), "tiny block, byte alignment accepted");
    Expect(a.BlockSize() == sizeof(void*),
           "block grows to hold the free-list link");
    Expect(a.Reset(20, 4096, 16), "20 bytes at 16 accepted");
    Expect(a.BlockSize() == 32, "block size rounded to alignment");
    Expect(a.BlocksPerPage() >= 1 && a.BlocksPerPage() * 32 < 4096,
           "header reserved inside the page");
    Expect(a.Reset(48, 128, 64), "alignment larger than data accepted");
    Expect(a.BlockSize() == 64, "block size equals alignment");
    Expect(a.BlocksPerPage() == 1, "exactly one block fits");
}

void TestAllocationBasics() {
    MemoryManager mm;
    {
        BlockAllocator a(&mm, 24, 1024, 8);
        Expect(a.IsConfigured(), "constructor configures");
        Expect(a.Allocate(25) == nullptr, "oversized request rejected");
        Expect(a.PageCount() == 0, "rejected request allocates no page");

        void* p = a.Allocate(24);
        Expect(p != nullptr, "sized allocate works");
        Expect(a.PageCount() == 1 && a.BlockCount() == a.BlocksPerPage(),
               "first page created");
        Expect(a.FreeBlockCount() == a.BlocksPerPage() - 1,
               "free block count tracks allocation");

        a.Free(nullptr);
        Expect(a.FreeBlockCount() == a.BlocksPerPage() - 1,
               "freeing null changes nothing");

        a.Free(p);
        Expect(a.FreeBlockCount() == a.BlocksPerPage(), "free restores count");
        Expect(a.Allocate() == p, "freed block is reused first");

        Expect(mm.PageCount() == 1, "allocator holds page until destroyed");
    }
    Expect(mm.PageCount() == 0, "destructor releases pages");
    Expect(mm.InvalidFreeCount() == 0, "no invalid page frees");
}

void TestAlignmentAndGrowth() {
    for (size_t align : {1u, 8u, 16u, 32u, 64u, 128u, 256u}) {
        MemoryManager mm;
        BlockAllocator a(&mm, 40, 1024, align);
        Expect(a.IsConfigured(), "configuration accepted");

        std::vector<uint8_t*> blocks;
        std::set<uintptr_t> unique;
        const size_t count = a.BlocksPerPage() * 5 + 3;
        for (size_t i = 0; i < count; i++) {
            auto* p = static_cast<uint8_t*>(a.Allocate());
            Expect(p != nullptr, "allocation succeeds");
            if (!p) return;
            Expect(Aligned(p, align), "block honors alignment");
            std::memset(p, static_cast<int>(i & 0xFF), 40);
            blocks.push_back(p);
            unique.insert(reinterpret_cast<uintptr_t>(p));
        }
        Expect(unique.size() == count, "all blocks distinct");
        Expect(a.PageCount() == 6, "pages grow on demand");
        Expect(a.BlockCount() == 6 * a.BlocksPerPage(), "block count");
        Expect(a.FreeBlockCount() == 6 * a.BlocksPerPage() - count,
               "free count after growth");

        bool intact = true;
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < 40; j++) {
                if (blocks[i][j] != (i & 0xFF)) intact = false;
            }
        }
        Expect(intact, "blocks do not overlap");

        for (auto* p : blocks) a.Free(p);
        Expect(a.FreeBlockCount() == a.BlockCount(), "everything freed");
        Expect(a.InvalidFreeCount() == 0, "no invalid frees");
    }
}

void TestBlocksPerPageBoundary() {
    MemoryManager mm;
    BlockAllocator a(&mm, 16, 16 + 16 * 4, 16);
    Expect(a.IsConfigured() && a.BlocksPerPage() == 4, "four blocks per page");
    std::vector<void*> blocks;
    for (int i = 0; i < 4; i++) blocks.push_back(a.Allocate());
    Expect(a.PageCount() == 1, "page filled exactly");
    blocks.push_back(a.Allocate());
    Expect(a.PageCount() == 2, "fifth block starts a new page");
    for (auto* p : blocks) a.Free(p);
}

void TestValidation() {
    MemoryManager mm;
    BlockAllocator a(&mm, 32, 1024, 16);
    a.SetValidation(true);

    auto* p = static_cast<uint8_t*>(a.Allocate());
    auto* q = a.Allocate();
    size_t before = a.FreeBlockCount();

    int local;
    a.Free(&local);
    Expect(a.InvalidFreeCount() == 1, "foreign pointer rejected");
    a.Free(p + 1);
    Expect(a.InvalidFreeCount() == 2, "misaligned interior pointer rejected");
    a.Free(p + a.BlockSize() * a.BlocksPerPage() * 2);
    Expect(a.InvalidFreeCount() == 3, "pointer beyond page rejected");
    Expect(a.FreeBlockCount() == before, "invalid frees leave state intact");

    // a never-handed-out block of the current page
    a.Free(static_cast<uint8_t*>(q) + a.BlockSize());
    Expect(a.InvalidFreeCount() == 4, "untouched block rejected");

    a.Free(p);
    Expect(a.FreeBlockCount() == before + 1, "valid free accepted");
    a.Free(p);
    Expect(a.InvalidFreeCount() == 5, "double free rejected");
    Expect(a.FreeBlockCount() == before + 1, "double free leaves count intact");

    void* r1 = a.Allocate();
    void* r2 = a.Allocate();
    Expect(r1 == p, "free list not corrupted by double free");
    Expect(r2 != r1 && r2 != q, "no block handed out twice");
}

void TestFreeAllAndReset() {
    MemoryManager mm;
    BlockAllocator a(&mm, 32, 512, 16);
    for (int i = 0; i < 100; i++) a.Allocate();
    Expect(mm.PageCount() == a.PageCount() && a.PageCount() > 1, "pages held");

    a.FreeAll();
    Expect(mm.PageCount() == 0 && a.PageCount() == 0 && a.BlockCount() == 0 &&
               a.FreeBlockCount() == 0,
           "FreeAll releases everything");
    Expect(a.IsConfigured(), "FreeAll keeps configuration");

    void* p = a.Allocate();
    Expect(p != nullptr && a.PageCount() == 1, "usable after FreeAll");
    a.FreeAll();
    a.FreeAll();
    Expect(mm.InvalidFreeCount() == 0, "repeated FreeAll is harmless");

    a.Allocate();
    Expect(a.Reset(64, 1024, 64), "reset to new configuration");
    Expect(mm.PageCount() == 0, "reset releases pages");
    auto* r = a.Allocate();
    Expect(r != nullptr && Aligned(r, 64) && a.BlockSize() == 64,
           "new configuration applies");

    Expect(!a.Reset(64, 1024, 3), "failed reset reported");
    Expect(mm.PageCount() == 0, "failed reset still releases pages");
    Expect(a.Allocate() == nullptr, "failed reset leaves allocator unusable");
}

void TestOutOfMemory() {
    FailingMemoryManager mm;
    BlockAllocator a(&mm, 32, 256, 16);
    std::vector<void*> blocks;
    for (size_t i = 0; i < a.BlocksPerPage(); i++) blocks.push_back(a.Allocate());
    Expect(a.PageCount() == 1, "first page obtained");

    mm.fail = true;
    Expect(a.Allocate() == nullptr, "out of memory returns null");
    Expect(a.PageCount() == 1 && a.BlockCount() == a.BlocksPerPage() &&
               a.FreeBlockCount() == 0,
           "failed page leaves statistics intact");

    a.Free(blocks.back());
    Expect(a.Allocate() == blocks.back(), "free list works while OOM");

    mm.fail = false;
    Expect(a.Allocate() != nullptr, "recovers when memory is available");
    Expect(a.PageCount() == 2, "second page obtained");
}

void TestRandomizedStress() {
    MemoryManager mm;
    BlockAllocator a(&mm, 48, 2048, 32);
    a.SetValidation(true);

    std::mt19937 rng(12345);
    struct Live {
        uint8_t* p;
        uint8_t tag;
    };
    std::vector<Live> live;
    bool intact = true;

    for (int step = 0; step < 20000; step++) {
        if (live.empty() || rng() % 100 < 55) {
            auto* p = static_cast<uint8_t*>(a.Allocate());
            if (!p) {
                intact = false;
                break;
            }
            auto tag = static_cast<uint8_t>(rng());
            std::memset(p, tag, 48);
            live.push_back({p, tag});
        } else {
            size_t i = rng() % live.size();
            for (size_t j = 0; j < 48; j++) {
                if (live[i].p[j] != live[i].tag) intact = false;
            }
            a.Free(live[i].p);
            live[i] = live.back();
            live.pop_back();
        }
    }
    Expect(intact, "live blocks keep their contents");
    Expect(a.InvalidFreeCount() == 0, "no invalid frees during stress");
    Expect(a.BlockCount() - a.FreeBlockCount() == live.size(),
           "statistics match live set");

    for (auto& l : live) a.Free(l.p);
    Expect(a.FreeBlockCount() == a.BlockCount(), "all returned");
}
}  // namespace

int main() {
    TestMemoryManager();
    TestConfiguration();
    TestAllocationBasics();
    TestAlignmentAndGrowth();
    TestBlocksPerPageBoundary();
    TestValidation();
    TestFreeAllAndReset();
    TestOutOfMemory();
    TestRandomizedStress();

    if (failures == 0) std::cout << "All memory allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

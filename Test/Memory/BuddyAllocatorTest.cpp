#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "BuddyAllocator.hpp"
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

class FailingMemoryManager : public MemoryManager {
   public:
    void* AllocateAlignedPage(size_t size, size_t alignment) override {
        if (fail) return nullptr;
        return MemoryManager::AllocateAlignedPage(size, alignment);
    }
    void* AllocatePage(size_t size) override {
        if (failPlain) return nullptr;
        return MemoryManager::AllocatePage(size);
    }
    bool fail = false;
    bool failPlain = false;
};

void TestConfiguration() {
    FailingMemoryManager mm;
    {
        BuddyAllocator b(&mm);
        Expect(!b.IsConfigured(), "default unconfigured");
        Expect(b.Allocate(8) == nullptr && b.FailedAllocationCount() == 1,
               "unconfigured allocate fails");
        b.Free(reinterpret_cast<void*>(64));
        Expect(b.InvalidFreeCount() == 1, "unconfigured free rejected");
        Expect(!b.Initialize(1000), "non power-of-two capacity rejected");
        Expect(!b.Initialize(1024, 48), "non power-of-two min block rejected");
        Expect(!b.Initialize(1024, 8), "min block below 16 rejected");
        Expect(!b.Initialize(64, 128), "min block above capacity rejected");
        mm.fail = true;
        Expect(!b.Initialize(4096), "memory manager failure reported");
        mm.fail = false;
        Expect(mm.PageCount() == 0, "failed initialize leaks nothing");
        Expect(b.Initialize(4096, 64) && b.Capacity() == 4096 &&
                   b.MinBlockSize() == 64,
               "initialize");
        Expect(b.Check() && b.LargestFreeBlock() == 4096, "one free block");
        Expect(b.Initialize(8192, 16) && mm.PageCount() == 2,
               "re-initialize releases old memory");
        Expect(b.Initialize(16, 16) && b.Allocate(16) != nullptr,
               "single-unit arena works");
    }
    Expect(mm.PageCount() == 0, "destructor releases memory");
}

void TestBlockSizes() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 4096, 64);
    Expect(b.BlockSizeFor(1) == 64, "1 -> 64");
    Expect(b.BlockSizeFor(64) == 64, "64 -> 64");
    Expect(b.BlockSizeFor(65) == 128, "65 -> 128");
    Expect(b.BlockSizeFor(129) == 256, "129 -> 256");
    Expect(b.BlockSizeFor(4096) == 4096, "whole arena");
    Expect(b.BlockSizeFor(4097) == 0 && b.BlockSizeFor(0) == 0,
           "out of range is 0");
}

void TestSplitAndMerge() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 4096, 64);
    void* p1 = b.Allocate(64);
    Expect(p1 != nullptr && b.Used() == 64, "smallest block");
    Expect(b.LargestFreeBlock() == 2048, "split leaves buddies of each order");
    void* p2 = b.Allocate(64);
    Expect(p2 == static_cast<uint8_t*>(p1) + 64, "buddy handed out next");
    Expect(b.Check(), "consistent after splits");
    b.Free(p1);
    b.Free(p2);
    Expect(b.Check() && b.Used() == 0 && b.LargestFreeBlock() == 4096,
           "freeing both buddies merges up to the whole arena");

    // merge only happens with the matching buddy
    void* a = b.Allocate(64);
    void* c = b.Allocate(64);
    void* d = b.Allocate(64);
    b.Free(c);
    Expect(b.Check() && b.LargestFreeBlock() < 4096, "neighbour still live");
    b.Free(a);
    b.Free(d);
    Expect(b.Check() && b.LargestFreeBlock() == 4096, "full merge in any order");
}

void TestAlignmentOfBlocks() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 1 << 16, 64);
    std::vector<void*> live;
    bool ok = true;
    for (size_t size : {1u, 64u, 65u, 200u, 500u, 1000u, 4000u, 4096u}) {
        void* p = b.Allocate(size);
        if (!p) {
            Expect(false, "allocation failed");
            return;
        }
        const size_t block = b.BlockSizeFor(size);
        ok = ok && Aligned(p, (std::min)(block, BuddyAllocator::kBaseAlignment));
        live.push_back(p);
    }
    Expect(ok, "blocks are aligned to their size (up to the base alignment)");

    for (size_t align : {1u, 16u, 64u, 256u, 1024u, 4096u}) {
        void* p = b.Allocate(8, align);
        Expect(p && Aligned(p, align), "explicit alignment honored");
        live.push_back(p);
    }
    Expect(b.Allocate(8, 8192) == nullptr, "alignment above base rejected");
    Expect(b.Allocate(8, 3) == nullptr, "non power-of-two alignment rejected");
    for (void* p : live) b.Free(p);
    Expect(b.Check() && b.Used() == 0 && b.InvalidFreeCount() == 0, "cleanup");
}

void TestExhaustion() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 4096, 64);
    Expect(b.Allocate(0) == nullptr, "zero size fails");
    Expect(b.Allocate(4097) == nullptr, "oversize fails");
    Expect(b.Allocate(~size_t(0)) == nullptr, "overflowing size fails");

    std::vector<void*> live;
    while (void* p = b.Allocate(64)) live.push_back(p);
    Expect(live.size() == 64 && b.Used() == 4096, "arena filled exactly");
    Expect(b.Allocate(1) == nullptr && b.Check(), "full arena fails");
    for (void* p : live) b.Free(p);
    Expect(b.Check() && b.LargestFreeBlock() == 4096, "drained and merged");

    void* whole = b.Allocate(4096);
    Expect(whole != nullptr && b.Allocate(64) == nullptr,
           "whole-arena allocation uses everything");
    b.Free(whole);

    // fragmentation: free space exists but not contiguous
    std::vector<void*> small;
    for (int i = 0; i < 64; i++) small.push_back(b.Allocate(64));
    for (int i = 0; i < 64; i += 2) b.Free(small[i]);
    Expect(b.Used() == 2048 && b.Allocate(128) == nullptr,
           "no merged block available despite free space");
    for (int i = 1; i < 64; i += 2) b.Free(small[i]);
    Expect(b.Check() && b.LargestFreeBlock() == 4096, "recovers");
}

void TestInvalidFrees() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 4096, 64);
    b.Free(nullptr);
    Expect(b.InvalidFreeCount() == 0, "free(nullptr) is a no-op");

    auto* p = static_cast<uint8_t*>(b.Allocate(200));  // 256-byte block
    int local = 0;
    b.Free(&local);
    b.Free(p + 64);
    b.Free(p + 1);
    b.Free(p - 64);
    Expect(b.InvalidFreeCount() == 4, "foreign, interior, misaligned rejected");
    Expect(b.Used() == 256 && b.Check(), "rejected frees change nothing");

    b.Free(p);
    b.Free(p);
    Expect(b.InvalidFreeCount() == 5 && b.Check(), "double free rejected");
    b.Free(p + 64);
    Expect(b.InvalidFreeCount() == 6 && b.Check(),
           "pointer into a free block rejected");
}

void TestFreeAll() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 1 << 14, 64);
    void* first = b.Allocate(300);
    for (int i = 0; i < 20; i++) b.Allocate(100 + i);
    b.FreeAll();
    Expect(b.Check() && b.Used() == 0 && b.AllocationCount() == 0, "reset");
    Expect(b.Allocate(300) == first && b.LargestFreeBlock() == 8192,
           "usable from scratch after FreeAll");
}

void TestDataIntegrity() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 1 << 16, 64);
    std::vector<std::pair<uint8_t*, size_t>> live;
    for (size_t n = 1; n < 2000; n += 37) {
        auto* p = static_cast<uint8_t*>(b.Allocate(n));
        if (!p) break;
        std::memset(p, int(n & 0xFF), n);
        live.emplace_back(p, n);
    }
    bool intact = true;
    for (auto& [p, n] : live) {
        for (size_t j = 0; j < n; j++) intact = intact && p[j] == (n & 0xFF);
    }
    Expect(intact && live.size() > 20, "blocks do not overlap");
    for (auto& [p, n] : live) b.Free(p);
    Expect(b.Check() && b.Used() == 0, "cleanup");
}

void TestStateAllocationFailure() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm);
    mm.failPlain = true;
    Expect(!b.Initialize(4096) && mm.PageCount() == 0,
           "state table failure releases the arena");
    mm.failPlain = false;
    Expect(b.Initialize(4096), "initialize recovers");
}

void TestOversizeAndFullArena() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm);
    Expect(b.Initialize(1024, 64), "initialize");
    Expect(b.Allocate(2048) == nullptr && b.Allocate(8, 4096) == nullptr &&
               b.FailedAllocationCount() == 2,
           "requests above capacity fail");
    Expect(b.Allocate(1024) != nullptr && b.LargestFreeBlock() == 0,
           "full arena has no free block");
}

void TestCheckDetectsCorruption() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm);
    Expect(b.Initialize(1024, 64), "initialize");
    auto* a = static_cast<uint8_t*>(b.Allocate(64));
    Expect(a != nullptr && b.Check(), "healthy before corruption");
    // the neighbouring free block stores its list links in its first bytes
    std::memset(a + 64, 0xFF, 2 * sizeof(void*));
    Expect(!b.Check(), "corrupted free list link detected");
    std::memset(a + 64, 0, 2 * sizeof(void*));
}

void TestRandomStress() {
    FailingMemoryManager mm;
    BuddyAllocator b(&mm, 1 << 20, 64);
    std::mt19937 rng(31337);
    struct Entry {
        uint8_t* p;
        size_t n;
        uint8_t tag;
    };
    std::vector<Entry> live;
    bool intact = true;

    for (int step = 0; step < 40000; step++) {
        const unsigned r = rng() % 100;
        if (r < 55 || live.empty()) {
            const unsigned kind = rng() % 10;
            const size_t n = kind < 7 ? 1 + rng() % 300
                                      : kind < 9 ? 1 + rng() % 5000
                                                 : 1 + rng() % 70000;
            auto* p = static_cast<uint8_t*>(b.Allocate(n));
            if (!p) continue;
            const uint8_t tag = uint8_t(rng());
            std::memset(p, tag, n);
            live.push_back({p, n, tag});
        } else {
            const size_t i = rng() % live.size();
            const Entry e = live[i];
            for (size_t j = 0; j < e.n; j++) intact = intact && e.p[j] == e.tag;
            b.Free(e.p);
            live[i] = live.back();
            live.pop_back();
        }
        if (step % 2000 == 0 && !b.Check()) {
            Expect(false, "inconsistent during stress");
            return;
        }
    }
    Expect(intact, "contents survive random workload");
    Expect(b.Check() && b.AllocationCount() == live.size(), "state matches");
    for (auto& e : live) b.Free(e.p);
    Expect(b.Check() && b.Used() == 0 && b.InvalidFreeCount() == 0,
           "drained cleanly");
    Expect(b.LargestFreeBlock() == b.Capacity(), "fully merged after stress");
}
}  // namespace

int main() {
    TestConfiguration();
    TestBlockSizes();
    TestSplitAndMerge();
    TestAlignmentOfBlocks();
    TestExhaustion();
    TestInvalidFrees();
    TestFreeAll();
    TestDataIntegrity();
    TestStateAllocationFailure();
    TestOversizeAndFullArena();
    TestCheckDetectsCorruption();
    TestRandomStress();

    if (failures == 0) std::cout << "All buddy allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "MemoryManager.hpp"
#include "TlsfAllocator.hpp"

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
    bool fail = false;
};

void TestConfiguration() {
    FailingMemoryManager mm;
    {
        TlsfAllocator t(&mm);
        Expect(!t.IsConfigured(), "default unconfigured");
        Expect(t.Allocate(8) == nullptr && t.FailedAllocationCount() == 1,
               "unconfigured allocate fails");
        t.Free(reinterpret_cast<void*>(64));
        Expect(t.InvalidFreeCount() == 1, "unconfigured free rejected");
        Expect(!t.Initialize(0) && !t.Initialize(32), "tiny capacity rejected");
        mm.fail = true;
        Expect(!t.Initialize(4096), "memory manager failure reported");
        mm.fail = false;
        Expect(t.Initialize(4100) && t.Capacity() == 4096,
               "capacity rounded down to 16");
        Expect(t.Check() && t.Used() == 0 &&
                   t.LargestFreeBlock() == 4096 - TlsfAllocator::kHeaderSize,
               "fresh arena is a single free block");
        Expect(t.Initialize(8192) && mm.PageCount() == 1,
               "re-initialize releases old arena");
    }
    Expect(mm.PageCount() == 0, "destructor releases arena");
}

void TestBasic() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 18);
    std::vector<std::pair<uint8_t*, size_t>> live;
    std::set<void*> unique;
    bool aligned = true;
    for (size_t n = 1; n <= 700; n += 3) {
        auto* p = static_cast<uint8_t*>(t.Allocate(n));
        if (!p) {
            Expect(false, "allocation failed");
            return;
        }
        aligned = aligned && Aligned(p, 16);
        std::memset(p, int(n & 0xFF), n);
        live.emplace_back(p, n);
        unique.insert(p);
    }
    Expect(aligned && unique.size() == live.size(), "aligned, distinct");
    Expect(t.Check() && t.AllocationCount() == live.size(), "heap consistent");

    bool intact = true;
    for (auto& [p, n] : live) {
        for (size_t j = 0; j < n; j++) intact = intact && p[j] == (n & 0xFF);
    }
    Expect(intact, "blocks do not overlap");

    for (auto& [p, n] : live) t.Free(p);
    Expect(t.Check() && t.Used() == 0 && t.InvalidFreeCount() == 0,
           "all freed");
    Expect(t.LargestFreeBlock() == t.Capacity() - TlsfAllocator::kHeaderSize,
           "everything coalesced back into one block");
    Expect(t.PeakUsed() > 0, "peak tracked");
}

void TestCoalescing() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 14);
    void* a = t.Allocate(1000);
    void* b = t.Allocate(1000);
    void* c = t.Allocate(1000);
    void* d = t.Allocate(1000);
    t.Free(b);
    t.Free(d);
    Expect(t.Check(), "holes recorded");
    t.Free(c);  // merges with both neighbours
    Expect(t.Check(), "merge both sides consistent");
    void* big = t.Allocate(2900);
    Expect(big == b, "merged region is reusable as one block");
    t.Free(a);
    t.Free(big);
    Expect(t.Check() &&
               t.LargestFreeBlock() == t.Capacity() - TlsfAllocator::kHeaderSize,
           "order of frees does not matter");
}

void TestAlignment() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 18);
    std::vector<void*> live;
    bool ok = true;
    for (size_t align = 1; align <= 4096; align <<= 1) {
        for (size_t size : {1u, 17u, 100u, 1000u}) {
            void* p = t.Allocate(size, align);
            if (!p) {
                Expect(false, "aligned allocation failed");
                return;
            }
            ok = ok && Aligned(p, align) && Aligned(p, 16);
            std::memset(p, 0xAB, size);
            live.push_back(p);
        }
        ok = ok && t.Check();
    }
    Expect(ok, "alignment honored, heap stays consistent");
    for (void* p : live) t.Free(p);
    Expect(t.Check() && t.InvalidFreeCount() == 0 && t.Used() == 0,
           "aligned blocks free cleanly");
    Expect(t.LargestFreeBlock() == t.Capacity() - TlsfAllocator::kHeaderSize,
           "leading gaps coalesce");
    Expect(t.Allocate(8, 3) == nullptr && t.Allocate(8, 0) == nullptr,
           "invalid alignment rejected");
    Expect(t.Allocate(8, size_t(1) << 20) == nullptr, "impossible alignment");
}

void TestExhaustion() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 4096);
    Expect(t.Allocate(0) == nullptr, "zero size fails");
    Expect(t.Allocate(5000) == nullptr, "oversize fails");
    Expect(t.Allocate(~size_t(0)) == nullptr, "overflowing size fails");
    Expect(t.Allocate(~size_t(0) - 8, 64) == nullptr, "overflow with alignment");

    std::vector<void*> live;
    while (void* p = t.Allocate(100)) live.push_back(p);
    Expect(live.size() >= 30 && t.Check(), "arena filled with small blocks");
    Expect(t.Allocate(100) == nullptr, "full arena fails");
    t.Free(live[3]);
    Expect(t.Allocate(100) == live[3], "freed block is reused");
    for (void* p : live) t.Free(p);
    Expect(t.Check() && t.Used() == 0, "drained");
    // good-fit rounding may refuse a request within ~1/32 of the only free
    // block, so ask for slightly less than the whole arena
    Expect(t.Allocate(4096 - 2 * TlsfAllocator::kHeaderSize - 96) != nullptr,
           "nearly the whole arena can be one allocation");
}

void TestInvalidFrees() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 14);
    t.Free(nullptr);
    Expect(t.InvalidFreeCount() == 0, "free(nullptr) is a no-op");

    auto* a = static_cast<uint8_t*>(t.Allocate(200));
    auto* b = static_cast<uint8_t*>(t.Allocate(200));
    int local = 0;
    t.Free(&local);
    t.Free(a + 8);
    t.Free(a + 1);
    t.Free(a - 16);
    Expect(t.InvalidFreeCount() == 4, "outside, interior, misaligned rejected");
    Expect(t.Used() > 0 && t.AllocationCount() == 2 && t.Check(),
           "rejected frees change nothing");

    t.Free(a);
    t.Free(a);
    Expect(t.InvalidFreeCount() == 5 && t.Check(), "double free rejected");
    t.Free(b);
    t.Free(b);
    Expect(t.InvalidFreeCount() == 6 && t.Check(), "double free of merged block");
}

void TestFreeAll() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 14);
    void* first = t.Allocate(100);
    for (int i = 0; i < 50; i++) t.Allocate(10 + i);
    t.FreeAll();
    Expect(t.Check() && t.Used() == 0 && t.AllocationCount() == 0, "reset");
    Expect(t.Allocate(100) == first, "usable from scratch after FreeAll");
}

void TestBinBoundaries() {
    // sizes around the first- and second-level bin edges
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, size_t(1) << 22);
    std::vector<void*> live;
    bool ok = true;
    for (size_t size : {16u, 31u, 32u, 224u, 240u, 241u, 255u, 256u, 257u, 480u,
                        496u, 512u, 513u, 1000u, 1024u, 1025u, 4095u, 4096u,
                        4097u, 65535u, 65536u, 100000u}) {
        void* p = t.Allocate(size);
        ok = ok && p != nullptr && t.Check();
        if (p) {
            std::memset(p, 0x5A, size);
            live.push_back(p);
        }
    }
    Expect(ok, "boundary sizes allocate and keep the heap consistent");
    // free in an interleaved order
    for (size_t i = 0; i < live.size(); i += 2) t.Free(live[i]);
    Expect(t.Check(), "after freeing even entries");
    for (size_t i = 1; i < live.size(); i += 2) t.Free(live[i]);
    Expect(t.Check() && t.Used() == 0, "after freeing odd entries");
}

void TestGoodFit() {
    // a block that is large enough must be found even when only a bigger bin
    // holds one
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 16);
    void* a = t.Allocate(1000);
    void* sep1 = t.Allocate(16);
    void* b = t.Allocate(3000);
    void* sep2 = t.Allocate(16);
    t.Free(a);
    t.Free(b);
    void* p = t.Allocate(2500);
    Expect(p == b, "request served from the hole that fits");
    Expect(t.Check(), "consistent");
    t.Free(p);
    t.Free(sep1);
    t.Free(sep2);
    Expect(t.Check() && t.Used() == 0, "cleanup");
}

void TestRandomStress() {
    FailingMemoryManager mm;
    TlsfAllocator t(&mm, 1 << 20);
    std::mt19937 rng(777);
    struct Entry {
        uint8_t* p;
        size_t n;
        uint8_t tag;
    };
    std::vector<Entry> live;
    bool intact = true, aligned = true;

    for (int step = 0; step < 40000; step++) {
        const unsigned r = rng() % 100;
        if (r < 55 || live.empty()) {
            const unsigned kind = rng() % 10;
            const size_t n = kind < 7 ? 1 + rng() % 256
                                      : kind < 9 ? 1 + rng() % 4096
                                                 : 1 + rng() % 30000;
            const size_t align = (rng() % 8 == 0) ? size_t(32) << (rng() % 4) : 16;
            auto* p = static_cast<uint8_t*>(t.Allocate(n, align));
            if (!p) continue;  // fragmentation or full is allowed
            aligned = aligned && Aligned(p, align);
            const uint8_t tag = uint8_t(rng());
            std::memset(p, tag, n);
            live.push_back({p, n, tag});
        } else {
            const size_t i = rng() % live.size();
            const Entry e = live[i];
            for (size_t j = 0; j < e.n; j++) intact = intact && e.p[j] == e.tag;
            t.Free(e.p);
            live[i] = live.back();
            live.pop_back();
        }
        if (step % 2000 == 0 && !t.Check()) {
            Expect(false, "heap inconsistent during stress");
            return;
        }
    }
    Expect(intact && aligned, "contents and alignment survive random workload");
    Expect(t.Check() && t.AllocationCount() == live.size(), "state matches");
    for (auto& e : live) t.Free(e.p);
    Expect(t.Check() && t.Used() == 0 && t.InvalidFreeCount() == 0,
           "drained cleanly");
    Expect(t.LargestFreeBlock() == t.Capacity() - TlsfAllocator::kHeaderSize,
           "fully coalesced after stress");
}
}  // namespace

int main() {
    TestConfiguration();
    TestBasic();
    TestCoalescing();
    TestAlignment();
    TestExhaustion();
    TestInvalidFrees();
    TestFreeAll();
    TestBinBoundaries();
    TestGoodFit();
    TestRandomStress();

    if (failures == 0) std::cout << "All TLSF allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

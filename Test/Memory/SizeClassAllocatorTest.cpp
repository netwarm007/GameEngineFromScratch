#include <cstdint>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "MemoryManager.hpp"
#include "SizeClassAllocator.hpp"

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

void TestClassMapping() {
    Expect(SizeClassAllocator::ClassSize(1) == 16, "1 -> 16");
    Expect(SizeClassAllocator::ClassSize(16) == 16, "16 -> 16");
    Expect(SizeClassAllocator::ClassSize(17) == 32, "17 -> 32");
    Expect(SizeClassAllocator::ClassSize(128) == 128, "128 -> 128");
    Expect(SizeClassAllocator::ClassSize(129) == 160, "129 -> 160");
    Expect(SizeClassAllocator::ClassSize(257) == 320, "257 -> 320");
    Expect(SizeClassAllocator::ClassSize(512) == 512, "512 -> 512");
    Expect(SizeClassAllocator::ClassSize(513) == 0, "513 out of range");
    Expect(SizeClassAllocator::ClassSize(0) == 0, "0 out of range");

    bool ok = true;
    size_t waste = 0;
    for (size_t n = 1; n <= SizeClassAllocator::kMaxSize; n++) {
        const size_t c = SizeClassAllocator::ClassSize(n);
        ok = ok && c >= n && c % 16 == 0 && c - n < c / 4 + 16;
        waste = (std::max)(waste, c - n);
    }
    Expect(ok, "class covers request, 16-aligned, bounded internal waste");
    Expect(waste < 64, "worst-case waste below 64 bytes");

    bool increasing = true;
    for (size_t i = 1; i < SizeClassAllocator::kClassCount; i++) {
        increasing = increasing && SizeClassAllocator::ClassSizeAt(i) >
                                       SizeClassAllocator::ClassSizeAt(i - 1);
    }
    Expect(increasing && SizeClassAllocator::ClassSizeAt(15) == 512 &&
               SizeClassAllocator::ClassSizeAt(16) == 0,
           "class table sane");
}

void TestConfiguration() {
    FailingMemoryManager mm;
    SizeClassAllocator bad(&mm, 1000);
    Expect(!bad.IsConfigured(), "non power-of-two slab size rejected");
    Expect(bad.Allocate(16) == nullptr, "unconfigured allocate fails");
    bad.Free(reinterpret_cast<void*>(1));
    Expect(bad.InvalidFreeCount() == 1, "unconfigured free rejected");

    SizeClassAllocator tiny(&mm, 256);
    Expect(!tiny.IsConfigured(), "slab too small for the largest class");

    SizeClassAllocator ok(&mm);
    Expect(ok.IsConfigured() && ok.SlabCount() == 0, "default configuration");
    Expect(mm.PageCount() == 0, "slabs are created lazily");
}

void TestAllocateEverySize() {
    FailingMemoryManager mm;
    {
        SizeClassAllocator a(&mm);
        std::vector<std::pair<uint8_t*, size_t>> live;
        bool aligned = true, intact = true;
        for (size_t n = 1; n <= SizeClassAllocator::kMaxSize; n++) {
            auto* p = static_cast<uint8_t*>(a.Allocate(n));
            if (!p) {
                Expect(false, "allocation failed");
                return;
            }
            aligned = aligned && Aligned(p, 16);
            std::memset(p, int(n & 0xFF), n);
            live.emplace_back(p, n);
        }
        Expect(aligned, "all results 16-byte aligned");
        for (auto& [p, n] : live) {
            for (size_t j = 0; j < n; j++) intact = intact && p[j] == (n & 0xFF);
        }
        Expect(intact, "no overlap between size classes");
        Expect(a.LiveCount() == 512 && a.SlabCount() >= 16,
               "every class got at least one slab");
        const size_t slabs = a.SlabCount();
        for (auto& [p, n] : live) a.Free(p);
        Expect(a.LiveCount() == 0 && a.InvalidFreeCount() == 0, "all freed");
        Expect(a.Trim() == slabs && a.SlabCount() == 0 && mm.PageCount() == 0,
               "Trim returns every slab");
    }
}

void TestRejects() {
    FailingMemoryManager mm;
    SizeClassAllocator a(&mm);
    a.SetValidation(true);
    Expect(a.Allocate(0) == nullptr, "zero size fails");
    Expect(a.Allocate(513) == nullptr, "oversize fails");
    Expect(a.FailedAllocationCount() == 2, "failures counted");

    a.Free(nullptr);
    Expect(a.InvalidFreeCount() == 0, "free(nullptr) is a no-op");
    int local = 0;
    a.Free(&local);
    Expect(a.InvalidFreeCount() == 1, "foreign pointer rejected");

    void* p = a.Allocate(40);
    a.Free(static_cast<uint8_t*>(p) + 4);
    Expect(a.InvalidFreeCount() == 2 && a.LiveCount() == 1,
           "interior pointer rejected");
    a.Free(p);
    a.Free(p);
    Expect(a.InvalidFreeCount() == 3 && a.LiveCount() == 0, "double free");

    SizeClassAllocator b(&mm);
    b.SetValidation(true);
    void* q = b.Allocate(40);
    a.Free(q);
    Expect(a.InvalidFreeCount() == 4 && b.LiveCount() == 1,
           "pointer from another allocator rejected");

    // without validation the slab header still identifies the true owner
    a.SetValidation(false);
    void* r = a.Allocate(100);
    a.Free(r);
    Expect(a.LiveCount() == 0, "free without validation finds the class");
    b.Free(q);
}

void TestOutOfMemory() {
    FailingMemoryManager mm;
    SizeClassAllocator a(&mm);
    mm.fail = true;
    Expect(a.Allocate(64) == nullptr && a.FailedAllocationCount() == 1,
           "memory manager failure reported");
    mm.fail = false;
    void* p = a.Allocate(64);
    Expect(p != nullptr, "recovers");
    a.Free(p);
}

void TestFreeAll() {
    FailingMemoryManager mm;
    SizeClassAllocator a(&mm);
    for (int i = 0; i < 1000; i++) a.Allocate(8 + (i % 60) * 8);
    const size_t slabs = a.SlabCount();
    Expect(slabs >= 8 && a.LiveCount() == 1000, "populated");
    a.FreeAll();
    Expect(a.LiveCount() == 0 && a.SlabCount() == slabs, "FreeAll keeps slabs");
    Expect(a.Allocate(100) != nullptr && a.SlabCount() == slabs,
           "usable after FreeAll without new slabs");
}

void TestReuseAndLocality() {
    FailingMemoryManager mm;
    SizeClassAllocator a(&mm);
    void* p1 = a.Allocate(24);
    void* p2 = a.Allocate(24);
    Expect(p1 != p2, "distinct blocks");
    a.Free(p1);
    Expect(a.Allocate(20) == p1, "same class reuses the freed object");
    std::vector<void*> v;
    for (int i = 0; i < 10000; i++) v.push_back(a.Allocate(24));
    for (void* p : v) a.Free(p);
    const size_t pages = mm.PageCount();
    for (int i = 0; i < 10000; i++) v[i] = a.Allocate(24);
    Expect(mm.PageCount() == pages, "second round recycles slabs");
    for (void* p : v) a.Free(p);
}

void TestRandomStress() {
    FailingMemoryManager mm;
    SizeClassAllocator a(&mm);
    std::mt19937 rng(2024);
    struct Entry {
        uint8_t* p;
        size_t n;
        uint8_t tag;
    };
    std::vector<Entry> live;
    bool intact = true;

    for (int step = 0; step < 50000; step++) {
        const unsigned r = rng() % 100;
        if (r < 55 || live.empty()) {
            const size_t n = 1 + rng() % SizeClassAllocator::kMaxSize;
            auto* p = static_cast<uint8_t*>(a.Allocate(n));
            if (!p) {
                Expect(false, "stress allocation failed");
                return;
            }
            const uint8_t tag = uint8_t(rng());
            std::memset(p, tag, n);
            live.push_back({p, n, tag});
        } else if (r < 99) {
            const size_t i = rng() % live.size();
            const Entry e = live[i];
            for (size_t j = 0; j < e.n; j++) intact = intact && e.p[j] == e.tag;
            a.Free(e.p);
            live[i] = live.back();
            live.pop_back();
        } else {
            a.Trim();
        }
    }
    Expect(intact, "contents survive random workload");
    Expect(a.LiveCount() == live.size(), "live count matches");
    for (auto& e : live) a.Free(e.p);
    Expect(a.LiveCount() == 0 && a.InvalidFreeCount() == 0, "drained cleanly");
    a.Trim();
    Expect(a.SlabCount() == 0 && mm.PageCount() == 0, "all memory returned");
}
}  // namespace

int main() {
    TestClassMapping();
    TestConfiguration();
    TestAllocateEverySize();
    TestRejects();
    TestOutOfMemory();
    TestFreeAll();
    TestReuseAndLocality();
    TestRandomStress();

    if (failures == 0) std::cout << "All size class allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

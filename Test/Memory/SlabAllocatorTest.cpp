#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <random>
#include <set>
#include <vector>

#include "MemoryManager.hpp"
#include "SlabAllocator.hpp"

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

int g_constructed = 0;
int g_destroyed = 0;

struct Widget {
    int magic;
    int value;
    double extra;
};

void ConstructWidget(void* p) {
    auto* w = new (p) Widget{0xC0FFEE, 0, 1.5};
    (void)w;
    ++g_constructed;
}

void DestroyWidget(void* p) {
    static_cast<Widget*>(p)->~Widget();
    ++g_destroyed;
}

void TestConfiguration() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm);
    Expect(!s.IsConfigured(), "default unconfigured");
    Expect(s.Allocate(8) == nullptr && s.FailedAllocationCount() == 1,
           "unconfigured allocate fails");
    Expect(!s.Reset(0, 8, 4096), "zero object size rejected");
    Expect(!s.Reset(16, 3, 4096), "non power-of-two alignment rejected");
    Expect(!s.Reset(16, 8, 3000), "non power-of-two slab rejected");
    Expect(!s.Reset(16, 8192, 4096), "alignment above slab size rejected");
    Expect(!s.Reset(4096, 8, 4096), "slab too small for one object rejected");
    Expect(!s.Reset(~size_t(0), 16, 4096), "overflowing object size rejected");
    Expect(s.Reset(24, 16, 4096), "valid configuration");
    Expect(s.ObjectSize() == 32, "object size rounded to alignment");
    Expect(s.ObjectsPerSlab() >= 100 && s.ObjectsPerSlab() * 32 < 4096,
           "header leaves room, slab is mostly objects");
    Expect(s.Reset(1, 1, 256) && s.ObjectsPerSlab() >= 20, "tiny objects");
    Expect(s.Allocate(2) == nullptr, "request above object size fails");
    Expect(s.Allocate(0) == nullptr, "zero size fails");
}

void TestAllocateFree() {
    FailingMemoryManager mm;
    {
        SlabAllocator s(&mm, 48, 16, 4096);
        std::vector<void*> ptrs;
        std::set<void*> unique;
        const size_t n = s.ObjectsPerSlab() * 3 + 1;
        bool aligned = true;
        for (size_t i = 0; i < n; i++) {
            void* p = s.Allocate();
            if (!p) {
                Expect(false, "allocation failed");
                return;
            }
            aligned = aligned && Aligned(p, 16);
            std::memset(p, int(i & 0xFF), 48);
            ptrs.push_back(p);
            unique.insert(p);
        }
        Expect(aligned, "objects aligned");
        Expect(unique.size() == n, "objects are distinct");
        Expect(s.SlabCount() == 4 && s.LiveCount() == n, "slab and live counts");
        Expect(s.EmptySlabCount() == 0, "no empty slabs while full or partial");

        bool intact = true;
        for (size_t i = 0; i < n; i++) {
            auto* b = static_cast<uint8_t*>(ptrs[i]);
            for (size_t j = 0; j < 48; j++) intact = intact && b[j] == (i & 0xFF);
        }
        Expect(intact, "objects do not overlap or corrupt each other");

        for (void* p : ptrs) s.Free(p);
        Expect(s.LiveCount() == 0 && s.InvalidFreeCount() == 0, "all freed");
        Expect(s.EmptySlabCount() == 4 && s.SlabCount() == 4,
               "empty slabs retained");
        Expect(s.PeakLiveCount() == n, "peak tracked");
        const size_t pages = mm.PageCount();
        void* again = s.Allocate();
        Expect(again != nullptr && mm.PageCount() == pages,
               "empty slab reused, no new page");
        s.Free(again);
    }
    Expect(mm.PageCount() == 0, "destructor releases slabs");
}

void TestInvalidFrees() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm, 64, 16, 4096);
    SlabAllocator other(&mm, 64, 16, 4096);
    s.SetValidation(true);
    other.SetValidation(true);

    void* p = s.Allocate();
    void* q = other.Allocate();
    s.Free(nullptr);
    Expect(s.InvalidFreeCount() == 0, "free(nullptr) is a no-op");

    s.Free(static_cast<uint8_t*>(p) + 1);
    Expect(s.InvalidFreeCount() == 1, "interior pointer rejected");
    s.Free(q);
    Expect(s.InvalidFreeCount() == 2 && other.LiveCount() == 1,
           "foreign pointer rejected");
    int local = 0;
    s.Free(&local);
    Expect(s.InvalidFreeCount() == 3, "stack pointer rejected");

    s.Free(p);
    Expect(s.LiveCount() == 0, "valid free");
    s.Free(p);
    Expect(s.InvalidFreeCount() == 4 && s.LiveCount() == 0, "double free");
    other.Free(q);

    // without validation a foreign slab is still caught by its owner field
    s.SetValidation(false);
    void* r = s.Allocate();
    q = other.Allocate();
    s.Free(q);
    Expect(s.InvalidFreeCount() == 5 && other.LiveCount() == 1,
           "foreign slab rejected without validation");
    s.Free(static_cast<uint8_t*>(r) + 8);
    Expect(s.InvalidFreeCount() == 6, "misaligned pointer rejected");
    s.Free(r);
    other.Free(q);
}

void TestTrimAndFreeAll() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm, 128, 16, 4096);
    const size_t per = s.ObjectsPerSlab();
    std::vector<void*> ptrs;
    for (size_t i = 0; i < per * 3; i++) ptrs.push_back(s.Allocate());
    Expect(s.SlabCount() == 3, "three slabs");

    // empty the first and last slab fully, keep a few in the middle one
    for (size_t i = 0; i < per; i++) s.Free(ptrs[i]);
    for (size_t i = per * 2; i < per * 3; i++) s.Free(ptrs[i]);
    for (size_t i = per; i < per + per - 2; i++) s.Free(ptrs[i]);
    Expect(s.EmptySlabCount() == 2, "two slabs fully free");
    Expect(s.Trim() == 2 && s.SlabCount() == 1 && mm.PageCount() == 1,
           "Trim returns empty slabs to the memory manager");
    Expect(s.Trim() == 0, "nothing left to trim");
    Expect(s.LiveCount() == 2, "live objects untouched by Trim");

    s.FreeAll();
    Expect(s.LiveCount() == 0 && s.EmptySlabCount() == 1, "FreeAll empties");
    Expect(s.InvalidFreeCount() == 0, "no invalid frees");
    void* p = s.Allocate();
    Expect(p != nullptr && s.SlabCount() == 1, "usable after FreeAll");
    s.Free(ptrs[per]);  // stale pointer from before FreeAll
    s.Free(p);
}

void TestOutOfMemory() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm, 32, 16, 4096);
    mm.fail = true;
    Expect(s.Allocate() == nullptr && s.FailedAllocationCount() == 1,
           "page failure reported");
    mm.fail = false;
    void* p = s.Allocate();
    Expect(p != nullptr, "recovers after memory returns");
    mm.fail = true;
    bool ok = true;
    for (size_t i = 1; i < s.ObjectsPerSlab(); i++) ok = ok && s.Allocate();
    Expect(ok, "existing slab still serves objects");
    Expect(s.Allocate() == nullptr, "failure only when a new slab is needed");
}

void TestConstructedCache() {
    FailingMemoryManager mm;
    g_constructed = g_destroyed = 0;
    {
        SlabAllocator s(&mm, sizeof(Widget), alignof(Widget), 4096,
                        ConstructWidget, DestroyWidget);
        const size_t per = s.ObjectsPerSlab();
        auto* w = static_cast<Widget*>(s.Allocate());
        Expect(w && w->magic == 0xC0FFEE && w->extra == 1.5,
               "object handed out is constructed");
        Expect(g_constructed == int(per), "whole slab constructed once");
        w->value = 42;
        s.Free(w);
        auto* again = static_cast<Widget*>(s.Allocate());
        Expect(again == w && again->value == 42 && again->magic == 0xC0FFEE,
               "freed object keeps its constructed state");
        Expect(g_constructed == int(per) && g_destroyed == 0,
               "reuse does not construct or destroy");
        s.Free(again);
        Expect(s.Trim() == 1 && g_destroyed == int(per),
               "Trim destroys every object of the slab");
        auto* fresh = static_cast<Widget*>(s.Allocate());
        Expect(fresh && fresh->value == 0, "fresh slab reconstructs");
    }
    Expect(g_constructed == g_destroyed, "destructor destroys remaining slabs");
}

void TestLargeAlignment() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm, 100, 256, 8192);
    bool aligned = true;
    for (size_t i = 0; i < s.ObjectsPerSlab(); i++) {
        aligned = aligned && Aligned(s.Allocate(), 256);
    }
    Expect(aligned && s.SlabCount() == 1, "256-byte alignment honored");
}

void TestResetReleases() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm, 32, 16, 4096);
    s.Allocate();
    Expect(mm.PageCount() == 1, "one page in use");
    Expect(s.Reset(64, 16, 4096) && mm.PageCount() == 0 && s.LiveCount() == 0,
           "Reset releases slabs");
    Expect(!s.Reset(0, 16, 4096) && s.Allocate() == nullptr,
           "failed Reset leaves allocator unusable");
}

void TestRandomStress() {
    FailingMemoryManager mm;
    SlabAllocator s(&mm, 80, 16, 4096);
    std::mt19937 rng(12345);
    std::vector<std::pair<uint8_t*, uint8_t>> live;
    bool intact = true;

    for (int step = 0; step < 20000; step++) {
        const unsigned r = rng() % 100;
        if (r < 55 || live.empty()) {
            auto* p = static_cast<uint8_t*>(s.Allocate());
            if (!p) {
                Expect(false, "stress allocation failed");
                return;
            }
            const uint8_t tag = uint8_t(rng());
            std::memset(p, tag, 80);
            live.emplace_back(p, tag);
        } else if (r < 98) {
            const size_t i = rng() % live.size();
            auto [p, tag] = live[i];
            for (int j = 0; j < 80; j++) intact = intact && p[j] == tag;
            s.Free(p);
            live[i] = live.back();
            live.pop_back();
        } else {
            s.Trim();
        }
        if (s.LiveCount() != live.size()) {
            Expect(false, "live count drift");
            return;
        }
    }
    Expect(intact, "contents survive random alloc/free/trim");
    for (auto& [p, tag] : live) s.Free(p);
    Expect(s.LiveCount() == 0 && s.InvalidFreeCount() == 0, "drained cleanly");
    s.Trim();
    Expect(s.SlabCount() == 0 && mm.PageCount() == 0, "all slabs returned");
}
}  // namespace

int main() {
    TestConfiguration();
    TestAllocateFree();
    TestInvalidFrees();
    TestTrimAndFreeAll();
    TestOutOfMemory();
    TestConstructedCache();
    TestLargeAlignment();
    TestResetReleases();
    TestRandomStress();

    if (failures == 0) std::cout << "All slab allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

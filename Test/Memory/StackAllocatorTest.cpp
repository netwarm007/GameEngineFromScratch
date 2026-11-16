#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "MemoryManager.hpp"
#include "StackAllocator.hpp"

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
        StackAllocator s(&mm);
        Expect(!s.IsConfigured(), "default unconfigured");
        Expect(s.Allocate(8) == nullptr, "unconfigured allocate fails");
        Expect(!s.Initialize(0), "zero capacity rejected");
        mm.fail = true;
        Expect(!s.Initialize(512), "memory manager failure reported");
        mm.fail = false;
        Expect(s.Initialize(512) && s.Capacity() == 512, "initialize");
        s.Allocate(8);
        Expect(s.Initialize(1024) && s.Used() == 0 && mm.PageCount() == 1,
               "re-initialize releases old arena and resets state");
        s.Free(reinterpret_cast<void*>(1));
        Expect(s.InvalidFreeCount() == 1, "free on empty stack rejected");
    }
    Expect(mm.PageCount() == 0, "destructor frees page");
    {
        StackAllocator s(&mm, 64);
        Expect(s.IsConfigured(), "constructor configures");
        s.Release();
        Expect(!s.IsConfigured() && mm.PageCount() == 0, "Release frees page");
    }
    Expect(mm.InvalidFreeCount() == 0, "no invalid page frees");
}

void TestPushPop() {
    MemoryManager mm;
    StackAllocator s(&mm, 1024);

    Expect(s.Allocate(0) == nullptr, "zero size rejected");
    Expect(s.Allocate(8, 3) == nullptr, "bad alignment rejected");
    Expect(s.Used() == 0, "rejected requests consume nothing");

    auto* a = static_cast<uint8_t*>(s.Allocate(24));
    const size_t usedA = s.Used();
    auto* b = static_cast<uint8_t*>(s.Allocate(40, 64));
    auto* c = static_cast<uint8_t*>(s.Allocate(1, 1));
    Expect(a && b && c, "allocations succeed");
    Expect(Aligned(a, 16) && Aligned(b, 64), "alignment honored");
    Expect(b >= a + 24 && c >= b + 40, "monotonic layout");

    std::memset(a, 0xA1, 24);
    std::memset(b, 0xB2, 40);
    *c = 0xC3;
    Expect(a[23] == 0xA1 && b[0] == 0xB2, "no overlap");

    const size_t usedC = s.Used();
    s.Free(b);
    Expect(s.InvalidFreeCount() == 1 && s.Used() == usedC,
           "out-of-order free rejected");
    s.Free(nullptr);
    s.Free(a - 1);
    Expect(s.InvalidFreeCount() == 2, "foreign pointer rejected");

    s.Free(c);
    Expect(s.Used() < usedC, "top popped");
    s.Free(b);
    Expect(s.Used() == usedA, "second pop restores previous top");
    Expect(a[0] == 0xA1, "remaining allocation intact");
    s.Free(a);
    Expect(s.Used() == 0, "empty again");

    s.Free(a);
    Expect(s.InvalidFreeCount() == 3, "double pop rejected");
    Expect(s.Allocate(24) == a, "memory reused after pop");
}

void TestMarkers() {
    MemoryManager mm;
    StackAllocator s(&mm, 2048);

    auto* base = s.Allocate(32);
    auto m0 = s.GetMarker();
    Expect(m0.top == s.Used(), "marker captures top");

    auto* x = s.Allocate(100);
    auto m1 = s.GetMarker();
    s.Allocate(100);
    s.Allocate(100, 128);
    const size_t deep = s.Used();

    Expect(s.FreeToMarker(m1), "rewind to inner marker");
    Expect(s.Used() == m1.top, "top restored");
    s.Free(x);
    Expect(s.InvalidFreeCount() == 0 && s.Used() == m0.top,
           "pop after marker rewind uses restored chain");

    auto m2 = s.GetMarker();
    s.Allocate(500);
    Expect(s.FreeToMarker(m2) && s.Used() == m2.top, "rewind again");
    Expect(s.FreeToMarker(m2), "rewind to current top is fine");

    {
        StackAllocator::Marker future{deep + 64, StackAllocator::Marker{}.last};
        Expect(!s.FreeToMarker(future), "marker above top rejected");
        Expect(s.InvalidFreeCount() == 1 && s.Used() == m2.top,
               "rejected marker changes nothing");
    }

    s.Free(base);
    Expect(s.Used() == 0, "base popped");

    StackAllocator::Marker empty = s.GetMarker();
    s.Allocate(10);
    s.Allocate(10);
    Expect(s.FreeToMarker(empty) && s.Used() == 0, "rewind to empty marker");
    Expect(s.Allocate(10) != nullptr, "usable after rewinding to empty");
}

void TestStaleMarkers() {
    MemoryManager mm;
    StackAllocator s(&mm, 2048);

    s.Allocate(32);
    auto* x = s.Allocate(64);
    auto stale = s.GetMarker();
    s.Free(x);
    // same offset, different allocation: the old marker must not match
    s.Allocate(64);
    const size_t top = s.Used();
    Expect(!s.FreeToMarker(stale), "marker of unwound allocation rejected");
    Expect(s.InvalidFreeCount() == 1 && s.Used() == top,
           "stale marker changes nothing");

    auto fresh = s.GetMarker();
    s.Allocate(8);
    Expect(s.FreeToMarker(fresh), "fresh marker accepted");

    // marker survives pops above it
    auto older = s.GetMarker();
    auto* y = s.Allocate(16);
    s.Free(y);
    Expect(s.FreeToMarker(older), "marker valid after pops above it");

    // markers do not survive FreeAll or re-initialization
    s.Allocate(16);
    auto beforeReset = s.GetMarker();
    s.FreeAll();
    s.Allocate(16);
    s.Allocate(16);
    Expect(!s.FreeToMarker(beforeReset), "marker rejected after FreeAll");
    auto beforeInit = s.GetMarker();
    s.Initialize(2048);
    s.Allocate(16);
    s.Allocate(16);
    s.Allocate(16);
    Expect(!s.FreeToMarker(beforeInit), "marker rejected after Initialize");

    StackAllocator::Marker forged;
    forged.top = 100;
    forged.last = 64;
    forged.serial = 1;
    s.FreeAll();
    Expect(!s.FreeToMarker(forged), "forged marker rejected");
    Expect(s.FreeToMarker(s.GetMarker()), "marker of empty stack accepted");
}

void TestExhaustionAndFreeAll() {
    MemoryManager mm;
    StackAllocator s(&mm, 128);

    Expect(s.Allocate(1000, 1) == nullptr, "oversized fails");
    Expect(s.Allocate((std::numeric_limits<size_t>::max)(), 1) == nullptr,
           "SIZE_MAX fails without overflow");
    Expect(s.Allocate((std::numeric_limits<size_t>::max)() - 4, 16) == nullptr,
           "near SIZE_MAX fails");
    Expect(s.FailedAllocationCount() == 3, "failures counted");

    // header + payload must fit
    Expect(s.Allocate(128, 1) == nullptr, "header overhead counted");
    void* fill = s.Allocate(128 - StackAllocator::kHeaderSize, 1);
    Expect(fill != nullptr, "largest possible allocation fits");
    Expect(s.Remaining() == 0, "stack full");
    Expect(s.Allocate(1, 1) == nullptr, "full stack fails");
    Expect(s.PeakUsed() == 128, "peak tracked");

    s.FreeAll();
    Expect(s.Used() == 0 && s.PeakUsed() == 128, "FreeAll keeps peak");
    Expect(s.Allocate(16) != nullptr, "usable after FreeAll");
    s.Free(nullptr);
    s.FreeAll();
    s.FreeAll();
    Expect(s.InvalidFreeCount() == 0, "repeated FreeAll is harmless");
}

void TestPaddingBeyondCapacity() {
    MemoryManager mm;
    StackAllocator s(&mm, 256);
    // a 4096 boundary may or may not fall inside the arena, depending on its
    // address; either the result is aligned or nothing is consumed
    void* p = s.Allocate(8, 4096);
    if (p) {
        Expect(Aligned(p, 4096), "padded allocation is aligned");
    } else {
        Expect(s.Used() == 0 && s.FailedAllocationCount() == 1,
               "failed padded allocation consumes nothing");
    }
}

void TestRandomStress() {
    MemoryManager mm;
    StackAllocator s(&mm, 1 << 16);
    std::mt19937 rng(777);

    struct Entry {
        uint8_t* p;
        size_t size;
        uint8_t tag;
        StackAllocator::Marker before;
    };
    std::vector<Entry> live;
    bool intact = true;

    auto verify = [&](const Entry& e) {
        for (size_t i = 0; i < e.size; i++) {
            if (e.p[i] != e.tag) intact = false;
        }
    };

    for (int step = 0; step < 30000; step++) {
        unsigned r = rng() % 100;
        if (r < 55 || live.empty()) {
            size_t size = 1 + rng() % 200;
            size_t align = size_t(1) << (rng() % 7);
            auto before = s.GetMarker();
            auto* p = static_cast<uint8_t*>(s.Allocate(size, align));
            if (!p) {
                // full: unwind everything
                for (auto& e : live) verify(e);
                live.clear();
                s.FreeAll();
                continue;
            }
            if (!Aligned(p, align)) intact = false;
            auto tag = static_cast<uint8_t>(rng());
            std::memset(p, tag, size);
            live.push_back({p, size, tag, before});
        } else if (r < 90) {
            verify(live.back());
            s.Free(live.back().p);
            live.pop_back();
        } else {
            size_t keep = rng() % live.size();
            for (auto& e : live) verify(e);
            Entry cut = live[keep];
            live.resize(keep);
            if (!s.FreeToMarker(cut.before)) intact = false;
        }
        if (!live.empty() && s.Used() < live.back().size) intact = false;
    }
    Expect(intact, "random push/pop/rewind keeps data and chain intact");
    Expect(s.InvalidFreeCount() == 0, "no invalid operations");
}
}  // namespace

int main() {
    TestConfiguration();
    TestPushPop();
    TestMarkers();
    TestStaleMarkers();
    TestExhaustionAndFreeAll();
    TestPaddingBeyondCapacity();
    TestRandomStress();

    if (failures == 0) std::cout << "All stack allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

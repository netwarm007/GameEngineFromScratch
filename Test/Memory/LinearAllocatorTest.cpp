#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

#include "LinearAllocator.hpp"
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
    bool fail = false;
};

void TestConfiguration() {
    FailingMemoryManager mm;
    {
        LinearAllocator a(&mm);
        Expect(!a.IsConfigured() && a.Capacity() == 0, "default unconfigured");
        Expect(a.Allocate(8) == nullptr, "unconfigured allocate fails");
        Expect(a.FailedAllocationCount() == 1, "failure counted");
        Expect(!a.Initialize(0), "zero capacity rejected");
        mm.fail = true;
        Expect(!a.Initialize(1024), "memory manager failure reported");
        Expect(!a.IsConfigured(), "stays unconfigured on failure");
        mm.fail = false;
        Expect(a.Initialize(1024) && a.IsConfigured(), "initialize succeeds");
        Expect(a.Capacity() == 1024 && a.Used() == 0 && a.Remaining() == 1024,
               "capacity reported");
        Expect(mm.PageCount() == 1, "one backing page");
        Expect(Aligned(a.Allocate(1), 16), "default alignment is 16");

        a.Allocate(8);
        Expect(a.Initialize(2048), "re-initialize");
        Expect(mm.PageCount() == 1, "old arena released on re-initialize");
        Expect(a.Used() == 0 && a.PeakUsed() == 0 && a.AllocationCount() == 0,
               "state reset on re-initialize");
        mm.fail = true;
        Expect(!a.Initialize(64) && !a.IsConfigured() && mm.PageCount() == 0,
               "failed re-initialize releases old arena");
        mm.fail = false;
    }
    {
        LinearAllocator a(&mm, 256);
        Expect(a.IsConfigured(), "constructor configures");
        a.Release();
        Expect(!a.IsConfigured() && mm.PageCount() == 0, "Release frees page");
        a.Release();
    }
    Expect(mm.PageCount() == 0, "destructor frees page");
    Expect(mm.InvalidFreeCount() == 0, "no invalid page frees");
}

void TestAllocation() {
    MemoryManager mm;
    LinearAllocator a(&mm, 256);

    Expect(a.Allocate(0) == nullptr, "zero size rejected");
    Expect(a.Allocate(8, 0) == nullptr, "zero alignment rejected");
    Expect(a.Allocate(8, 24) == nullptr, "non power-of-two alignment rejected");
    Expect(a.Used() == 0 && a.FailedAllocationCount() == 3,
           "rejected requests consume nothing");

    auto* p1 = static_cast<uint8_t*>(a.Allocate(10, 1));
    auto* p2 = static_cast<uint8_t*>(a.Allocate(10, 1));
    Expect(p1 && p2 && p2 == p1 + 10, "byte-aligned allocations are packed");
    Expect(Aligned(p1, 64), "arena starts cache-line aligned");

    auto* p3 = static_cast<uint8_t*>(a.Allocate(4, 32));
    Expect(Aligned(p3, 32) && p3 >= p2 + 10, "alignment padding applied");
    Expect(a.Used() == static_cast<size_t>(p3 - p1) + 4, "used includes padding");
    Expect(a.AllocationCount() == 3, "allocation count");

    std::memset(p1, 1, 10);
    std::memset(p2, 2, 10);
    std::memset(p3, 3, 4);
    Expect(p1[9] == 1 && p2[0] == 2, "allocations do not overlap");

    a.Free(p1);
    Expect(a.Used() == static_cast<size_t>(p3 - p1) + 4,
           "Free is a no-op");
    a.Free(nullptr);

    Expect(Aligned(a.Allocate(1, 128), 128), "alignment above arena start");
}

void TestExhaustion() {
    MemoryManager mm;
    LinearAllocator a(&mm, 128);

    Expect(a.Allocate(129, 1) == nullptr, "larger than capacity fails");
    Expect(a.Allocate((std::numeric_limits<size_t>::max)(), 1) == nullptr,
           "huge size fails without overflow");
    Expect(a.Allocate((std::numeric_limits<size_t>::max)() - 8, 16) == nullptr,
           "size near SIZE_MAX fails");
    Expect(a.Allocate(128, 1) != nullptr, "exact fit succeeds");
    Expect(a.Remaining() == 0 && a.Used() == 128, "full");
    Expect(a.Allocate(1, 1) == nullptr, "full arena fails");

    const size_t failures_before = a.FailedAllocationCount();
    Expect(a.Allocate(1) == nullptr &&
               a.FailedAllocationCount() == failures_before + 1,
           "failures counted");

    a.FreeAll();
    Expect(a.Used() == 0 && a.Allocate(128, 1) != nullptr,
           "FreeAll restores full capacity");

    a.FreeAll();
    Expect(a.Allocate(100, 1) != nullptr, "setup");
    // the arena is 64-byte aligned, so aligning to 64 needs padding to 128
    Expect(a.Allocate(1, 64) == nullptr, "padding beyond capacity fails");
    Expect(a.Used() == 100, "failed padded allocation consumes nothing");
}

void TestFreeAllReuse() {
    MemoryManager mm;
    LinearAllocator a(&mm, 512);

    void* first = a.Allocate(64);
    a.Allocate(64);
    Expect(a.PeakUsed() >= 128, "peak tracked");
    const size_t peak = a.PeakUsed();

    a.FreeAll();
    Expect(a.Used() == 0 && a.AllocationCount() == 0, "reset");
    Expect(a.PeakUsed() == peak, "peak survives FreeAll");
    Expect(a.Allocate(64) == first, "memory reused from the start");
    Expect(mm.PageCount() == 1, "backing memory kept");

#if defined(_DEBUG)
    auto* q = static_cast<uint8_t*>(a.Allocate(16));
    Expect(q[0] == IAllocator::PATTERN_ALLOC, "debug alloc pattern");
    a.FreeAll();
    Expect(q[0] == IAllocator::PATTERN_FREE, "debug free pattern");
#endif
}

void TestInterface() {
    MemoryManager mm;
    LinearAllocator linear(&mm, 128);
    IAllocator* a = &linear;
    void* p = a->Allocate(16);
    Expect(p != nullptr, "allocate through interface");
    a->Free(p);
    a->FreeAll();
    Expect(linear.Used() == 0, "FreeAll through interface");
}

void TestOddAlignmentSweep() {
    MemoryManager mm;
    LinearAllocator a(&mm, 1 << 16);
    std::set<uintptr_t> seen;
    uint8_t* prevEnd = nullptr;
    bool ordered = true;
    for (int i = 0; i < 200; i++) {
        size_t align = size_t(1) << (i % 8);
        size_t size = 1 + (i * 7) % 33;
        auto* p = static_cast<uint8_t*>(a.Allocate(size, align));
        if (!p) {
            Expect(false, "sweep allocation failed");
            return;
        }
        if (!Aligned(p, align)) ordered = false;
        if (prevEnd && p < prevEnd) ordered = false;
        prevEnd = p + size;
        seen.insert(reinterpret_cast<uintptr_t>(p));
    }
    Expect(ordered, "aligned, monotonically increasing, non-overlapping");
    Expect(seen.size() == 200, "unique addresses");
}
}  // namespace

int main() {
    TestConfiguration();
    TestAllocation();
    TestExhaustion();
    TestFreeAllReuse();
    TestInterface();
    TestOddAlignmentSweep();

    if (failures == 0) std::cout << "All linear allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

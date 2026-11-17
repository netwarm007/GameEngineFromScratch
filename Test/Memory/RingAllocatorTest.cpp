#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <deque>

#include "MemoryManager.hpp"
#include "RingAllocator.hpp"

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
        RingAllocator r(&mm);
        Expect(!r.IsConfigured(), "default unconfigured");
        Expect(r.Allocate(8) == nullptr, "unconfigured allocate fails");
        Expect(!r.Retire(1) && r.InvalidRetireCount() == 1,
               "retire beyond head rejected when unconfigured");
        Expect(!r.Initialize(0), "zero capacity rejected");
        mm.fail = true;
        Expect(!r.Initialize(256), "memory manager failure reported");
        mm.fail = false;
        Expect(r.Initialize(256) && r.Capacity() == 256 && r.Remaining() == 256,
               "initialize");
        r.Allocate(32);
        Expect(r.Initialize(512) && r.Used() == 0 && r.GetHead() == 0 &&
                   mm.PageCount() == 1,
               "re-initialize resets and releases old arena");
    }
    Expect(mm.PageCount() == 0, "destructor releases arena");
    RingAllocator r(&mm, 64);
    r.Release();
    Expect(!r.IsConfigured() && mm.PageCount() == 0, "Release");
    Expect(mm.InvalidFreeCount() == 0, "no invalid page frees");
}

void TestBasicAllocation() {
    MemoryManager mm;
    RingAllocator r(&mm, 256);

    Expect(r.Allocate(0) == nullptr, "zero size rejected");
    Expect(r.Allocate(8, 5) == nullptr, "bad alignment rejected");
    Expect(r.Allocate(257, 1) == nullptr, "larger than capacity rejected");
    Expect(r.Used() == 0 && r.FailedAllocationCount() == 3,
           "rejected requests consume nothing");

    auto* a = static_cast<uint8_t*>(r.Allocate(10, 1));
    auto* b = static_cast<uint8_t*>(r.Allocate(10, 1));
    Expect(b == a + 10 && r.Used() == 20, "packed allocations");
    auto* c = static_cast<uint8_t*>(r.Allocate(4, 32));
    Expect(Aligned(c, 32) && r.Used() == static_cast<size_t>(c - a) + 4,
           "alignment padding counts as used");
    Expect(Aligned(a, 64), "buffer start is cache-line aligned");
    Expect(r.PeakUsed() == r.Used(), "peak tracked");
}

void TestWrapAndFull() {
    MemoryManager mm;
    RingAllocator r(&mm, 128);

    auto* a = static_cast<uint8_t*>(r.Allocate(100, 1));
    const uint64_t afterA = r.GetHead();
    std::memset(a, 0xAA, 100);

    // 28 bytes remain at the end; a 40-byte block would need to wrap
    Expect(r.Allocate(40, 1) == nullptr, "full ring fails without overwrite");
    Expect(a[0] == 0xAA && a[99] == 0xAA, "live data untouched");
    Expect(r.Used() == 100, "failed allocation consumes nothing");

    Expect(r.Retire(afterA), "retire first batch");
    Expect(r.Used() == 0, "all reclaimed");
    auto* b = static_cast<uint8_t*>(r.Allocate(40, 1));
    Expect(b == a, "wrap restarts at the beginning (tail wasted)");
    Expect(r.Used() == 28 + 40, "skipped tail counts as used");

    // the 28-byte tail is waste; a 20-byte block fits right after b
    auto* c = static_cast<uint8_t*>(r.Allocate(20, 1));
    Expect(c == b + 40, "allocation continues after wrapped block");
    Expect(r.Remaining() == 128 - 28 - 60, "remaining after wrap");

    Expect(r.Allocate(41, 1) == nullptr, "one byte too many");
    Expect(r.Allocate(40, 1) != nullptr && r.Remaining() == 0,
           "exact remainder fits");
}

void TestExactFit() {
    MemoryManager mm;
    RingAllocator r(&mm, 128);
    Expect(r.Allocate(128, 1) != nullptr, "full-capacity block fits");
    Expect(r.Remaining() == 0 && r.Allocate(1, 1) == nullptr, "now full");
    Expect(r.Retire(r.GetHead()), "retire all");
    // head is at a multiple of the capacity; the next block starts at 0
    auto* p = static_cast<uint8_t*>(r.Allocate(128, 1));
    Expect(p != nullptr && r.Used() == 128, "exact fit again after cycling");
}

void TestRetire() {
    MemoryManager mm;
    RingAllocator r(&mm, 256);

    r.Allocate(32, 1);
    const uint64_t m1 = r.GetHead();
    r.Allocate(32, 1);
    const uint64_t m2 = r.GetHead();
    Expect(m2 > m1, "head advances");

    Expect(!r.Retire(m2 + 1), "position beyond head rejected");
    Expect(r.InvalidRetireCount() == 1 && r.Used() == 64,
           "rejected retire changes nothing");
    Expect(r.Retire(m1) && r.Used() == 32, "partial retire");
    Expect(r.Retire(m1), "retiring the same position again is harmless");
    Expect(!r.Retire(m1 - 1) && r.InvalidRetireCount() == 2,
           "retiring backwards rejected");
    Expect(r.Retire(m2) && r.Used() == 0, "retire everything");

    r.Allocate(16, 1);
    r.FreeAll();
    Expect(r.Used() == 0, "FreeAll reclaims everything");
    const uint64_t head = r.GetHead();
    Expect(!r.Retire(head - 1), "FreeAll does not rewind positions");
    IAllocator* a = &r;
    a->Free(a->Allocate(8));
    Expect(r.Used() == 8, "Free through interface is a no-op");
}

void TestFencedFrames() {
    // Models a producer writing per-frame data while a consumer lags behind.
    MemoryManager mm;
    RingAllocator r(&mm, 1024);
    struct Frame {
        uint64_t end;
        uint8_t* p;
        size_t size;
        uint8_t tag;
    };
    std::deque<Frame> inflight;
    std::mt19937 rng(99);
    bool intact = true;
    bool aligned = true;

    for (int i = 0; i < 20000; i++) {
        size_t size = 1 + rng() % 150;
        size_t align = size_t(1) << (rng() % 6);
        auto* p = static_cast<uint8_t*>(r.Allocate(size, align));
        if (!p) {
            // consumer catches up on the oldest frames until there is room
            while (!p && !inflight.empty()) {
                auto& f = inflight.front();
                for (size_t j = 0; j < f.size; j++) {
                    if (f.p[j] != f.tag) intact = false;
                }
                if (!r.Retire(f.end)) intact = false;
                inflight.pop_front();
                p = static_cast<uint8_t*>(r.Allocate(size, align));
            }
            if (!p) {
                intact = false;
                break;
            }
        }
        if (!Aligned(p, align)) aligned = false;
        auto tag = static_cast<uint8_t>(rng());
        std::memset(p, tag, size);
        inflight.push_back({r.GetHead(), p, size, tag});
    }
    Expect(intact, "in-flight data is never overwritten");
    Expect(aligned, "alignment honored across wraps");
    Expect(r.InvalidRetireCount() == 0, "no invalid retires");
    Expect(r.Used() <= r.Capacity(), "used never exceeds capacity");
}

void TestLargeAlignment() {
    MemoryManager mm;
    RingAllocator r(&mm, 1024);
    auto* p = r.Allocate(8, 256);
    Expect(p != nullptr && Aligned(p, 256), "alignment above arena start");
    // whether a 4096 boundary falls inside the buffer depends on its address
    void* q = r.Allocate(8, 4096);
    Expect(q == nullptr || Aligned(q, 4096), "large alignment honored or refused");
}

void TestAddressOverflow() {
    MemoryManager mm;
    RingAllocator r(&mm, 1024);
    Expect(r.Allocate(8, size_t(1) << (sizeof(size_t) * 8 - 1)) == nullptr &&
               r.Used() == 0,
           "alignment past the address space is refused");
}
}  // namespace

int main() {
    TestConfiguration();
    TestBasicAllocation();
    TestWrapAndFull();
    TestExactFit();
    TestRetire();
    TestFencedFrames();
    TestLargeAlignment();
    TestAddressOverflow();

    if (failures == 0) std::cout << "All ring allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

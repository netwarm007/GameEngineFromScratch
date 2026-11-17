#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include "GpuSubAllocator.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

constexpr uint64_t kMax = (std::numeric_limits<uint64_t>::max)();

void TestConfiguration() {
    GpuSubAllocator zeroCap(0);
    GpuSubAllocator zeroGran(1024, 0);
    GpuAllocation a;
    Expect(!zeroCap.IsConfigured() && !zeroCap.Allocate(1, 1, a) &&
               !a.IsValid(),
           "zero capacity rejected");
    Expect(!zeroGran.IsConfigured() && !zeroGran.Allocate(1, 1, a),
           "zero granularity rejected");
    Expect(zeroCap.Check() && zeroGran.Check(), "unconfigured state is valid");

    GpuSubAllocator ok(1 << 20);
    Expect(ok.IsConfigured() && ok.FreeBytes() == (1 << 20) &&
               ok.LargestFreeRange() == (1 << 20) && ok.Check(),
           "fresh heap is one free range");
}

void TestBasic() {
    GpuSubAllocator heap(1 << 20);
    GpuAllocation a, b;
    Expect(heap.Allocate(100, 1, a) && a.offset == 0 && a.size == 100 &&
               a.IsValid(),
           "first allocation at offset 0");
    Expect(heap.UsedBytes() == 256, "size rounded to granularity");
    Expect(heap.Allocate(256, 1, b) && b.offset == 256, "second follows");
    Expect(heap.Check() && heap.LiveCount() == 2, "consistent");
    Expect(heap.Free(a) && heap.Free(b), "frees succeed");
    Expect(heap.UsedBytes() == 0 && heap.FreeRangeCount() == 1 &&
               heap.LargestFreeRange() == (1 << 20) && heap.Check(),
           "everything coalesced");
    Expect(heap.PeakUsedBytes() == 512, "peak recorded");
}

void TestAlignment() {
    GpuSubAllocator heap(1 << 20);
    GpuAllocation small, big, fill;
    Expect(heap.Allocate(256, 1, small), "small");
    Expect(heap.Allocate(1024, 4096, big) && big.offset % 4096 == 0 &&
               big.offset == 4096,
           "aligned allocation skips ahead");
    Expect(heap.Check() && heap.FreeRangeCount() == 2,
           "leading gap stays free");
    // the gap [256, 4096) is reusable
    Expect(heap.Allocate(3840, 1, fill) && fill.offset == 256,
           "gap reused by exact fit");
    Expect(heap.Check(), "consistent after gap reuse");

    GpuAllocation huge;
    Expect(heap.Allocate(256, uint64_t(1) << 63, huge) == false ||
               huge.offset % (uint64_t(1) << 63) == 0,
           "huge alignment either fails or is honoured");
    GpuAllocation c;
    Expect(!heap.Allocate(16, 0, c) && !heap.Allocate(16, 3, c) &&
               !heap.Allocate(16, 48, c),
           "non power-of-two alignment rejected");
    Expect(heap.Check(), "failures leave the heap intact");
}

void TestLimits() {
    GpuSubAllocator heap(4096);
    GpuAllocation a;
    Expect(!heap.Allocate(0, 1, a), "zero size");
    Expect(!heap.Allocate(kMax, 1, a), "size overflowing the rounding");
    Expect(!heap.Allocate(kMax - 100, 256, a), "huge size");
    Expect(!heap.Allocate(4097, 1, a), "larger than capacity");
    Expect(heap.Allocate(4096, 1, a) && heap.FreeBytes() == 0, "exact fill");
    GpuAllocation b;
    Expect(!heap.Allocate(1, 1, b), "full heap");
    Expect(heap.FailedAllocationCount() == 5, "failures counted");

    GpuSubAllocator big(kMax, 1);
    GpuAllocation x, y, z, w;
    const uint64_t q = uint64_t(1) << 62;
    Expect(big.Allocate(q, 1, x) && big.Allocate(q, 1, y) &&
               big.Allocate(q, 1, z) && !big.Allocate(q, 1, w) &&
               big.Allocate(q - 1, 1, w),
           "capacity near 2^64 does not overflow");
    Expect(big.UsedBytes() == kMax && big.FreeBytes() == 0 && big.Check(),
           "bookkeeping at the limit");
}

void TestCoalescingOrders() {
    for (int order = 0; order < 6; order++) {
        GpuSubAllocator heap(4096, 256);
        GpuAllocation a[4];
        for (auto& x : a) Expect(heap.Allocate(1024, 1, x), "alloc");
        static const int perms[6][4] = {{0, 1, 2, 3}, {3, 2, 1, 0},
                                        {1, 3, 0, 2}, {2, 0, 3, 1},
                                        {0, 2, 1, 3}, {3, 0, 2, 1}};
        for (int i : perms[order]) {
            Expect(heap.Free(a[i]) && heap.Check(), "free keeps invariants");
        }
        Expect(heap.FreeRangeCount() == 1 && heap.LargestFreeRange() == 4096,
               "fully coalesced regardless of order");
    }
}

void TestBestFit() {
    GpuSubAllocator heap(16 * 256, 256);
    GpuAllocation a[8];
    // layout: 8 blocks with sizes 1,2,1,3,1,... separated by live blocks
    const uint64_t sizes[] = {2, 1, 4, 1, 3, 1, 1, 1};
    for (int i = 0; i < 8; i++) Expect(heap.Allocate(sizes[i] * 256, 1, a[i]), "a");
    // free holes of 2, 4 and 3 granules
    heap.Free(a[0]);
    heap.Free(a[2]);
    heap.Free(a[4]);
    GpuAllocation r;
    Expect(heap.Allocate(3 * 256, 1, r) && r.offset == a[4].offset,
           "best fit picks the 3-granule hole");
    Expect(heap.Allocate(2 * 256, 1, r) && r.offset == a[0].offset,
           "best fit picks the 2-granule hole");
    Expect(heap.Check(), "consistent");
}

void TestInvalidFrees() {
    GpuSubAllocator heap(8192);
    GpuAllocation a, b;
    heap.Allocate(512, 1, a);
    GpuAllocation stale = a;
    Expect(heap.Free(a) && !heap.Free(a), "double free rejected");
    heap.Allocate(512, 1, b);
    Expect(b.offset == stale.offset && b.id != stale.id,
           "offset reused with a new id");
    Expect(!heap.Free(stale), "stale handle rejected after reuse");

    GpuAllocation forged = b;
    forged.size = 1;
    Expect(!heap.Free(forged), "wrong size rejected");
    forged = b;
    forged.offset += 256;
    Expect(!heap.Free(forged), "wrong offset rejected");
    Expect(!heap.Free(GpuAllocation{}), "invalid handle rejected");
    Expect(heap.InvalidFreeCount() == 5, "invalid frees counted");
    Expect(heap.Free(b) && heap.Check() && heap.UsedBytes() == 0,
           "valid handle still frees");
}

void TestDeferred() {
    GpuSubAllocator heap(4096, 256);
    GpuAllocation a, b, c;
    heap.Allocate(256, 1, a);
    heap.Allocate(256, 1, b);
    heap.Allocate(256, 1, c);

    Expect(heap.FreeDeferred(a, 5) && heap.FreeDeferred(b, 3),
           "deferred frees accepted out of order");
    Expect(heap.UsedBytes() == 768 && heap.PendingBytes() == 512 &&
               heap.PendingCount() == 2,
           "memory stays reserved");
    Expect(!heap.Free(a) && !heap.FreeDeferred(a, 9),
           "pending handle cannot be freed again");
    GpuAllocation d;
    Expect(heap.Allocate(4096 - 768, 1, d), "remaining space usable");
    GpuAllocation e;
    Expect(!heap.Allocate(256, 1, e), "pending memory is not reused");
    heap.Free(d);

    Expect(heap.Retire(2) == 0, "nothing completed below the fence");
    Expect(heap.Retire(3) == 1 && heap.UsedBytes() == 512 &&
               heap.PendingBytes() == 256,
           "fence 3 releases b only");
    Expect(heap.Retire(10) == 1 && heap.PendingCount() == 0 &&
               heap.UsedBytes() == 256,
           "fence 10 releases a");
    Expect(heap.Check(), "consistent");

    // fences at or below the completed value release immediately
    Expect(heap.FreeDeferred(c, 7) && heap.UsedBytes() == 0 &&
               heap.PendingCount() == 0,
           "already completed fence frees immediately");
    Expect(heap.Check() && heap.FreeRangeCount() == 1, "fully free");
}

void TestReset() {
    GpuSubAllocator heap(4096, 256);
    GpuAllocation a, b;
    heap.Allocate(512, 1, a);
    heap.Allocate(512, 1, b);
    heap.FreeDeferred(b, 1);
    heap.Reset();
    Expect(heap.UsedBytes() == 0 && heap.LiveCount() == 0 &&
               heap.PendingCount() == 0 && heap.Check(),
           "reset clears everything");
    Expect(!heap.Free(a) && !heap.Free(b), "old handles invalid after reset");
    GpuAllocation c;
    Expect(heap.Allocate(4096, 1, c) && c.id != a.id, "heap fully usable");
}

void TestRandomized() {
    std::mt19937_64 rng(12345);
    constexpr uint64_t kCapacity = 1 << 22;
    GpuSubAllocator heap(kCapacity, 64);
    std::vector<GpuAllocation> live;
    std::vector<std::pair<GpuAllocation, uint64_t>> deferred;
    uint64_t frame = 0;
    bool overlap = false;

    auto verify = [&] {
        std::map<uint64_t, uint64_t> ranges;
        auto add = [&](const GpuAllocation& a) {
            ranges.emplace(a.offset, a.size);
        };
        for (auto& a : live) add(a);
        for (auto& d : deferred) add(d.first);
        uint64_t end = 0;
        for (auto& [off, size] : ranges) {
            if (off < end) overlap = true;
            end = off + size;
        }
    };

    for (int i = 0; i < 20000; i++) {
        const unsigned op = rng() % 10;
        if (op < 5) {
            GpuAllocation a;
            const uint64_t size = 1 + rng() % 20000;
            const uint64_t align = uint64_t(1) << (rng() % 13);
            if (heap.Allocate(size, align, a)) {
                if (a.offset % align != 0 || a.offset + size > kCapacity) {
                    overlap = true;
                }
                live.push_back(a);
            }
        } else if (op < 7 && !live.empty()) {
            const size_t k = rng() % live.size();
            heap.Free(live[k]);
            live[k] = live.back();
            live.pop_back();
        } else if (op < 9 && !live.empty()) {
            const size_t k = rng() % live.size();
            const uint64_t fence = frame + 1 + rng() % 3;
            heap.FreeDeferred(live[k], fence);
            deferred.emplace_back(live[k], fence);
            live[k] = live.back();
            live.pop_back();
        } else {
            ++frame;
            heap.Retire(frame);
            std::erase_if(deferred, [&](auto& d) { return d.second <= frame; });
        }
        if (i % 500 == 0) {
            verify();
            if (!heap.Check()) {
                Expect(false, "invariants broken mid-run");
                return;
            }
        }
    }
    Expect(!overlap, "live and pending ranges never overlap");
    for (auto& a : live) heap.Free(a);
    heap.Retire(frame + 10);
    Expect(heap.UsedBytes() == 0 && heap.FreeRangeCount() == 1 &&
               heap.LargestFreeRange() == kCapacity && heap.Check(),
           "heap returns to a single free range");
    Expect(heap.InvalidFreeCount() == 0, "no invalid frees in valid workload");
}
}  // namespace

int main() {
    TestConfiguration();
    TestBasic();
    TestAlignment();
    TestLimits();
    TestCoalescingOrders();
    TestBestFit();
    TestInvalidFrees();
    TestDeferred();
    TestReset();
    TestRandomized();

    if (failures == 0) std::cout << "All GPU sub-allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

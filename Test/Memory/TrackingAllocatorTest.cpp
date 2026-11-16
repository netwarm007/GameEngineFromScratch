#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "BlockAllocator.hpp"
#include "LinearAllocator.hpp"
#include "MemoryManager.hpp"
#include "TlsfAllocator.hpp"
#include "TrackingAllocator.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void TestStats() {
    MemoryManager mm;
    TlsfAllocator tlsf(&mm, 1 << 16);
    TrackingAllocator t(&mm, tlsf);

    void* a = t.Allocate(100);
    void* b = t.Allocate(200);
    auto s = t.GetStats();
    Expect(s.allocations == 2 && s.liveCount == 2 && s.liveBytes == 300 &&
               s.totalBytes == 300 && s.peakLiveBytes == 300,
           "counts after two allocations");
    t.Free(a);
    s = t.GetStats();
    Expect(s.frees == 1 && s.liveCount == 1 && s.liveBytes == 200 &&
               s.peakLiveBytes == 300 && s.peakLiveCount == 2,
           "peaks survive frees");
    t.Free(b);
    Expect(t.GetStats().liveBytes == 0 && tlsf.Used() == 0,
           "everything returned to the wrapped allocator");

    Expect(t.Allocate(0) == nullptr && t.Allocate(~size_t(0)) == nullptr &&
               t.Allocate(1 << 20) == nullptr,
           "zero, overflow and exhaustion fail");
    Expect(t.GetStats().failedAllocations == 3, "failures counted");
    t.Free(nullptr);
    Expect(t.GetStats().invalidFrees == 0, "free(nullptr) is a no-op");
}

void TestInvalidFreesNotForwarded() {
    MemoryManager mm;
    BlockAllocator pool(&mm, 64, 4096, 16);
    pool.SetValidation(false);  // the inner allocator would not notice
    TrackingAllocator t(&mm, pool);

    void* p = t.Allocate(32);
    t.Free(p);
    t.Free(p);
    int local = 0;
    t.Free(&local);
    Expect(t.GetStats().invalidFrees == 2, "double and foreign free rejected");
    Expect(pool.FreeBlockCount() == pool.BlockCount() &&
               pool.InvalidFreeCount() == 0,
           "bad frees never reach the wrapped allocator");
    Expect(t.Allocate(60) == nullptr, "guard bytes count against block size");
}

void TestGuards() {
    MemoryManager mm;
    TlsfAllocator tlsf(&mm, 1 << 16);
    TrackingAllocator t(&mm, tlsf);

    auto* a = static_cast<uint8_t*>(t.Allocate(40));
    auto* b = static_cast<uint8_t*>(t.Allocate(40));
    std::memset(a, 1, 40);
    Expect(t.CheckGuards() == 0, "clean blocks pass");

    a[40] = 0;  // one byte past the end
    Expect(t.CheckGuards() == 1, "overrun detected while live");
    t.Free(a);
    Expect(t.GetStats().guardViolations == 1, "overrun counted on free");
    Expect(tlsf.Check(), "corrupted block still released consistently");

    b[-1] = 0;  // underrun is outside the guard and not detected; no crash
    b[39] = 7;  // last valid byte does not trip the guard
    Expect(t.CheckGuards() == 0, "writes inside the block are fine");
    t.Free(b);
    Expect(t.GetStats().guardViolations == 1, "no false positive");
}

void TestLeakReport() {
    MemoryManager mm;
    LinearAllocator arena(&mm, 4096);
    std::string captured;
    {
        TrackingAllocator t(&mm, arena, "frame arena");
        t.SetLeakHandler([&](const std::string& r) { captured = r; });
        void* a = t.AllocateTagged(10, "mesh");
        void* b = t.Allocate(20);
        t.AllocateTagged(30, "texture");
        t.Free(b);
        (void)a;

        auto leaks = t.Leaks();
        Expect(leaks.size() == 2 && leaks[0].size == 10 &&
                   std::string(leaks[0].tag) == "mesh" && leaks[1].size == 30 &&
                   leaks[0].sequence < leaks[1].sequence,
               "leaks listed in allocation order with tags");
        const std::string report = t.LeakReport();
        Expect(report.find("frame arena: 2 leaked block(s), 40 byte(s)") == 0,
               "report header");
        Expect(report.find("[mesh]") != std::string::npos &&
                   report.find("[texture]") != std::string::npos,
               "report lists tags");
        Expect(captured.empty(), "handler not called before destruction");
    }
    Expect(captured.find("2 leaked block(s)") != std::string::npos,
           "destructor reports leaks to the handler");

    captured.clear();
    {
        TrackingAllocator t(&mm, arena);
        t.SetLeakHandler([&](const std::string& r) { captured = r; });
        void* p = t.Allocate(8);
        t.Free(p);
    }
    Expect(captured.empty(), "no report without leaks");
}

void TestFreeAll() {
    MemoryManager mm;
    TlsfAllocator tlsf(&mm, 1 << 14);
    TrackingAllocator t(&mm, tlsf);
    void* stale = t.Allocate(64);
    t.Allocate(64);
    t.FreeAll();
    Expect(t.Leaks().empty() && t.GetStats().liveCount == 0 &&
               tlsf.Used() == 0,
           "FreeAll clears records and the wrapped allocator");
    t.Free(stale);
    Expect(t.GetStats().invalidFrees == 1, "pre-FreeAll pointer rejected");
    Expect(t.GetStats().allocations == 2, "cumulative stats survive FreeAll");
}

void TestThreadSafety() {
    MemoryManager mm;
    TlsfAllocator tlsf(&mm, 1 << 22);  // not thread-safe by itself
    TrackingAllocator t(&mm, tlsf);

    std::vector<std::thread> threads;
    for (int id = 0; id < 4; id++) {
        threads.emplace_back([&t, id] {
            std::vector<void*> mine;
            for (int i = 0; i < 5000; i++) {
                if (mine.size() < 50 && i % 3 != 2) {
                    void* p = t.Allocate(16 + (i * 7 + id) % 200);
                    if (p) mine.push_back(p);
                } else if (!mine.empty()) {
                    t.Free(mine.back());
                    mine.pop_back();
                }
            }
            for (void* p : mine) t.Free(p);
        });
    }
    for (auto& th : threads) th.join();

    const auto s = t.GetStats();
    Expect(s.liveCount == 0 && s.allocations == s.frees &&
               s.invalidFrees == 0 && s.guardViolations == 0,
           "balanced under concurrency");
    Expect(tlsf.Check() && tlsf.Used() == 0, "wrapped allocator intact");
}
}  // namespace

int main() {
    TestStats();
    TestInvalidFreesNotForwarded();
    TestGuards();
    TestLeakReport();
    TestFreeAll();
    TestThreadSafety();

    if (failures == 0) std::cout << "All tracking allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

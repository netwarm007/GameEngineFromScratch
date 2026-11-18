#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "MemoryManager.hpp"
#include "RoutingAllocator.hpp"

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

constexpr size_t kCapacity = 1 << 20;

void TestInvalidRequests() {
    MemoryManager mm;
    RoutingAllocator a(&mm, kCapacity);
    Expect(a.IsConfigured(), "configured");
    Expect(a.Allocate(0) == nullptr, "zero size rejected");
    Expect(a.Allocate(16, 0) == nullptr, "zero alignment rejected");
    Expect(a.Allocate(16, 24) == nullptr, "non power of two alignment rejected");
    Expect(a.FailedAllocationCount() == 0, "invalid requests are not failures");
    a.Free(nullptr);

    RoutingAllocator bad(&mm, 8);
    Expect(!bad.IsConfigured() && bad.Allocate(16) == nullptr,
           "unconfigured allocator allocates nothing");
}

void TestRoutingBoundary() {
    MemoryManager mm;
    RoutingAllocator a(&mm, kCapacity);

    void* small = a.Allocate(RoutingAllocator::kSmallLimit);
    Expect(small != nullptr && Aligned(small, 16), "512 bytes allocated");
    Expect(a.SmallRouteCount() == 1 && a.LargeRouteCount() == 0,
           "512 bytes goes to the small path");
    a.FlushThreadCache();
    Expect(a.SmallLiveCount() == 1 && a.LargeAllocationCount() == 0,
           "small object counted by the thread cache");

    void* large = a.Allocate(RoutingAllocator::kSmallLimit + 1);
    Expect(large != nullptr && Aligned(large, 16), "513 bytes allocated");
    Expect(a.SmallRouteCount() == 1 && a.LargeRouteCount() == 1,
           "513 bytes goes to TLSF");
    Expect(a.LargeAllocationCount() == 1, "large object counted by TLSF");
    Expect(a.LargeUsed() > 513, "large usage includes the block");

    std::memset(small, 0xAB, RoutingAllocator::kSmallLimit);
    std::memset(large, 0xCD, RoutingAllocator::kSmallLimit + 1);

    a.Free(small);
    a.Free(large);
    a.FlushThreadCache();
    Expect(a.SmallLiveCount() == 0 && a.LargeAllocationCount() == 0,
           "both paths return their objects");
    Expect(a.InvalidFreeCount() == 0, "no invalid frees");
}

void TestAlignment() {
    MemoryManager mm;
    RoutingAllocator a(&mm, kCapacity);

    for (size_t align : {1u, 2u, 8u, 16u}) {
        void* p = a.Allocate(24, align);
        Expect(p != nullptr && Aligned(p, 16), "small result is 16-aligned");
        a.Free(p);
    }
    Expect(a.LargeRouteCount() == 0, "small alignments stay on the small path");

    std::vector<std::pair<void*, size_t>> blocks;
    for (size_t align : {32u, 64u, 256u, 4096u}) {
        for (size_t size : {1u, 100u, 512u, 513u, 5000u}) {
            void* p = a.Allocate(size, align);
            Expect(p != nullptr && Aligned(p, align),
                   "over-aligned request honored");
            if (p) std::memset(p, 0x5A, size);
            blocks.emplace_back(p, size);
        }
    }
    a.FlushThreadCache();
    Expect(a.SmallLiveCount() == 0, "over-aligned requests bypass the cache");
    for (auto& [p, size] : blocks) a.Free(p);
    Expect(a.LargeAllocationCount() == 0 && a.LargeUsed() == 0,
           "over-aligned blocks all freed");
    Expect(a.InvalidFreeCount() == 0, "no invalid frees");
}

void TestForeignAndDoubleFree() {
    MemoryManager mm;
    RoutingAllocator a(&mm, kCapacity);

    alignas(16) uint8_t local[64];
    a.Free(local);
    Expect(a.InvalidFreeCount() >= 1, "foreign pointer rejected");

    void* big = a.Allocate(4096);
    a.Free(big);
    const size_t before = a.InvalidFreeCount();
    a.Free(big);
    Expect(a.InvalidFreeCount() == before + 1, "large double free detected");
    Expect(a.LargeAllocationCount() == 0, "double free changed nothing");
}

void TestExhaustionAndFallback() {
    FailingMemoryManager mm;
    RoutingAllocator a(&mm, 64 * 1024);

    // The slab source fails; small requests fall back to TLSF.
    mm.fail = true;
    void* p = a.Allocate(64);
    Expect(p != nullptr, "small request falls back to TLSF");
    Expect(a.LargeRouteCount() == 1 && a.SmallRouteCount() == 0,
           "fallback counted as a large route");
    a.Free(p);
    Expect(a.LargeAllocationCount() == 0, "fallback block freed through TLSF");
    mm.fail = false;

    std::vector<void*> held;
    while (void* q = a.Allocate(4096)) held.push_back(q);
    Expect(!held.empty() && a.FailedAllocationCount() == 1,
           "exhaustion fails once and is counted");
    for (void* q : held) a.Free(q);
    Expect(a.LargeAllocationCount() == 0, "arena recovered");
    Expect(a.Allocate(4096) != nullptr, "allocation works again");
}

void TestFreeAll() {
    MemoryManager mm;
    RoutingAllocator a(&mm, kCapacity);
    for (int i = 0; i < 100; ++i) {
        a.Allocate(32);
        a.Allocate(2000);
    }
    a.FreeAll();
    Expect(a.SmallLiveCount() == 0 && a.LargeAllocationCount() == 0,
           "FreeAll drops everything");
    Expect(a.Allocate(32) != nullptr && a.Allocate(2000) != nullptr,
           "usable after FreeAll");
}

void TestStress() {
    MemoryManager mm;
    RoutingAllocator a(&mm, 8 << 20);
    constexpr int kThreads = 4;
    constexpr int kIterations = 20000;
    std::atomic<int> corrupt{0};

    // Blocks handed to the next thread are freed by a different thread.
    std::vector<std::vector<std::pair<void*, size_t>>> handoff(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(t + 1);
            std::vector<std::pair<void*, size_t>> live;
            auto stamp = [&](void* p, size_t n) {
                std::memset(p, static_cast<int>(n & 0xFF), n);
            };
            auto check = [&](void* p, size_t n) {
                const auto* b = static_cast<const uint8_t*>(p);
                for (size_t i = 0; i < n; ++i) {
                    if (b[i] != (n & 0xFF)) {
                        ++corrupt;
                        return;
                    }
                }
            };
            for (int i = 0; i < kIterations; ++i) {
                if (live.size() < 64 && (rng() & 1)) {
                    const size_t size = (rng() % 4 == 0) ? 513 + rng() % 3000
                                                         : 1 + rng() % 512;
                    const size_t align = (rng() % 8 == 0) ? 64 : 16;
                    void* p = a.Allocate(size, align);
                    if (p == nullptr) continue;
                    if (!Aligned(p, align)) ++corrupt;
                    stamp(p, size);
                    live.emplace_back(p, size);
                } else if (!live.empty()) {
                    const size_t k = rng() % live.size();
                    check(live[k].first, live[k].second);
                    a.Free(live[k].first);
                    live[k] = live.back();
                    live.pop_back();
                }
            }
            handoff[t] = std::move(live);
            a.FlushThreadCache();
        });
    }
    for (auto& t : threads) t.join();

    std::vector<std::thread> freers;
    for (int t = 0; t < kThreads; ++t) {
        freers.emplace_back([&, t] {
            for (auto& [p, n] : handoff[(t + 1) % kThreads]) a.Free(p);
            a.FlushThreadCache();
        });
    }
    for (auto& t : freers) t.join();

    Expect(corrupt == 0, "no corruption or misalignment under contention");
    Expect(a.SmallLiveCount() == 0, "all small objects returned");
    Expect(a.LargeAllocationCount() == 0 && a.LargeUsed() == 0,
           "all large blocks returned");
    Expect(a.InvalidFreeCount() == 0, "no invalid frees");
}
}  // namespace

int main() {
    TestInvalidRequests();
    TestRoutingBoundary();
    TestAlignment();
    TestForeignAndDoubleFree();
    TestExhaustionAndFallback();
    TestFreeAll();
    TestStress();

    if (failures == 0) std::cout << "All routing allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

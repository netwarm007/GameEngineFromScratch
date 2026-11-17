#include <atomic>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "MemoryManager.hpp"
#include "ThreadCacheAllocator.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
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
    MemoryManager mm;
    ThreadCacheAllocator badCache(&mm, 16 * 1024, 0);
    ThreadCacheAllocator badSlab(&mm, 1000);
    Expect(!badCache.IsConfigured() && badCache.Allocate(16) == nullptr,
           "zero cache bound rejected");
    Expect(!badSlab.IsConfigured() && badSlab.Allocate(16) == nullptr,
           "bad slab size rejected");
    ThreadCacheAllocator ok(&mm);
    Expect(ok.IsConfigured(), "default configuration");
    Expect(ok.Allocate(0) == nullptr && ok.Allocate(513) == nullptr &&
               ok.Allocate(~size_t(0)) == nullptr,
           "out-of-range sizes fail");
    Expect(ok.FailedAllocationCount() == 3, "failures counted");
    ok.Free(nullptr);
    Expect(ok.InvalidFreeCount() == 0, "free(nullptr) is a no-op");
}

void TestBasicAndReuse() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm);
    std::vector<std::pair<uint8_t*, size_t>> live;
    bool aligned = true, intact = true;
    for (size_t n = 1; n <= ThreadCacheAllocator::kMaxSize; n += 7) {
        auto* p = static_cast<uint8_t*>(a.Allocate(n));
        if (!p) {
            Expect(false, "allocation failed");
            return;
        }
        aligned = aligned && reinterpret_cast<uintptr_t>(p) % 16 == 0;
        std::memset(p, int(n & 0xFF), n);
        live.emplace_back(p, n);
    }
    for (auto& [p, n] : live) {
        for (size_t j = 0; j < n; j++) intact = intact && p[j] == (n & 0xFF);
    }
    Expect(aligned && intact, "aligned, non-overlapping objects");
    for (auto& [p, n] : live) a.Free(p);

    void* p = a.Allocate(40);
    a.Free(p);
    Expect(a.Allocate(40) == p, "freed object is reused from the cache");
    Expect(a.InvalidFreeCount() == 0, "no invalid frees");
}

void TestBatching() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm, 16 * 1024, 64);
    void* p = a.Allocate(100);
    Expect(a.RefillCount() == 1 && a.BackingLiveCount() == 32 &&
               a.CachedCount() == 31,
           "first allocation refills one batch");
    for (int i = 0; i < 31; i++) a.Allocate(100);
    Expect(a.RefillCount() == 1, "batch served without more refills");
    a.Allocate(100);
    Expect(a.RefillCount() == 2, "empty cache refills again");

    a.FlushThreadCache();
    Expect(a.CachedCount() == 0 && a.BackingLiveCount() == 33,
           "flush returns only cached objects");
    (void)p;
}

void TestCacheBound() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm, 16 * 1024, 8);
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; i++) ptrs.push_back(a.Allocate(64));
    bool bounded = true;
    for (void* p : ptrs) {
        a.Free(p);
        bounded = bounded && a.CachedCount() <= 8;
    }
    Expect(bounded, "cache never exceeds its bound");
    Expect(a.FlushCount() > 0, "overflow flushed to the backing allocator");
    a.FlushThreadCache();
    Expect(a.BackingLiveCount() == 0, "all objects returned");
    Expect(a.Trim() >= 1 && a.BackingSlabCount() == 0, "Trim releases slabs");
}

void TestCrossThreadFree() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm);
    std::vector<void*> ptrs;
    std::thread producer([&] {
        for (int i = 0; i < 200; i++) {
            void* p = a.Allocate(48);
            std::memset(p, 0xAB, 48);
            ptrs.push_back(p);
        }
    });
    producer.join();
    for (void* p : ptrs) a.Free(p);
    a.FlushThreadCache();
    Expect(a.BackingLiveCount() == 0 && a.InvalidFreeCount() == 0,
           "objects freed by another thread are accepted");
}

void TestThreadExitFlush() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm);
    std::thread t([&] {
        std::vector<void*> v;
        for (int i = 0; i < 100; i++) v.push_back(a.Allocate(32));
        for (void* p : v) a.Free(p);
        Expect(a.CachedCount() > 0, "thread holds cached objects");
    });
    t.join();
    Expect(a.BackingLiveCount() == 0, "thread exit returns its cache");
}

void TestAllocatorLifetime() {
    FailingMemoryManager mm;
    // thread caches for destroyed allocators must never be used or flushed
    for (int i = 0; i < 50; i++) {
        ThreadCacheAllocator a(&mm);
        void* p = a.Allocate(64);
        a.Free(p);
        std::thread([&] { a.Free(a.Allocate(64)); }).join();
    }
    Expect(mm.PageCount() == 0, "all slabs released");

    // a thread outliving its allocator exits cleanly
    std::atomic<bool> allocated{false}, destroyed{false};
    std::thread t;
    {
        auto a = std::make_unique<ThreadCacheAllocator>(&mm);
        t = std::thread([&] {
            a->Free(a->Allocate(64));
            allocated = true;
            while (!destroyed) std::this_thread::yield();
        });
        while (!allocated) std::this_thread::yield();
        a.reset();
        destroyed = true;
    }
    t.join();
    Expect(mm.PageCount() == 0, "outliving thread did not touch freed state");
}

void TestFreeAll() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm);
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; i++) ptrs.push_back(a.Allocate(80));
    for (int i = 0; i < 50; i++) a.Free(ptrs[i]);
    a.FreeAll();
    Expect(a.CachedCount() == 0 && a.BackingLiveCount() == 0,
           "FreeAll drops caches and objects");
    std::vector<void*> again;
    for (int i = 0; i < 200; i++) {
        void* p = a.Allocate(80);
        if (!p) break;
        std::memset(p, 0x5A, 80);
        again.push_back(p);
    }
    bool unique = true;
    std::sort(again.begin(), again.end());
    for (size_t i = 1; i < again.size(); i++) {
        unique = unique && again[i] != again[i - 1];
    }
    Expect(again.size() == 200 && unique, "no stale cached pointer reused");
    for (void* p : again) a.Free(p);
    a.FlushThreadCache();
    Expect(a.BackingLiveCount() == 0, "balanced after FreeAll");
}

void TestForeignPointer() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm), b(&mm);
    void* p = b.Allocate(64);
    a.Free(p);
    Expect(a.InvalidFreeCount() == 1, "pointer from another allocator rejected");
    Expect(a.CachedCount() == 0, "rejected pointer is not cached");
    b.Free(p);
}

void TestExhaustion() {
    FailingMemoryManager mm;
    ThreadCacheAllocator a(&mm);
    mm.fail = true;
    Expect(a.Allocate(64) == nullptr && a.FailedAllocationCount() == 1,
           "backing failure surfaces as nullptr");
    mm.fail = false;
    Expect(a.Allocate(64) != nullptr, "recovers once memory is available");
}

void TestStress() {
    MemoryManager mm;
    ThreadCacheAllocator a(&mm, 16 * 1024, 32);
    constexpr int kThreads = 8;
    std::mutex handoffMutex;
    std::vector<std::pair<uint8_t*, size_t>> handoff;
    std::atomic<int> corrupt{0};

    auto check = [&](uint8_t* p, size_t n) {
        const uint8_t tag = p[0];
        for (size_t i = 0; i < n; i++) {
            if (p[i] != tag) {
                ++corrupt;
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int id = 0; id < kThreads; id++) {
        threads.emplace_back([&, id] {
            std::mt19937 rng(id + 1);
            std::vector<std::pair<uint8_t*, size_t>> mine;
            for (int i = 0; i < 20000; i++) {
                const unsigned op = rng() % 8;
                if (op < 4 && mine.size() < 200) {
                    const size_t n = 1 + rng() % 512;
                    auto* p = static_cast<uint8_t*>(a.Allocate(n));
                    if (!p) continue;
                    std::memset(p, id * 16 + int(rng() % 16), n);
                    mine.emplace_back(p, n);
                } else if (op < 7 && !mine.empty()) {
                    const size_t k = rng() % mine.size();
                    check(mine[k].first, mine[k].second);
                    a.Free(mine[k].first);
                    mine[k] = mine.back();
                    mine.pop_back();
                } else if (!mine.empty()) {
                    std::lock_guard<std::mutex> lock(handoffMutex);
                    handoff.push_back(mine.back());
                    mine.pop_back();
                }
                if (op == 7) {
                    std::pair<uint8_t*, size_t> got{nullptr, 0};
                    {
                        std::lock_guard<std::mutex> lock(handoffMutex);
                        if (!handoff.empty()) {
                            got = handoff.back();
                            handoff.pop_back();
                        }
                    }
                    if (got.first) {
                        check(got.first, got.second);
                        a.Free(got.first);
                    }
                }
            }
            for (auto& [p, n] : mine) {
                check(p, n);
                a.Free(p);
            }
        });
    }
    for (auto& t : threads) t.join();
    for (auto& [p, n] : handoff) {
        check(p, n);
        a.Free(p);
    }
    a.FlushThreadCache();

    Expect(corrupt == 0, "no object was handed out twice");
    Expect(a.BackingLiveCount() == 0, "every object returned");
    Expect(a.InvalidFreeCount() == 0, "no invalid frees");
}
}  // namespace

int main() {
    TestConfiguration();
    TestBasicAndReuse();
    TestBatching();
    TestCacheBound();
    TestCrossThreadFree();
    TestThreadExitFlush();
    TestAllocatorLifetime();
    TestFreeAll();
    TestForeignPointer();
    TestExhaustion();
    TestStress();

    if (failures == 0) std::cout << "All thread cache allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

// Manual micro-benchmark, not registered as a test. Build at -O2 -DNDEBUG:
//   g++ -std=c++20 -O2 -DNDEBUG -pthread -IFramework/Common -IFramework/Interface
//     -IFramework/Manager Test/Memory/AllocatorBenchmark.cpp
//     Framework/Manager/{MemoryManager,MemorySystem,RoutingAllocator,
//     ThreadCacheAllocator,SizeClassAllocator,SlabAllocator,TlsfAllocator,
//     FrameAllocator,LinearAllocator,TrackingAllocator,EngineAllocation}.cpp
//     -o bench
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <memory_resource>
#include <thread>
#include <vector>

#include "EngineAllocation.hpp"
#include "MemorySystem.hpp"

using namespace My;
using Clock = std::chrono::steady_clock;

namespace {
constexpr size_t kSizes[] = {32, 64, 128, 480};

struct Heap {
    static void* Alloc(size_t n) { return ::operator new(n); }
    static void Free(void* p) { ::operator delete(p); }
};

// Allocates a batch, frees it in reverse order, repeats; ns per alloc+free.
template <typename A, typename F>
double Churn(A alloc, F free_, int iterations, size_t batch) {
    std::vector<void*> live(batch);
    const auto t0 = Clock::now();
    for (int it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < batch; ++i)
            live[i] = alloc(kSizes[i % std::size(kSizes)]);
        for (size_t i = batch; i-- > 0;) free_(live[i]);
    }
    const auto ns =
        std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    return ns / (double(iterations) * double(batch));
}

template <typename A, typename F>
double Threaded(A alloc, F free_, int threads, int iterations) {
    std::vector<double> out(threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] { out[t] = Churn(alloc, free_, iterations, 256); });
    for (auto& th : pool) th.join();
    return *std::max_element(out.begin(), out.end());
}

struct Probe {
    int v[4];
};
}  // namespace

int main() {
    const int iters = 4000;
    MemorySystem plain({});
    MemorySystem::Config tc;
    tc.tracking = true;
    tc.reportHandler = [](const std::string&) {};
    MemorySystem tracked(tc);

    struct Row {
        const char* name;
        IAllocator* a;
    } rows[] = {{"default heap", nullptr},
                {"RoutingAllocator", &plain.Allocator()},
                {"Routing+Tracking", &tracked.Allocator()}};

    std::printf("ns per alloc+free, 256-block churn, sizes 32..480 B (best of 5)\n");
    std::printf("%-18s %8s %8s %8s\n", "", "1 thr", "4 thr", "8 thr");
    for (auto& r : rows) {
        auto al = [&](size_t n) { return r.a ? r.a->Allocate(n) : Heap::Alloc(n); };
        auto fr = [&](void* p) { r.a ? r.a->Free(p) : Heap::Free(p); };
        double v[3];
        const int tcs[] = {1, 4, 8};
        for (int k = 0; k < 3; ++k) {
            double best = 1e30;
            for (int i = 0; i < 5; ++i)
                best = std::min(best, Threaded(al, fr, tcs[k], iters));
            v[k] = best;
        }
        std::printf("%-18s %8.1f %8.1f %8.1f\n", r.name, v[0], v[1], v[2]);
    }

    // Large blocks go through the mutex-guarded TLSF path.
    std::printf("\nlarge 4 KiB blocks, ns per alloc+free (best of 5)\n");
    for (int threads : {1, 4}) {
        auto fr = [&](void* p) { plain.Allocator().Free(p); };
        auto al = [&](size_t) { return plain.Allocator().Allocate(4096); };
        auto hal = [&](size_t) { return Heap::Alloc(4096); };
        double a = 1e30, b = 1e30;
        for (int i = 0; i < 5; ++i) {
            a = std::min(a, Threaded(hal, Heap::Free, threads, 1000));
            b = std::min(b, Threaded(al, fr, threads, 1000));
        }
        std::printf("%d thr: default heap %.1f, RoutingAllocator %.1f\n", threads, a, b);
    }

    // Frame scratch: pmr vector of ints over the arena vs the default heap.
    std::printf("\nper-frame scratch (pmr vector, 64 pushes x 100 vectors per frame), ns per frame\n");
    auto frame = [&](std::pmr::memory_resource* res, bool arena) {
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = Clock::now();
            for (int f = 0; f < 2000; ++f) {
                if (arena) plain.BeginFrame();
                for (int v = 0; v < 100; ++v) {
                    std::pmr::vector<int> vec(res);
                    for (int i = 0; i < 64; ++i) vec.push_back(i);
                }
            }
            best = std::min(best, std::chrono::duration<double, std::nano>(
                                      Clock::now() - t0).count() / 2000);
        }
        return best;
    };
    std::printf("default heap %.0f, frame arena %.0f\n",
                frame(std::pmr::new_delete_resource(), false),
                frame(&plain.FrameResource(), true));

    // Hook path used by the pilot: shared_ptr of a small object.
    std::printf("\nMakeEngineShared<16 B> create+destroy, ns (hook off / on)\n");
    auto shared = [&] {
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = Clock::now();
            for (int i = 0; i < 200000; ++i) {
                auto p = MakeEngineShared<Probe>();
                asm volatile("" : : "r"(p.get()) : "memory");
            }
            best = std::min(best, std::chrono::duration<double, std::nano>(
                                      Clock::now() - t0).count() / 200000);
        }
        return best;
    };
    SetEngineAllocator(nullptr);
    const double off = shared();
    SetEngineAllocator(&plain.Allocator());
    const double on = shared();
    SetEngineAllocator(&tracked.Allocator());
    const double onTracked = shared();
    SetEngineAllocator(nullptr);
    std::printf("off %.1f, on %.1f, on+tracking %.1f\n", off, on, onTracked);
    return 0;
}

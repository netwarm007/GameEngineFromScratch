#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "EngineAllocation.hpp"
#include "MemorySystem.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

struct Probe {
    explicit Probe(int& counter) : c(counter) { ++c; }
    ~Probe() { --c; }
    int& c;
};

class FailingAllocator : public IAllocator {
   public:
    FailingAllocator() : IAllocator(nullptr) {}
    void* Allocate(size_t) override { return nullptr; }
    void Free(void*) override {}
    void FreeAll() override {}
};

void TestNoHook() {
    SetEngineAllocator(nullptr);
    void* p = EngineAllocate(100);
    Expect(p && reinterpret_cast<uintptr_t>(p) % 16 == 0,
           "default heap block is 16-byte aligned");
    Expect(EngineAllocationCount() == 0, "default heap is not counted");
    EngineFree(p);
    EngineFree(nullptr);
    int live = 0;
    {
        auto sp = MakeEngineShared<Probe>(live);
        Expect(live == 1, "object constructed without hook");
    }
    Expect(live == 0, "object destroyed without hook");
}

void TestHooked() {
    MemorySystem sys({});
    SetEngineAllocator(&sys.Allocator());
    void* p = EngineAllocate(100);
    Expect(p && reinterpret_cast<uintptr_t>(p) % 16 == 0,
           "hooked block is 16-byte aligned");
    Expect(EngineAllocationCount() == 1, "hooked block is counted");
    EngineFree(p);
    Expect(EngineAllocationCount() == 0, "freed block is uncounted");

    int live = 0;
    {
        std::vector<std::shared_ptr<Probe>> v;
        for (int i = 0; i < 100; ++i) v.push_back(MakeEngineShared<Probe>(live));
        Expect(live == 100 && EngineAllocationCount() == 100,
               "shared objects come from the hook");
    }
    Expect(live == 0 && EngineAllocationCount() == 0,
           "shared objects return to the hook");

    {
        std::vector<int, EngineStlAllocator<int>> v;
        for (int i = 0; i < 10000; ++i) v.push_back(i);
        Expect(v[9999] == 9999, "stl allocator holds data");
    }
    Expect(EngineAllocationCount() == 0, "stl allocator frees through hook");
    SetEngineAllocator(nullptr);
}

void TestLifetimeTransitions() {
    int live = 0;
    std::shared_ptr<Probe> early;  // allocated on the default heap
    SetEngineAllocator(nullptr);
    early = MakeEngineShared<Probe>(live);

    std::shared_ptr<Probe> pooled;
    {
        MemorySystem sys({});
        SetEngineAllocator(&sys.Allocator());
        pooled = MakeEngineShared<Probe>(live);
        early.reset();  // default-heap block still goes to the default heap
        Expect(live == 1, "default-heap object freed while hooked");
        pooled.reset();
        Expect(live == 0 && EngineAllocationCount() == 0,
               "pooled object freed while hooked");

        pooled = MakeEngineShared<Probe>(live);
        SetEngineAllocator(nullptr);
        pooled.reset();  // allocator unhooked: leaked, must not crash
        Expect(live == 0, "destructor still runs after unhook");
    }

    // A block from a replaced allocator is skipped, not freed into the new
    // one. The old allocator stays alive here; outliving it is a caller error.
    MemorySystem a({}), b({});
    SetEngineAllocator(&a.Allocator());
    auto stale = MakeEngineShared<Probe>(live);
    SetEngineAllocator(&b.Allocator());
    stale.reset();
    Expect(live == 0 && EngineAllocationCount() == 0,
           "block from earlier generation is ignored");
    SetEngineAllocator(nullptr);
}

void TestFallback() {
    FailingAllocator failing;
    SetEngineAllocator(&failing);
    int live = 0;
    {
        auto sp = MakeEngineShared<Probe>(live);
        Expect(live == 1, "exhausted allocator falls back to default heap");
    }
    Expect(live == 0, "fallback block freed");
    SetEngineAllocator(nullptr);
}

void TestThreads() {
    MemorySystem sys({});
    SetEngineAllocator(&sys.Allocator());
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([] {
                int live = 0;
                for (int i = 0; i < 2000; ++i) {
                    auto a = MakeEngineShared<Probe>(live);
                    auto b = MakeEngineShared<Probe>(live);
                }
            });
        }
        for (auto& th : threads) th.join();
    }
    sys.Routing().FlushThreadCache();
    Expect(EngineAllocationCount() == 0, "threads leave no live blocks");
    SetEngineAllocator(nullptr);
}
}  // namespace

int main() {
    TestNoHook();
    TestHooked();
    TestLifetimeTransitions();
    TestFallback();
    TestThreads();
    if (failures == 0) std::cout << "EngineAllocationTest passed\n";
    return failures;
}

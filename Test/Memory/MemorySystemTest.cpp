#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>

#include "MemoryManager.hpp"
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

class NullMemoryManager : public MemoryManager {
   public:
    void* AllocateAlignedPage(size_t, size_t) override { return nullptr; }
    void* AllocatePage(size_t) override { return nullptr; }
};

void TestPlain() {
    MemoryManager mm;
    {
        MemorySystem sys(&mm, {});
        Expect(sys.IsConfigured(), "configured over a working page source");
        Expect(sys.Tracking() == nullptr, "tracking is off by default");
        Expect(&sys.Allocator() == &sys.Routing(),
               "plain system hands out the routing allocator");
        void* small = sys.Allocator().Allocate(64);
        void* large = sys.Allocator().Allocate(4096);
        Expect(small && large, "allocations succeed");
        sys.Allocator().Free(small);
        sys.Allocator().Free(large);
    }
    Expect(mm.PageCount() == 0, "destruction returns every page");
}

void TestFrameResource() {
    MemoryManager mm;
    MemorySystem::Config config;
    config.frameCapacity = 4096;
    {
        MemorySystem sys(&mm, config);
        const size_t used0 = sys.Frame().Used();
        {
            std::pmr::vector<int> v(&sys.FrameResource());
            for (int i = 0; i < 100; ++i) v.push_back(i);
            Expect(v[99] == 99, "container works on the frame arena");
        }
        Expect(sys.Frame().Used() > used0, "container drew from the arena");

        const uint64_t frame = sys.Frame().FrameNumber();
        sys.BeginFrame();
        sys.BeginFrame();
        Expect(sys.Frame().FrameNumber() == frame + 2, "BeginFrame advances");
        Expect(sys.Frame().Used() == used0, "arena recycled after rotation");

        bool threw = false;
        try {
            std::pmr::vector<char> big(&sys.FrameResource());
            big.resize(1 << 20);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        Expect(threw, "exhausting the arena throws bad_alloc");
    }
    Expect(mm.PageCount() == 0, "frame arenas return their pages");
}

void TestTracking() {
    MemoryManager mm;
    std::string report;  // outlives the system that reports into it
    {
        MemorySystem::Config config;
        config.tracking = true;
        MemorySystem sys(&mm, config);
        Expect(sys.Tracking() != nullptr, "tracking enabled");
        Expect(&sys.Allocator() == sys.Tracking(),
               "tracked system hands out the tracker");

        sys.Tracking()->SetLeakHandler(
            [&](const std::string& r) { report = r; });
        void* kept = sys.Allocator().Allocate(100);
        void* freed = sys.Allocator().Allocate(2000);
        sys.Allocator().Free(freed);
        Expect(sys.Tracking()->GetStats().liveCount == 1, "one live block");
        sys.Allocator().Free(kept);
        Expect(sys.Tracking()->Leaks().empty(), "no leaks after freeing");
        Expect(report.empty(), "no report while clean");

        sys.Allocator().Allocate(32);  // intentionally leaked
        Expect(sys.Tracking()->Leaks().size() == 1, "leak is visible");
    }
    Expect(report.find("1") != std::string::npos,
           "destruction reports the leaked block");
    Expect(mm.PageCount() == 0, "tracked destruction returns every page");
}

void TestUnavailablePages() {
    NullMemoryManager mm;
    MemorySystem sys(&mm, {});
    Expect(!sys.IsConfigured(), "unconfigured without pages");
    MemorySystem::Config config;
    config.tracking = true;
    MemorySystem tracked(&mm, config);
    Expect(tracked.Tracking() == nullptr, "no tracker over a dead allocator");
}
}  // namespace

int main() {
    TestPlain();
    TestFrameResource();
    TestTracking();
    TestUnavailablePages();

    if (failures == 0) std::cout << "All memory system tests passed\n";
    return failures == 0 ? 0 : 1;
}

#include <cstdint>
#include <cstring>
#include <iostream>

#include "FrameAllocator.hpp"
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

class FailingMemoryManager : public MemoryManager {
   public:
    void* AllocateAlignedPage(size_t size, size_t alignment) override {
        if (calls++ >= failAfter) return nullptr;
        return MemoryManager::AllocateAlignedPage(size, alignment);
    }
    int calls = 0;
    int failAfter = 1 << 30;
};

void TestConfiguration() {
    FailingMemoryManager mm;
    {
        FrameAllocator f(&mm);
        Expect(!f.IsConfigured() && f.FrameCount() == 0, "default unconfigured");
        Expect(f.Allocate(8) == nullptr, "unconfigured allocate fails");
        f.BeginFrame();
        f.FreeAll();
        Expect(f.Used() == 0 && f.Remaining() == 0 && f.CapacityPerFrame() == 0,
               "unconfigured queries are zero");

        Expect(!f.Initialize(0), "zero capacity rejected");
        Expect(!f.Initialize(256, 1), "single frame rejected");
        Expect(!f.Initialize(256, 0), "zero frames rejected");
        Expect(mm.PageCount() == 0, "rejected configuration holds nothing");

        // failure while acquiring the third arena releases the first two
        mm.calls = 0;
        mm.failAfter = 2;
        Expect(!f.Initialize(256, 3), "partial failure reported");
        Expect(!f.IsConfigured() && mm.PageCount() == 0,
               "partial failure releases acquired arenas");
        mm.failAfter = 1 << 30;

        Expect(f.Initialize(256, 3) && f.FrameCount() == 3, "three frames");
        Expect(mm.PageCount() == 3, "one arena per frame");
        Expect(f.CapacityPerFrame() == 256 && f.Remaining() == 256,
               "capacity per frame");
        Expect(f.Initialize(128) && mm.PageCount() == 2 && f.FrameCount() == 2,
               "re-initialize replaces arenas");
    }
    Expect(mm.PageCount() == 0, "destructor releases arenas");
    Expect(mm.InvalidFreeCount() == 0, "no invalid page frees");
    FrameAllocator g(&mm, 64);
    g.Release();
    Expect(!g.IsConfigured() && mm.PageCount() == 0, "Release");
}

void TestRotation() {
    MemoryManager mm;
    FrameAllocator f(&mm, 256);

    auto* a = static_cast<uint8_t*>(f.Allocate(64));
    std::memset(a, 0xA1, 64);
    Expect(f.Used() >= 64 && f.FrameNumber() == 0, "frame 0 allocation");

    f.BeginFrame();
    Expect(f.FrameNumber() == 1 && f.Used() == 0, "new frame starts empty");
    auto* b = static_cast<uint8_t*>(f.Allocate(64));
    Expect(b != a, "frames use different arenas");
    std::memset(b, 0xB2, 64);
    Expect(a[0] == 0xA1 && a[63] == 0xA1,
           "previous frame data survives the current frame");

    f.BeginFrame();
    Expect(f.Used() == 0, "third frame empty");
    Expect(f.Allocate(64) == a, "arena of frame N-2 is recycled");
    Expect(b[0] == 0xB2, "previous frame data still intact");

    for (int i = 0; i < 1001; i++) f.BeginFrame();
    Expect(f.FrameNumber() == 1003, "frame number counts every BeginFrame");
    Expect(mm.PageCount() == 2, "no extra arenas appear");
}

void TestDeeperHistory() {
    MemoryManager mm;
    FrameAllocator f(&mm, 128, 4);
    uint8_t* frames[4];
    for (int i = 0; i < 4; i++) {
        frames[i] = static_cast<uint8_t*>(f.Allocate(16));
        std::memset(frames[i], 0x10 + i, 16);
        if (i < 3) f.BeginFrame();
    }
    bool intact = true;
    for (int i = 0; i < 4; i++) {
        if (frames[i][0] != 0x10 + i) intact = false;
    }
    Expect(intact, "three previous frames stay valid with four arenas");
    f.BeginFrame();
    Expect(f.Allocate(16) == frames[0], "oldest arena recycled first");
}

void TestAllocationRules() {
    MemoryManager mm;
    FrameAllocator f(&mm, 128);

    Expect(f.Allocate(0) == nullptr, "zero size rejected");
    Expect(f.Allocate(8, 3) == nullptr, "bad alignment rejected");
    Expect(f.FailedAllocationCount() == 2, "failures counted");
    Expect(f.Allocate(200, 1) == nullptr, "oversized fails");
    Expect(reinterpret_cast<uintptr_t>(f.Allocate(1, 64)) % 64 == 0,
           "alignment honored");
    Expect(f.Allocate(128, 1) == nullptr, "exhausted frame fails");

    // exhaustion in one frame does not affect the next
    f.BeginFrame();
    Expect(f.Allocate(128, 1) != nullptr, "next frame has full capacity");

    IAllocator* a = &f;
    void* p = a->Allocate(1);
    a->Free(p);
    Expect(f.Used() >= 128, "Free is a no-op");
}

void TestFreeAll() {
    MemoryManager mm;
    FrameAllocator f(&mm, 128);
    auto* a = static_cast<uint8_t*>(f.Allocate(32, 1));
    f.BeginFrame();
    f.Allocate(32, 1);
    f.FreeAll();
    Expect(f.Used() == 0, "current frame cleared");
    f.BeginFrame();
    Expect(f.Allocate(32, 1) == a, "other arenas cleared too");
    Expect(mm.PageCount() == 2, "FreeAll keeps arenas");
}
}  // namespace

int main() {
    TestConfiguration();
    TestRotation();
    TestDeeperHistory();
    TestAllocationRules();
    TestFreeAll();

    if (failures == 0) std::cout << "All frame allocator tests passed\n";
    return failures == 0 ? 0 : 1;
}

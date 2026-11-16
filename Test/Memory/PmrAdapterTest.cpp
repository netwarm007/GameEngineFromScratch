#include <iostream>
#include <list>
#include <map>
#include <memory_resource>
#include <string>
#include <vector>

#include "BlockAllocator.hpp"
#include "FrameAllocator.hpp"
#include "LinearAllocator.hpp"
#include "MemoryManager.hpp"
#include "PmrAdapter.hpp"
#include "RingAllocator.hpp"
#include "StackAllocator.hpp"

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void TestVectorOnLinear() {
    MemoryManager mm;
    LinearAllocator arena(&mm, 4096);
    PmrAdapter<LinearAllocator> resource(arena);

    {
        std::pmr::vector<int> v(&resource);
        for (int i = 0; i < 100; i++) v.push_back(i);
        Expect(v.size() == 100 && v[99] == 99, "vector contents");
        Expect(arena.Used() >= 100 * sizeof(int), "memory came from the arena");
    }

    arena.FreeAll();
    std::pmr::string s("a string long enough to need heap storage, yes", &resource);
    Expect(arena.Used() >= s.size(), "string allocates from the arena");

    std::pmr::map<int, int> m(&resource);
    for (int i = 0; i < 20; i++) m[i] = i * i;
    Expect(m.size() == 20 && m[7] == 49, "map nodes from arena");
}

void TestAlignment() {
    MemoryManager mm;
    LinearAllocator arena(&mm, 1024);
    PmrAdapter<LinearAllocator> resource(arena);

    void* p = resource.allocate(8, 64);
    Expect(reinterpret_cast<uintptr_t>(p) % 64 == 0, "alignment forwarded");
    void* q = resource.allocate(0, 1);
    Expect(q != nullptr, "zero-byte request returns a pointer");
    resource.deallocate(p, 8, 64);
    resource.deallocate(q, 0, 1);
}

void TestFailureThrows() {
    MemoryManager mm;
    LinearAllocator arena(&mm, 64);
    PmrAdapter<LinearAllocator> resource(arena);

    bool thrown = false;
    try {
        (void)resource.allocate(1000, 1);
    } catch (const std::bad_alloc&) {
        thrown = true;
    }
    Expect(thrown, "exhaustion throws bad_alloc");

    std::pmr::vector<int> v(&resource);
    thrown = false;
    try {
        for (int i = 0; i < 1000; i++) v.push_back(i);
    } catch (const std::bad_alloc&) {
        thrown = true;
    }
    Expect(thrown, "container growth beyond the arena throws");
}

void TestEquality() {
    MemoryManager mm;
    LinearAllocator a(&mm, 64), b(&mm, 64);
    PmrAdapter<LinearAllocator> ra(a), ra2(a), rb(b);
    Expect(ra.is_equal(ra), "equal to itself");
    Expect(!ra.is_equal(ra2) && !ra.is_equal(rb),
           "distinct adapters never compare equal");
}

void TestStackLifo() {
    MemoryManager mm;
    StackAllocator stack(&mm, 1024);
    PmrAdapter<StackAllocator> resource(stack);

    void* a = resource.allocate(32, 16);
    void* b = resource.allocate(32, 16);
    resource.deallocate(b, 32, 16);
    resource.deallocate(a, 32, 16);
    Expect(stack.Used() == 0 && stack.InvalidFreeCount() == 0,
           "LIFO deallocation pops the stack");

    a = resource.allocate(32, 16);
    b = resource.allocate(32, 16);
    resource.deallocate(a, 32, 16);
    Expect(stack.InvalidFreeCount() == 1, "out-of-order release is rejected");
}

void TestBlockAllocator() {
    MemoryManager mm;
    BlockAllocator pool(&mm, 32, 1024, 16);
    PmrAdapter<BlockAllocator> resource(pool);

    void* p = resource.allocate(32, 16);
    Expect(p != nullptr, "pool allocation");
    resource.deallocate(p, 32, 16);
    Expect(pool.FreeBlockCount() == pool.BlockCount(), "block returned");

    bool thrown = false;
    try {
        (void)resource.allocate(33, 16);
    } catch (const std::bad_alloc&) {
        thrown = true;
    }
    Expect(thrown, "oversized pool request throws");

    thrown = false;
    const size_t freeBefore = pool.FreeBlockCount();
    try {
        (void)resource.allocate(32, 4096);
    } catch (const std::bad_alloc&) {
        thrown = true;
    }
    Expect(thrown, "unsatisfiable alignment throws");
    Expect(pool.FreeBlockCount() == freeBefore, "misaligned block given back");
}

void TestFrameAndRing() {
    MemoryManager mm;
    FrameAllocator frames(&mm, 1024);
    PmrAdapter<FrameAllocator> fr(frames);
    std::pmr::vector<int> v(&fr);
    for (int i = 0; i < 50; i++) v.push_back(i);
    Expect(frames.Used() > 0 && v[49] == 49, "frame allocator backs a vector");

    RingAllocator ring(&mm, 1024);
    PmrAdapter<RingAllocator> rr(ring);
    std::pmr::list<int> l(&rr);
    for (int i = 0; i < 10; i++) l.push_back(i);
    Expect(ring.Used() > 0 && l.size() == 10, "ring allocator backs a list");
}

void TestPolymorphicAllocatorPropagation() {
    MemoryManager mm;
    LinearAllocator arena(&mm, 8192);
    PmrAdapter<LinearAllocator> resource(arena);

    std::pmr::vector<std::pmr::string> names(&resource);
    names.emplace_back("first entry that is long enough to allocate");
    names.emplace_back("second entry that is long enough to allocate");
    const size_t used = arena.Used();
    Expect(used > 0, "nested containers use the arena");
    names.clear();
    Expect(arena.Used() == used, "arena memory is only reclaimed by FreeAll");
    arena.FreeAll();
    Expect(arena.Used() == 0, "reset");
}
}  // namespace

int main() {
    std::cerr << "run TestVectorOnLinear" << std::endl;
    TestVectorOnLinear();
    std::cerr << "run TestAlignment" << std::endl;
    TestAlignment();
    std::cerr << "run TestFailureThrows" << std::endl;
    TestFailureThrows();
    std::cerr << "run TestEquality" << std::endl;
    TestEquality();
    std::cerr << "run TestStackLifo" << std::endl;
    TestStackLifo();
    std::cerr << "run TestBlockAllocator" << std::endl;
    TestBlockAllocator();
    std::cerr << "run TestFrameAndRing" << std::endl;
    TestFrameAndRing();
    std::cerr << "run TestPolymorphicAllocatorPropagation" << std::endl;
    TestPolymorphicAllocatorPropagation();

    if (failures == 0) std::cout << "All PMR adapter tests passed\n";
    return failures == 0 ? 0 : 1;
}

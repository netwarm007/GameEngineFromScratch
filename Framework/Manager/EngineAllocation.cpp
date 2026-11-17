#include "EngineAllocation.hpp"

#include <atomic>

namespace My {

namespace {
constexpr size_t kHeader = 16;

struct Header {
    uint64_t generation;  // 0: default heap
    uint64_t reserved;
};
static_assert(sizeof(Header) == kHeader);

struct State {
    std::atomic<IAllocator*> allocator{nullptr};
    std::atomic<uint64_t> generation{0};
    std::atomic<size_t> live{0};
};

State& GetState() {
    // Leaked so frees during static destruction stay safe.
    static State* state = new State;
    return *state;
}
}  // namespace

void SetEngineAllocator(IAllocator* allocator) {
    auto& s = GetState();
    s.allocator.store(nullptr);
    if (allocator) {
        s.generation.fetch_add(1);
        s.live.store(0);
        s.allocator.store(allocator);
    }
}

void* EngineAllocate(size_t size) {
    auto& s = GetState();
    if (size > SIZE_MAX - kHeader) throw std::bad_alloc();
    const size_t total = size + kHeader;

    if (IAllocator* a = s.allocator.load()) {
        if (void* base = a->Allocate(total)) {
            auto* h = static_cast<Header*>(base);
            h->generation = s.generation.load();
            h->reserved = 0;
            ++s.live;
            return static_cast<uint8_t*>(base) + kHeader;
        }
    }

    void* base = ::operator new(total, std::align_val_t(16));
    auto* h = static_cast<Header*>(base);
    h->generation = 0;
    h->reserved = 0;
    return static_cast<uint8_t*>(base) + kHeader;
}

void EngineFree(void* p) {
    if (p == nullptr) return;
    auto& s = GetState();
    void* base = static_cast<uint8_t*>(p) - kHeader;
    const uint64_t gen = static_cast<Header*>(base)->generation;
    if (gen == 0) {
        ::operator delete(base, std::align_val_t(16));
        return;
    }
    IAllocator* a = s.allocator.load();
    if (a && gen == s.generation.load()) {
        --s.live;
        a->Free(base);
    }
}

size_t EngineAllocationCount() { return GetState().live.load(); }

}  // namespace My

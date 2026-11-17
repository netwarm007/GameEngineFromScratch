#include "ThreadCacheAllocator.hpp"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace My {

namespace {
// Maps live allocator ids to allocators so a thread that exits can return its
// cache only to allocators that still exist. Ids are never reused, which also
// keeps a new allocator at a recycled address from matching an old cache.
// Intentionally leaked: thread caches may be destroyed after static
// destruction has begun.
struct Registry {
    std::mutex mutex;
    std::unordered_map<uint64_t, ThreadCacheAllocator*> live;
    std::atomic<uint64_t> nextId{1};
};

Registry& GetRegistry() {
    static Registry* registry = new Registry;
    return *registry;
}
}  // namespace

struct ThreadCacheAllocator::ThreadCache {
    uint64_t id = 0;
    ThreadCacheAllocator* owner = nullptr;
    uint64_t generation = 0;
    void* head[SizeClassAllocator::kClassCount] = {};
    uint32_t count[SizeClassAllocator::kClassCount] = {};
};

struct ThreadCacheAllocator::ThreadCacheList {
    std::vector<std::unique_ptr<ThreadCache>> caches;

    ~ThreadCacheList() {
        Registry& registry = GetRegistry();
        for (auto& cache : caches) {
            std::lock_guard<std::mutex> lock(registry.mutex);
            auto it = registry.live.find(cache->id);
            if (it == registry.live.end()) continue;
            // Holding the registry lock keeps the owner from finishing its
            // destructor while its backing allocator is in use.
            it->second->Sync(*cache);
            it->second->FlushAllClasses(*cache);
        }
    }
};

ThreadCacheAllocator::ThreadCacheAllocator(IMemoryManager* pMmgr,
                                           size_t slabSize,
                                           uint32_t maxCachedPerClass)
    : IAllocator(pMmgr),
      m_Backing(pMmgr, slabSize),
      m_nMaxCached(maxCachedPerClass),
      m_nBatch((std::max)(1u, maxCachedPerClass / 2)) {
    Registry& registry = GetRegistry();
    m_nId = registry.nextId++;
    m_bConfigured = m_Backing.IsConfigured() && maxCachedPerClass >= 1;
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.live.emplace(m_nId, this);
}

ThreadCacheAllocator::~ThreadCacheAllocator() {
    Registry& registry = GetRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.live.erase(m_nId);
}

ThreadCacheAllocator::ThreadCache* ThreadCacheAllocator::GetCache() {
    thread_local ThreadCacheList list;
    for (auto& cache : list.caches) {
        if (cache->id == m_nId) return cache.get();
    }

    // First use by this thread: drop caches of destroyed allocators.
    {
        Registry& registry = GetRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        list.caches.erase(
            std::remove_if(list.caches.begin(), list.caches.end(),
                           [&](const std::unique_ptr<ThreadCache>& c) {
                               return registry.live.count(c->id) == 0;
                           }),
            list.caches.end());
    }
    auto cache = std::make_unique<ThreadCache>();
    cache->id = m_nId;
    cache->owner = this;
    cache->generation = m_nGeneration.load();
    list.caches.push_back(std::move(cache));
    return list.caches.back().get();
}

void ThreadCacheAllocator::Sync(ThreadCache& cache) {
    const uint64_t generation = m_nGeneration.load();
    if (cache.generation == generation) return;
    // FreeAll() already reclaimed everything in the cache.
    std::fill(std::begin(cache.head), std::end(cache.head), nullptr);
    std::fill(std::begin(cache.count), std::end(cache.count), 0u);
    cache.generation = generation;
}

void ThreadCacheAllocator::FlushClass(ThreadCache& cache, size_t index,
                                      uint32_t keep) {
    if (cache.count[index] <= keep) return;
    std::lock_guard<std::mutex> lock(m_Mutex);
    while (cache.count[index] > keep) {
        void* p = cache.head[index];
        cache.head[index] = *static_cast<void**>(p);
        --cache.count[index];
        m_Backing.Free(p);
    }
    ++m_nFlushes;
}

void ThreadCacheAllocator::FlushAllClasses(ThreadCache& cache) {
    for (size_t i = 0; i < SizeClassAllocator::kClassCount; i++) {
        FlushClass(cache, i, 0);
    }
}

void* ThreadCacheAllocator::Allocate(size_t size) {
    const size_t index = SizeClassAllocator::ClassIndex(size);
    if (!m_bConfigured || index >= SizeClassAllocator::kClassCount) {
        ++m_nFailures;
        return nullptr;
    }

    ThreadCache& cache = *GetCache();
    Sync(cache);
    if (cache.count[index] == 0) {
        const size_t classSize = SizeClassAllocator::ClassSizeAt(index);
        std::lock_guard<std::mutex> lock(m_Mutex);
        for (uint32_t i = 0; i < m_nBatch; i++) {
            void* p = m_Backing.Allocate(classSize);
            if (!p) break;
            *static_cast<void**>(p) = cache.head[index];
            cache.head[index] = p;
            ++cache.count[index];
        }
        ++m_nRefills;
        if (cache.count[index] == 0) {
            ++m_nFailures;
            return nullptr;
        }
    }

    void* p = cache.head[index];
    cache.head[index] = *static_cast<void**>(p);
    --cache.count[index];
    return p;
}

void ThreadCacheAllocator::Free(void* p) {
    if (!p) return;
    const size_t index = m_bConfigured ? m_Backing.ClassIndexOf(p)
                                       : SizeClassAllocator::kClassCount;
    if (index >= SizeClassAllocator::kClassCount) {
        ++m_nInvalidFrees;
        return;
    }

    ThreadCache& cache = *GetCache();
    Sync(cache);
    *static_cast<void**>(p) = cache.head[index];
    cache.head[index] = p;
    if (++cache.count[index] > m_nMaxCached) {
        FlushClass(cache, index, m_nMaxCached - m_nBatch);
    }
}

void ThreadCacheAllocator::FreeAll() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Backing.FreeAll();
    ++m_nGeneration;
}

void ThreadCacheAllocator::FlushThreadCache() {
    if (!m_bConfigured) return;
    ThreadCache& cache = *GetCache();
    Sync(cache);
    FlushAllClasses(cache);
}

size_t ThreadCacheAllocator::Trim() {
    FlushThreadCache();
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Backing.Trim();
}

size_t ThreadCacheAllocator::BackingLiveCount() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Backing.LiveCount();
}

size_t ThreadCacheAllocator::BackingSlabCount() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Backing.SlabCount();
}

size_t ThreadCacheAllocator::CachedCount() const {
    // GetCache is non-const only because it may create the cache.
    auto* self = const_cast<ThreadCacheAllocator*>(this);
    if (!m_bConfigured) return 0;
    ThreadCache& cache = *self->GetCache();
    self->Sync(cache);
    size_t n = 0;
    for (uint32_t c : cache.count) n += c;
    return n;
}

size_t ThreadCacheAllocator::InvalidFreeCount() const {
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_nInvalidFrees + m_Backing.InvalidFreeCount();
}
}  // namespace My

# Memory allocator roadmap

## Purpose and scope

Build a tested, standalone set of memory allocators suited to a game engine
(`Framework/Manager`: `MemoryManager`, `IAllocator`, `BlockAllocator`,
`StackAllocator`). This is a maintained engineering plan: update it as
checkpoints land, findings change, or priorities shift.

**Out of scope for now:** switching other engine modules to these allocators.
Integration is a separate later effort. Each allocator is delivered and
verified on its own.

## Delivery rules

- Work in checkpoints; each is an independently testable unit.
- Every allocator ships with thorough unit tests (normal paths, boundaries,
  exhaustion, alignment, reset/reuse, misuse detection where applicable).
- Each checkpoint must pass the CircleCI gate on all enabled platforms (Windows,
  Linux, macOS) for its exact revision before it counts as done; see
  `docs/verification/`. Pull requests are not created automatically.
- Documentation-only commits use `[ci skip]`.

## Review findings (baseline)

Reviewed before implementation. No engine code uses the allocators yet.

### BlockAllocator

| Severity | Issue |
| --- | --- |
| High | Single-argument constructor leaves statistics and size members uninitialized; `Reset` calls `FreeAll` on them. |
| High | `AllocatePage` returning `nullptr` is not checked. |
| High | `m_nBlocksPerPage == 0` underflows `m_nBlocksPerPage - 1` and overruns the page. |
| High | Block addresses are only as aligned as the 8-byte `PageHeader`; alignment of 16/32/64 is not guaranteed. |
| Medium | `Free` does not check null, ownership, block boundary, or double free. |
| Medium | Size check is only an `assert`; oversized requests overrun in Release. |
| Low | Global `ALIGN` macro; redundant `pNext` writes in `FillFreePage`; per-instance debug pattern members; undefined `IAllocator::~IAllocator`. |
| Low | Free list is one list: poor locality over time, pages never returned, no thread-safety statement. |

### StackAllocator

An empty shell: `Allocate` returns `nullptr`, the destructor is not defined,
the header lacks `#pragma once`, and its members (`std::list`,
`shared_ptr<void>`) are unsuitable for a stack allocator.

### MemoryManager

- `std::map` bookkeeping per page: extra heap allocations, O(log n), no locking.
- `FreePage` silently ignores unknown pointers (hides double free).
- Leak detection is an `assert` only; no report in Release.
- `Tick` uses `#if DEBUG` (the rest uses `_DEBUG`), triggers once, no separators.
- `MemoryType::GPU` unused; no aligned page allocation; no statistics.

## Planned allocators

Priority order, highest first.

| # | Allocator | Use in a game engine |
| --- | --- | --- |
| 1 | Linear / arena (bump) | Per-frame scratch, command buffers, load-time data; reset as a whole. |
| 2 | Stack with markers | Scoped temporaries; `GetMarker` / `FreeToMarker`. |
| 3 | Double-buffered frame allocator | Data that must survive one more frame across threads. |
| 4 | Pool (fixed block) | Components, nodes, particles, tasks. Fixed `BlockAllocator`. |
| 5 | Size-class small-object allocator | Replace general `malloc` for small objects (8..512 B). |
| 6 | Slab | Pool plus cached constructed objects. |
| 7 | TLSF | O(1) general allocator, low fragmentation, real-time safe. |
| 8 | Buddy | Power-of-two page and heap management. |
| 9 | GPU sub-allocator | Linear / ring / TLSF over Vulkan, D3D12, Metal heaps (`MemoryType::GPU`). |
| 10 | Ring | Per-frame uploads, streaming; needs fence-based reclamation. |
| 11 | Handle-based / compacting | Resource manager data that must be defragmented. |
| 12 | Thread-cache layer | Per-thread caches over a shared backing; or adopt an existing allocator. |
| 13 | Debug / tracking decorator | Guard bytes, leak reports, statistics around any allocator. |

Also: adapt allocators to `std::pmr::memory_resource` so standard containers
can use them.

## Known limitations and follow-ups

Open items only; remove or mark resolved when fixed. Per-revision details are in
`docs/verification/`.

| Limitation | Planned handling |
| --- | --- |
| `LinearAllocator` and `StackAllocator` are single-threaded with fixed capacity (no growth). | Documented; use one instance per thread (the thread-cache layer in M7 covers small objects only). |
| `BlockAllocator` free-pointer validation is O(pages + free blocks) and on by default only in `_DEBUG`. | Use `TrackingAllocator` (M6) for registry-based ownership checks when validation is off. |
| `BlockAllocator` never returns pages before `FreeAll`/destruction; single free list gives poor locality over time. | Use `SlabAllocator`/`SizeClassAllocator` (`Trim`) where page return matters; `BlockAllocator` itself stays as is. |
| `SlabAllocator`/`SizeClassAllocator` are single-threaded; validation off (release default) cannot detect pointers outside any slab, and `Trim` is manual (no automatic shrink policy). | Resolved for small objects by `ThreadCacheAllocator` (M7); `TrackingAllocator` ownership checks (M6, done). |
| `SizeClassAllocator` serves only 1..512 bytes and 16-byte alignment; larger requests return `nullptr` and the caller must route them elsewhere. | General allocator (TLSF/Buddy, M5) and routing facade (M8). |
| Debug fill patterns are only verified in builds defining `_DEBUG`. | Open; `TrackingAllocator` adds guard bytes but does not check fill patterns. |
| `FrameAllocator` and `RingAllocator` are single-threaded; `RingAllocator` leaves fence tracking (when to `Retire`) to the caller. | Per-thread instances; `GpuSubAllocator` (M7) owns fence-based reclamation for GPU memory, `RingAllocator` stays caller-driven. |
| `PmrAdapter` forwards `deallocate` to `Free`, so LIFO/arena allocators only reclaim as their own rules allow, and failures throw `std::bad_alloc`. | Documented behavior; revisit if engine builds disable exceptions (M8). |
| `TlsfAllocator` and `BuddyAllocator` are single-threaded with fixed capacity (no growth by adding pools). | Per-thread instances or an external lock; pool growth is open and goes with engine integration (M8). |
| `TlsfAllocator` is good-fit: a request within ~1/32 of the only free block can fail; its `Free` checks inspect block headers and can be fooled by crafted user data. | Documented; `TrackingAllocator` (M6) rejects unregistered pointers before they reach `Free`. |
| `BuddyAllocator` wastes up to 50% per block and supports alignment only up to 4096 bytes. | Intended for large or page-like blocks; documented. |
| `TrackingAllocator` has a rear guard only (no underrun detection), no call stacks, adds 16 bytes plus a registry entry per block, and holds a global mutex; keeps the wrapped allocator's alignment (no aligned allocate); tag pointers must outlive allocations. | Debug/tooling use only; aligned allocate and call stacks if needed (M8). |
| `ThreadCacheAllocator` serves only 1..512 bytes, cannot detect double frees (cached objects look live), and returns a thread's cache only on thread exit, `FlushThreadCache()` or `Trim()`. `FreeAll()` must not run concurrently. | Use `TrackingAllocator` on top for double-free detection; routing facade (M8). |
| `GpuSubAllocator` manages one heap (no growth or multiple heaps), is single-threaded, uses O(log n) ordered maps (not a constant-time TLSF), and is not yet wired to any RHI backend (Vulkan/D3D12/Metal). | Heap growth and RHI binding with engine integration (M8). |
| Allocators are not used by any engine module. | Integration (M8). |

## Foundation improvements

- `AllocatePage` accepts an alignment; pages aligned to at least a cache line.
- Intrusive page metadata instead of `std::map`; keep a debug-only registry.
- Default member initializers and parameter validation everywhere.
- Report failures (null returns, invalid frees) rather than silently ignoring.
- Allocation statistics (current, peak, count) switchable by build option.
- Document the thread-safety model of each allocator.

## Checkpoints

| ID | Scope | Status |
| --- | --- | --- |
| M0 | This roadmap | Done (docs only) |
| M1 | Foundation and BlockAllocator fixes: initialization, validation, alignment, null handling, double-free detection; tests (`Test/Memory/MemoryAllocatorTest`). | Done (CI-verified, `docs/verification/03a8504ee0dc1fd042c3444d520f2b3863c85304.md`) |
| M2 | `LinearAllocator` and marker-based `StackAllocator` (replace the shell); tests (`LinearAllocatorTest`, `StackAllocatorTest`). | Done (CI-verified, `docs/verification/faa5744f42b9aec60d6b93f561af22a95378320e.md`) |
| M3 | Frame (double-buffered) and ring allocators; `std::pmr` adapter; stale `Marker` detection; tests (`FrameAllocatorTest`, `RingAllocatorTest`, `PmrAdapterTest`, `StackAllocatorTest`). | Done (CI-verified, `docs/verification/89926ec5081f370ca6519c6fbeba021a1fc6cbd6.md`) |
| M4 | `SlabAllocator` (constructed-object cache, empty-slab `Trim`) and `SizeClassAllocator` (16 classes, 16 B to 512 B); tests (`SlabAllocatorTest`, `SizeClassAllocatorTest`). | Done (CI-verified, `docs/verification/b6fab1866eb639ccc1dd0395e526b99527032a44.md`) |
| M5 | `TlsfAllocator` and `BuddyAllocator`; tests (`TlsfAllocatorTest`, `BuddyAllocatorTest`). | Done (CI-verified, `docs/verification/e81a6ccb009943eac7627ec6fc16acdb59810421.md`) |
| M6 | `TrackingAllocator` debug decorator: statistics, tags, guard bytes, registry-based free checks, leak reports; tests (`TrackingAllocatorTest`). | Done (CI-verified, `docs/verification/05b0b2616b0c56f5bca8b66f46147da3c254c326.md`) |
| M7 | `GpuSubAllocator` (offset allocator with fence-based deferred free) and `ThreadCacheAllocator` (per-thread caches over `SizeClassAllocator`); tests (`GpuSubAllocatorTest`, `ThreadCacheAllocatorTest`). | Done (CI-verified, `docs/verification/857656ce14a8d5d9192cd4a4e9b6ef7f6caae43e.md`) |
| M8 | Engine integration, pilot only and off by default (see "M8 plan"). | Planned |

M1 notes: `MemoryManager` no longer depends on `portable.hpp`/`config.h`, so
allocator tests build standalone (`cmake -S Test/Memory`). `IMemoryManager`
gained `AllocateAlignedPage` (default falls back to `AllocatePage`).
`BlockAllocator` allocates blocks lazily from the newest page; free-pointer
validation (ownership, alignment, double free) is on in `_DEBUG` builds and
enabled explicitly with `SetValidation(true)`.

M2 notes: both allocators own one fixed-capacity, 64-byte-aligned arena from
`AllocateAlignedPage`, accept per-allocation power-of-two alignment, and return
`nullptr` instead of growing. `LinearAllocator::FreeAll` keeps the arena.
`StackAllocator` keeps a 16-byte header per allocation so `Free(p)` can pop the
latest allocation; `Marker` rewinds many at once. Shared helpers are in
`AllocatorUtils.hpp`.

M7 notes: `GpuSubAllocator` is backend-agnostic: it hands out offsets into a
heap and keeps all bookkeeping on the CPU, so it also works for device-local
memory. Free ranges live in an ordered map (coalescing) plus a size-ordered set
(best fit); `FreeDeferred(handle, fence)` plus `Retire(completed)` keep memory
reserved until the GPU is done. Handles carry a serial id so stale handles are
rejected. `ThreadCacheAllocator` keeps an intrusive free list per size class in
each thread, moves objects to or from a mutex-protected `SizeClassAllocator` in
batches, and returns a thread's cache when the thread exits. Allocator ids are
never reused and a registry tells exiting threads which allocators still exist.
`SizeClassAllocator::ClassIndexOf` finds a pointer's class from the slab header.

M6 notes: `TrackingAllocator` wraps any `IAllocator&` without owning it. It
appends a 16-byte guard band after each block, keeps an `unordered_map` of live
blocks, and never forwards unknown or double-freed pointers to the wrapped
allocator (counted as `invalidFrees`). `CheckGuards()` scans live blocks;
`LeakReport()`/`SetLeakHandler` report survivors, and the destructor reports
leaks automatically.

M5 notes: `TlsfAllocator` keeps a 16-byte header per block (previous-physical
pointer plus size and free flag), 32 second-level bins per power of two, and
two bitmaps for O(1) lookup; neighbours coalesce on `Free`, and alignments above
16 split the leading gap off as a free block. `BuddyAllocator` keeps no headers
in user memory: a byte-per-unit state table records block starts, orders and
free flags, and free lists are intrusive. Both expose `Check()` for heap
invariants, which the tests call throughout randomized workloads.

M4 notes: `SlabAllocator` allocates slabs aligned to their own (power-of-two)
size, so `Free` finds the owning slab by masking the pointer and needs no size.
Free-index lists and in-use flags live in the slab header, which keeps freed
objects intact; with a constructor/destructor pair objects stay constructed
across `Free`/`Allocate` and are destroyed only when a slab is released.
`SizeClassAllocator` owns one `SlabAllocator` per class and finds the class
from the slab header (or by ownership walk when validation is on).

M3 notes: `StackAllocator::Header` now carries a never-reused 64-bit serial, and
`Marker` records the serial of the newest allocation, so markers of unwound
allocations, markers across `FreeAll`, and forged markers are rejected.
`FrameAllocator` rotates N (>=2) `LinearAllocator` arenas. `RingAllocator` uses
monotonic 64-bit logical positions; `GetHead()`/`Retire()` implement FIFO
reclamation. `PmrAdapter<A>` (header only) wraps any allocator as a
`std::pmr::memory_resource`.

Test lessons: debug-mode standard containers keep a live proxy allocated from
their memory resource, so containers must be destroyed before `FreeAll()`;
alignment-failure tests must not assume buffer addresses (a large alignment
boundary may fall inside a small buffer).

Checkpoint scopes may be re-split as implementation reveals complexity.

## M8 plan: engine integration

Goal: use the allocators in the engine behind switches, without changing
behavior or rendering output. Current state: `MemoryManager` is only a page
source registered with `BaseApplication`; no module uses an allocator and there
is no global `operator new` override.

In scope: a `MemorySystem` owned by the application; a routing facade (blocks
up to 512 bytes to `ThreadCacheAllocator`, larger to `TlsfAllocator`, with
aligned allocation); `FrameAllocator` plus `PmrAdapter` for per-frame data;
optional `TrackingAllocator` with a leak report at shutdown; periodic statistics
from `MemoryManager::Tick`.

Out of scope: global `new`/`delete` replacement, bulk container conversion, and
binding `GpuSubAllocator` to Vulkan/D3D12/Metal heaps (a later checkpoint).

Policy: CMake options `USE_ENGINE_ALLOCATORS` (default OFF) and
`ENABLE_ALLOCATOR_TRACKING`. Pilot only 1-2 per-frame temporary data sites and
1-2 high-frequency small object types.

| Step | Scope |
| --- | --- |
| M8.1 | Routing facade with aligned allocation; unit and multithreaded tests. |
| M8.2 | `MemorySystem` wiring in `BaseApplication`; CMake options. |
| M8.3 | Per-frame pilot with `FrameAllocator` and pmr containers. |
| M8.4 | Small-object pilot (class-level `operator new` or pmr). |
| M8.5 | Shutdown leak report and statistics output; tracking on in Debug. |
| M8.6 | Compare switch on/off; decide whether to enable by default. |

Tests: facade routing (512/513 boundary, alignment, double free), stress under
ASan/UBSan/TSan, `Test/Framework` lifecycle tests with the switch on and off
(no leaks), screenshot parity between on and off.

Acceptance: each step passes Windows, Linux and macOS CircleCI on its exact
revision with a verification record; switch OFF behaves exactly as today;
switch ON passes all existing tests and screenshot checks; no leaks reported.

Risks: Windows debug-container lifetime with pmr (seen in M3);
`ThreadCacheAllocator::FreeAll` must not run concurrently; cross-module frees
must match their allocator, hence the small pilot scope.

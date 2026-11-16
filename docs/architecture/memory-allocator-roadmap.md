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
| `LinearAllocator` and `StackAllocator` are single-threaded with fixed capacity (no growth). | Document per allocator; thread-cache layer in M7. |
| `BlockAllocator` free-pointer validation is O(pages + free blocks) and on by default only in `_DEBUG`. | Cheaper ownership lookup with the debug decorator (M6). |
| `BlockAllocator` never returns pages before `FreeAll`/destruction; single free list gives poor locality over time. | Revisit with size-class allocator (M4). |
| Debug fill patterns are only verified in builds defining `_DEBUG`. | Extend tests to check patterns in the debug decorator (M6). |
| `FrameAllocator` and `RingAllocator` are single-threaded; `RingAllocator` leaves fence tracking (when to `Retire`) to the caller. | Thread-cache layer (M7); GPU sub-allocator owns fences (M7). |
| `PmrAdapter` forwards `deallocate` to `Free`, so LIFO/arena allocators only reclaim as their own rules allow, and failures throw `std::bad_alloc`. | Documented behavior; revisit if engine builds disable exceptions (M8). |
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
| M3 | Frame (double-buffered) and ring allocators; `std::pmr` adapter; stale `Marker` detection; tests (`FrameAllocatorTest`, `RingAllocatorTest`, `PmrAdapterTest`, `StackAllocatorTest`). | Implemented; awaiting CI acceptance |
| M4 | Size-class small-object allocator and Slab; tests. | Planned |
| M5 | TLSF and Buddy; tests. | Planned |
| M6 | Debug/tracking decorator, statistics, leak reports; tests. | Planned |
| M7 | GPU sub-allocator and thread-cache layer; tests. | Planned |
| M8 | Engine integration (separate decision). | Not started |

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

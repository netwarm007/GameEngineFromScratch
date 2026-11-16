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
| M1 | Foundation and BlockAllocator fixes: initialization, validation, alignment, null handling, double-free detection; tests (`MemoryAllocatorTest`). | Planned |
| M2 | `LinearAllocator` and marker-based `StackAllocator` (replace the shell); tests. | Planned |
| M3 | Frame (double-buffered) and ring allocators; `std::pmr` adapter; tests. | Planned |
| M4 | Size-class small-object allocator and Slab; tests. | Planned |
| M5 | TLSF and Buddy; tests. | Planned |
| M6 | Debug/tracking decorator, statistics, leak reports; tests. | Planned |
| M7 | GPU sub-allocator and thread-cache layer; tests. | Planned |
| M8 | Engine integration (separate decision). | Not started |

Checkpoint scopes may be re-split as implementation reveals complexity.

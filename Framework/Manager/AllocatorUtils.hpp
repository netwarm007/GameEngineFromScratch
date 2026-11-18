#pragma once
#include <cstddef>
#include <cstdint>

namespace My {
inline bool IsPowerOfTwo(size_t a) { return a != 0 && (a & (a - 1)) == 0; }

// Rounds addr up to a power-of-two alignment; false on address overflow.
inline bool AlignAddress(uintptr_t addr, size_t align, uintptr_t& out) {
    const uintptr_t mask = align - 1;
    if (addr > UINTPTR_MAX - mask) return false;
    out = (addr + mask) & ~mask;
    return true;
}
}  // namespace My

#pragma once
// based on https://gist.github.com/mertin-kallman/5049614
// float32
// Martin Kallman
//
// Fast half-precision to single-precision floating point conversion
//  - Supports signed zero and denormals-as-zero (DAZ)
//  - Does not support infinities or NaN
//  - Few, partically pipelinable, non-branching instructions,
//  - Core operations ~6 clock cycles on modern x86-64

inline float float32(const uint16_t in) {
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = in & 0x7fffu;      // Non-sign bits
    t2 = in & 0x8000u;      // Sign bit
    t3 = in & 0x7c00u;      // Exponent

    t1 <<= 13u;             // Align mantissa on MSB
    t2 <<= 16u;             // Shift sign bit into position

    t1 += 0x38000000;       // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);

    t1 |= t2;

    return *reinterpret_cast<float *>(&t1);
}

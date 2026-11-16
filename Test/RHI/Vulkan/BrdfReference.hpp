#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "geommath.hpp"

namespace My::VulkanTest {
inline std::array<double, 2> ReferenceBrdf(uint32_t x, uint32_t y) {
    const double nv = (x + 1) / 512.0;
    const double roughness = (y + 1) / 512.0;
    const double vx = std::sqrt(1 - nv * nv);
    const double alphaSquared = std::pow(roughness, 4);
    const double k = roughness * roughness / 2;
    std::array<double, 2> integral{};
    for (uint32_t i = 0; i < 1024; ++i) {
        double radicalInverse = 0;
        double weight = 0.5;
        for (auto bits = i; bits; bits >>= 1, weight *= 0.5) {
            if (bits & 1) radicalInverse += weight;
        }
        const double phi = 2 * PI * i / 1024.0;
        const double hz = std::sqrt((1 - radicalInverse) /
            (1 + (alphaSquared - 1) * radicalInverse));
        const double hx = std::sin(phi) * std::sqrt((std::max)(0.0, 1 - hz * hz));
        const double vh = (std::max)(0.0, vx * hx + nv * hz);
        const double nl = 2 * vh * hz - nv;
        if (nl <= 0) continue;
        const double geometry = nv / (nv * (1 - k) + k) *
                                nl / (nl * (1 - k) + k);
        const double visibility = geometry * vh / (hz * nv);
        const double fresnel = std::pow(1 - vh, 5);
        integral[0] += (1 - fresnel) * visibility / 1024;
        integral[1] += fresnel * visibility / 1024;
    }
    return integral;
}
}  // namespace My::VulkanTest

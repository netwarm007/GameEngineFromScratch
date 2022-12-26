#pragma once
#include <concepts>
#include <random>
#include "geommath.hpp"

namespace My {
template <class T>
T random_f() {
    thread_local std::mt19937 generator;
    thread_local std::uniform_real_distribution<T> distribution(0.0, 1.0);
    return distribution(generator);
}

template <class T>
T random_f(T min, T max) {
    thread_local std::mt19937 generator;
    std::uniform_real_distribution<T> distribution(min, max);
    return distribution(generator);
}

template <class T> requires std::integral<T>
T random_int(T min, T max) {
    return static_cast<T>(random_f<double>(static_cast<double>(min), static_cast<double>(max) + 1.0));
}

template <class T, Scalar auto N>
Vector<T, N> random_v() {
    auto vec = Vector<T, N>();
    for (int i = 0; i < N; i++) {
        vec[i] = random_f<T>();
    }

    return vec;
}

template <class T, Scalar auto N>
Vector<T, N> random_v(T min, T max) {
    auto vec = Vector<T, N>();
    for (int i = 0; i < N; i++) {
        vec[i] = random_f<T>(min, max);
    }

    return vec;
}

template <class T, Scalar auto N>
Vector<T, N> random_in_unit_sphere() {
    while (true) {
        auto p = random_v<T, N>(T(-1), T(1));
        if (LengthSquared(p) >= 1) continue;
        return p;
    }
}

template <class T, Scalar auto N>
Vector<T, N> random_unit_vector() {
    auto p = random_in_unit_sphere<T, N>();
    Normalize(p);
    return p;
}

template <class T, Scalar auto N>
Vector<T, N> random_in_hemisphere(const Vector<T, N>& normal) {
    auto p = random_in_unit_sphere<T, N>();
    T result;
    DotProduct<T, N>(result, p, normal);
    if (result > 0.0) {
        return p;
    } else {
        return -p;
    }
}

template <class T>
Vector3<T> random_in_hemisphere_cosine_weighted(const Vector3<T>& normal) {
    auto uv = random_v<T, 2>();
    T phi = 2.0 * PI * uv[0];

    T cos_phi = std::cos(phi);
    T sin_phi = std::sin(phi);

    T cos_theta = sqrt(uv[1]);
    T sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    return Vector3<T>(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

template <class T>
Vector3<T> random_in_unit_disk() {
    while (true) {
        auto p = Vector3<T>({random_f(T(-1.0), T(1.0)), random_f(T(-1.0), T(1.0)), 0});
        if (LengthSquared(p) >= 1) continue;
        return p;
    }
}
}  // namespace My

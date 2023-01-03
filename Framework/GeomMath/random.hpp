#pragma once
#include <concepts>
#include "geommath.hpp"

#ifdef __CUDACC__
#include <curand_kernel.h>
#else
#include <random>
#endif

namespace My {
#ifdef __CUDACC__

template <class T>
__device__ T random_f(curandState *local_rand_state) {
    return curand_uniform(local_rand_state);
}

template <class T>
__device__ T random_f(T min, T max, curandState *local_rand_state) {
    T scale = max - min;
    return min + scale * curand_uniform(local_rand_state);
}

template <class T> requires std::integral<T>
__device__ T random_int(T min, T max, curandState *local_rand_state) {
    return static_cast<T>(random_f<T>(static_cast<T>(min), static_cast<T>(max), local_rand_state));
}

template <class T, Dimension auto N>
__device__ Vector<T, N> random_v(curandState *local_rand_state) {
    auto vec = Vector<T, N>();
    for (int i = 0; i < N; i++) {
        vec[i] = random_f<T>(local_rand_state);
    }

    return vec;
}

template <class T, Dimension auto N>
__device__ Vector<T, N> random_v(T min, T max, curandState *local_rand_state) {
    auto vec = Vector<T, N>();
    for (int i = 0; i < N; i++) {
        vec[i] = random_f<T>(min, max, local_rand_state);
    }

    return vec;
}

template <class T, Dimension auto N>
__device__ Vector<T, N> random_in_unit_sphere(curandState *local_rand_state) {
    while (true) {
        auto p = random_v<T, N>(T(-1), T(1), local_rand_state);
        if (LengthSquared(p) >= 1) continue;
        return p;
    }
}

template <class T, Dimension auto N>
__device__ Vector<T, N> random_unit_vector(curandState *local_rand_state) {
    auto p = random_in_unit_sphere<T, N>(local_rand_state);
    Normalize(p);
    return p;
}

template <class T, Dimension auto N>
__device__ Vector<T, N> random_in_hemisphere(const Vector<T, N>& normal, curandState *local_rand_state) {
    auto p = random_in_unit_sphere<T, N>(local_rand_state);
    T result;
    DotProduct<T, N>(result, p, normal);
    if (result > 0.0) {
        return p;
    } else {
        return -p;
    }
}

template <class T>
__device__ Vector3<T> random_in_hemisphere_cosine_weighted(const Vector3<T>& normal, curandState *local_rand_state) {
    auto uv = random_v<T, 2>(local_rand_state);
    T phi = 2.0 * PI * uv[0];

    T cos_phi = std::cos(phi);
    T sin_phi = std::sin(phi);

    T cos_theta = sqrt(uv[1]);
    T sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    return Vector3<T>({sin_theta * cos_phi, cos_theta, sin_theta * sin_phi}) + normal - Vector3<T>({0, 1, 0});
}

template <class T>
__device__ Vector3<T> random_in_unit_disk(curandState *local_rand_state) {
    while (true) {
        auto p = Vector3<T>({random_f(T(-1.0), T(1.0), local_rand_state), random_f(T(-1.0), T(1.0), local_rand_state), 0});
        if (LengthSquared(p) >= (T)1.0) continue;
        return p;
    }
}
#else
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

template <class T, Dimension auto N>
Vector<T, N> random_v() {
    auto vec = Vector<T, N>();
    for (int i = 0; i < N; i++) {
        vec[i] = random_f<T>();
    }

    return vec;
}

template <class T, Dimension auto N>
Vector<T, N> random_v(T min, T max) {
    auto vec = Vector<T, N>();
    for (int i = 0; i < N; i++) {
        vec[i] = random_f<T>(min, max);
    }

    return vec;
}

template <class T, Dimension auto N>
Vector<T, N> random_in_unit_sphere() {
    while (true) {
        auto p = random_v<T, N>(T(-1), T(1));
        if (LengthSquared(p) >= 1) continue;
        return p;
    }
}

template <class T, Dimension auto N>
Vector<T, N> random_unit_vector() {
    auto p = random_in_unit_sphere<T, N>();
    Normalize(p);
    return p;
}

template <class T, Dimension auto N>
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

    return Vector3<T>({sin_theta * cos_phi, cos_theta, sin_theta * sin_phi}) + normal - Vector3<T>({0, 1, 0});
}

template <class T>
Vector3<T> random_in_unit_disk() {
    while (true) {
        auto p = Vector3<T>({random_f(T(-1.0), T(1.0)), random_f(T(-1.0), T(1.0)), 0});
        if (LengthSquared(p) >= 1) continue;
        return p;
    }
}
#endif
}  // namespace My

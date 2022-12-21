#include "geommath.hpp"
#include <random>

namespace My {
    template <class T>
    inline T random_f() {
        static std::uniform_real_distribution<T> distribution(0.0,
                                                                            1.0);
        static std::mt19937 generator;
        return distribution(generator);
    }

    template <class T>
    inline T random_f(T min, T max) {
        static std::uniform_real_distribution<T> distribution(min,
                                                                            max);
        static std::mt19937 generator;
        return distribution(generator);
    }

    template <class T, size_t N>
    inline Vector<T, N> random_v() {
        auto vec = Vector<T, N>();
        for (int i = 0; i < N; i++) {
            vec[i] = random_f<T>();
        }

        return vec;
    }

    template <class T, size_t N>
    inline Vector<T, N> random_v(T min, T max) {
        auto vec = Vector<T, N>();
        for (int i = 0; i < N; i++) {
            vec[i] = random_f<T>(min, max);
        }

        return vec;
    }

    template <class T, size_t N>
    inline Vector<T, N> random_in_unit_sphere() {
        while (true) {
            auto p = random_v<T, N>(T(-1), T(1));
            if (LengthSquared(p) >= 1) continue;
            return p;
        }
    }
}

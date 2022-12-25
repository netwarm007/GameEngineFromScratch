#include <climits>
#include <cstdint>
#include <iostream>
#include <typeinfo>
#include "geommath.hpp"
#include "random.hpp"

template <class T>
    requires std::integral<T>
void test_random_int() {
    fprintf(stderr, "Random %s:\n", typeid(T).name());
    std::cerr.setf(std::ios::dec);
    for (int i = 0; i < 100; i++) {
        std::cerr << +My::random_int<T>((std::numeric_limits<T>::min)(),
                                        (std::numeric_limits<T>::max)())
                  << "\t";
    }
    std::putchar('\n');
}

template <class T>
    requires std::floating_point<T>
void test_random_f() {
    fprintf(stderr, "Random %s:\n", typeid(T).name());
    std::cerr.setf(std::ios::dec);
    for (int i = 0; i < 100; i++) {
        std::cerr << My::random_f<T>(-1.0, 1.0) << "\t";
    }
    std::putchar('\n');
}

int main() {
    test_random_int<int8_t>();
    test_random_int<uint8_t>();
    test_random_int<int16_t>();
    test_random_int<uint16_t>();
    test_random_int<int32_t>();
    test_random_int<uint32_t>();
    test_random_int<int64_t>();
    test_random_int<uint64_t>();
    test_random_int<long>();
    test_random_int<long long>();
    test_random_int<size_t>();

    test_random_f<float>();
    test_random_f<double>();

    return 0;
}
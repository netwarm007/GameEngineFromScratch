#pragma once
#include <cstdint>
#include <climits>
#include <memory>
#include "config.h"

typedef int32_t four_char_enum;

#if __cplusplus >= 201103L && !defined(__ORBIS__)
#define ENUM(e) enum class e : four_char_enum 
#else
#define ENUM(e) enum e : four_char_enum
#endif

#ifndef HAVE_MAKE_UNIQUE 
namespace std {
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

template <typename T>
T endian_native_unsigned_int(T net_number)
{
    T result = 0;

    for(size_t i = 0; i < sizeof(net_number); i++) {
        result <<= CHAR_BIT;
        result += ((reinterpret_cast<T*>(&net_number))[i] & UCHAR_MAX);
    }

    return result;
}

template <typename T>
T endian_net_unsigned_int(T native_number)
{
    T result = 0;

	size_t i = sizeof(native_number);
    do {
		i--;
        (reinterpret_cast<uint8_t*>(&result))[i] = native_number & UCHAR_MAX;
        native_number >>= CHAR_BIT;
    } while (i != 0);

    return result;
}


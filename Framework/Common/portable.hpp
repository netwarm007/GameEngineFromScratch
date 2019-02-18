#pragma once
#include <cstdint>
#include <climits>
#include <memory>
#include <algorithm>
#include <assert.h>
#include "config.h"

#ifdef ALIGN
#undef ALIGN
#endif

#define ALIGN(x, a)         (((x) + ((a) - 1)) & ~((a) - 1))

typedef int32_t four_char_enum;

#define ENUM(e) enum class e : four_char_enum 

#ifndef HAVE_MAKE_UNIQUE 
namespace std {
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

#ifndef HAVE_CLAMP
namespace std {
    template<class T>
    const T& clamp( const T& v, const T& lo, const T& hi )
    {
        return clamp( v, lo, hi, std::less<T>() );
    }

    template<class T, class Compare>
    const T& clamp( const T& v, const T& lo, const T& hi, Compare comp )
    {
        return assert( !comp(hi, lo) ),
            comp(v, lo) ? lo : comp(hi, v) ? hi : v;
    }
}
#endif

namespace My {
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

    namespace details {
        constexpr int32_t i32(const char* s, int32_t v) {
            return *s ? i32(s+1, v * 256 + *s) : v;
        }

        constexpr uint16_t u16(const char* s, uint16_t v) {
            return *s ? u16(s+1, v * 256 + *s) : v;
        }

        constexpr uint32_t u32(const char* s, uint32_t v) {
            return *s ? u32(s+1, v * 256 + *s) : v;
        }
    }

    constexpr int32_t operator "" _i32(const char* s, size_t) {
        return details::i32(s, 0);
    }

    constexpr uint32_t operator "" _u32(const char* s, size_t) {
        return details::u32(s, 0);
    }

    constexpr uint16_t operator "" _u16(const char* s, size_t) {
        return details::u16(s, 0);
    }
}

#ifdef __OBJC__
#define OBJC_CLASS(name) @class name
#else
#define OBJC_CLASS(name) typedef struct objc_object name
#endif

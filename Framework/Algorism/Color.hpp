#pragma once
#include "geommath.hpp"
#include "portable.hpp"

namespace My {
template <class T>
using RGB  = Vector<T, 3>;
using RGB8 = RGB<unsigned char> ;
using RGBf = RGB<float> ;
using YCbCrf = RGB<float> ;

template <class T>
inline __device__ RGB8 QuantizeUnsigned8Bits(RGB<T> rgb) {
    RGB8 result;
    rgb = clamp(rgb, (T)0.0, (T)1.0);
    rgb = rgb * (T)255.999;

    result = { 
        static_cast<unsigned char>(rgb[0]),
        static_cast<unsigned char>(rgb[1]),
        static_cast<unsigned char>(rgb[2])
    };

    return result;
}
}

#pragma once
#include "geommath.hpp"
#include "portable.hpp"
#include "Color.hpp"

namespace My {
const Matrix<float, 4, 4> RGB2YCbCr = {{{0.299f, -0.168736f, 0.5f, 0.0f},
                                        {0.587f, -0.331264f, -0.418688f, 0.0f},
                                        {0.114f, 0.5f, -0.081312f, 0.0f},
                                        {0.0f, 128.0f, 128.0f, 0.0f}}};

const Matrix<float, 4, 4> YCbCr2RGB = {{
    {1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, -0.344136f, 1.772f, 0.0f},
    {1.402f, -0.714136f, 0.0f, 0.0f},
    {-179.456f, 135.458816f, -226.816f, 0.0f},
}};

inline YCbCrf ConvertRGB2YCbCr(const RGBf& rgb) {
    Vector4f result({rgb[0], rgb[1], rgb[2], 1.0f});
    Transform(result, RGB2YCbCr);
    return YCbCrf({std::clamp<float>(result[0] + 0.5f, 0.0f, 255.0f),
                   std::clamp<float>(result[1] + 0.5f, 0.0f, 255.0f),
                   std::clamp<float>(result[2] + 0.5f, 0.0f, 255.0f)});
}

inline RGBf ConvertYCbCr2RGB(const YCbCrf& ycbcr) {
    Vector4f result({ycbcr[0], ycbcr[1], ycbcr[2], 1.0f});
    Transform(result, YCbCr2RGB);
    return RGBf({std::clamp<float>(result[0] + 0.5f, 0.0f, 255.0f),
                 std::clamp<float>(result[1] + 0.5f, 0.0f, 255.0f),
                 std::clamp<float>(result[2] + 0.5f, 0.0f, 255.0f)});
}

inline __device__ RGBf Linear2SRGB( const RGBf& c ) {
    float invGamma = 1.0f / 2.4f;
    RGBf powed    = pow(c, invGamma);
    return RGBf({
        c[0] < 0.0031308f ? 12.92f * c[0] : 1.055f * powed[0] - 0.055f,
        c[1] < 0.0031308f ? 12.92f * c[1] : 1.055f * powed[1] - 0.055f,
        c[2] < 0.0031308f ? 12.92f * c[2] : 1.055f * powed[2] - 0.055f});
}

template <class T>
inline __device__ RGB<T> Linear2SRGB( const RGB<T>& c ) {
    T invGamma = (T)1.0 / (T)2.4;
    RGB<T> powed    = pow(c, invGamma);
    return RGB<T>({
        c[0] < (T)0.0031308 ? (T)12.92 * c[0] : (T)1.055 * powed[0] - (T)0.055,
        c[1] < (T)0.0031308 ? (T)12.92 * c[1] : (T)1.055 * powed[1] - (T)0.055,
        c[2] < (T)0.0031308 ? (T)12.92 * c[2] : (T)1.055 * powed[2] - (T)0.055});
}

}  // namespace My

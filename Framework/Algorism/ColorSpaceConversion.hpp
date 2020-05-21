#pragma once
#include "geommath.hpp"
#include "portable.hpp"

namespace My {
typedef Vector<float, 3> RGBf;
typedef Vector<float, 3> YCbCrf;

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
}  // namespace My

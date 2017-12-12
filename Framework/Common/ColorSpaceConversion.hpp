#include "geommath.hpp"
#include "portable.hpp"

namespace My {
    typedef Vector3Type<float> RGB;
    typedef Vector3Type<float> YCbCr;
    typedef Vector3Type<uint8_t> RGBu8;
    typedef Vector3Type<uint8_t> YCbCru8;

    const Matrix<float, 4, 4> RGB2YCbCr = {{{
        { 0.299f, -0.168736f,  0.5f     ,    0.0f },
        { 0.587f, -0.331264f, -0.418688f,    0.0f },
        { 0.114f,  0.5f     , -0.081312f,    0.0f },
        { 0.0f  ,  128.0f   ,  128.0f   ,    0.0f }
    }}};

    const Matrix<float, 4, 4> YCbCr2RGB = {{{
        {      1.0f,    1.0f     ,    1.0f  ,    0.0f },
        {      0.0f, -  0.344136f,    1.772f,    0.0f },
        {    1.402f, -  0.714136f,    0.0f  ,    0.0f },
        { -179.456f,  135.458816f, -226.816f,    0.0f },
    }}};

    YCbCr ConvertRGB2YCbCr(const RGB& rgb)
    {
        Vector4f result (rgb.r, rgb.g, rgb.b, 1.0f);
        ispc::Transform(result, RGB2YCbCr);
        return YCbCr( std::clamp<float>(result.x + 0.5f, 0.0f, 255.0f), 
                      std::clamp<float>(result.y + 0.5f, 0.0f, 255.0f), 
                      std::clamp<float>(result.z + 0.5f, 0.0f, 255.0f));
    }

    RGB ConvertYCbCr2RGB(const YCbCr& ycbcr)
    {
        Vector4f result (ycbcr.x, ycbcr.y, ycbcr.z, 1.0f);
        ispc::Transform(result, YCbCr2RGB);
        return RGB( std::clamp<float>(result.r + 0.5f, 0.0f, 255.0f), 
                    std::clamp<float>(result.g + 0.5f, 0.0f, 255.0f), 
                    std::clamp<float>(result.b + 0.5f, 0.0f, 255.0f));
    }

    YCbCru8 ConvertRGB2YCbCr(const RGBu8& rgb)
    {
        Vector4f result (rgb.r, rgb.g, rgb.b, 1.0f);
        ispc::Transform(result, RGB2YCbCr);
        return YCbCru8( (uint8_t)std::clamp<float>(result.x + 0.5f, 0.0f, 255.0f), 
                        (uint8_t)std::clamp<float>(result.y + 0.5f, 0.0f, 255.0f), 
                        (uint8_t)std::clamp<float>(result.z + 0.5f, 0.0f, 255.0f));
    }

    RGBu8 ConvertYCbCr2RGB(const YCbCru8& ycbcr)
    {
        Vector4f result (ycbcr.x, ycbcr.y, ycbcr.z, 1.0f);
        ispc::Transform(result, YCbCr2RGB);
        return RGBu8( (uint8_t)std::clamp<float>(result.r + 0.5f, 0.0f, 255.0f), 
                      (uint8_t)std::clamp<float>(result.g + 0.5f, 0.0f, 255.0f), 
                      (uint8_t)std::clamp<float>(result.b + 0.5f, 0.0f, 255.0f));
    }
}



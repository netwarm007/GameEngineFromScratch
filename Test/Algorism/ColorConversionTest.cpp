#include <cmath>
#include <iostream>

#include "ColorSpaceConversion.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static bool Near(float a, float b, float eps) { return std::fabs(a - b) < eps; }

int main() {
    // RGB -> YCbCr -> RGB round trips within quantization error.
    const RGBf samples[] = {{64, 35, 17}, {255, 255, 255}, {0, 0, 0},
                            {200, 10, 90}, {12, 220, 140}};
    for (const auto& rgb : samples) {
        auto back = ConvertYCbCr2RGB(ConvertRGB2YCbCr(rgb));
        for (int i = 0; i < 3; i++) CHECK(Near(back[i], rgb[i], 2.0f));
    }

    // Pure white maps to full luma and neutral chroma.
    auto white = ConvertRGB2YCbCr(RGBf({255, 255, 255}));
    CHECK(Near(white[0], 255.0f, 1.0f));
    CHECK(Near(white[1], 128.0f, 1.0f) && Near(white[2], 128.0f, 1.0f));

    // Results are clamped to the 8-bit range.
    auto clamped = ConvertYCbCr2RGB(YCbCrf({255, 0, 255}));
    for (int i = 0; i < 3; i++) CHECK(clamped[i] >= 0.0f && clamped[i] <= 255.0f);

    // sRGB transfer: linear segment, power segment, endpoints.
    auto lo = Linear2SRGB(RGBf({0.001f, 0.0f, 0.0f}));
    CHECK(Near(lo[0], 12.92f * 0.001f, 1e-6f) && lo[1] == 0.0f);
    auto hi = Linear2SRGB(RGBf({1.0f, 0.5f, 0.2f}));
    CHECK(Near(hi[0], 1.0f, 1e-4f));
    CHECK(Near(hi[1], 0.7354f, 1e-3f));
    CHECK(Near(hi[2], 0.4845f, 1e-3f));

    auto generic = Linear2SRGB(RGB<double>({1.0, 0.5, 0.001}));
    CHECK(Near((float)generic[0], 1.0f, 1e-4f));
    CHECK(Near((float)generic[1], 0.7354f, 1e-3f));
    CHECK(Near((float)generic[2], 0.01292f, 1e-5f));

    return 0;
}

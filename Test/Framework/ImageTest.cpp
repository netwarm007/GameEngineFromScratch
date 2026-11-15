#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <utility>

#include "Image.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {
bool Near(float actual, float expected) {
    return std::abs(actual - expected) < 1e-5f;
}

Image MakeImage(uint32_t width, uint32_t height, uint16_t bitcount,
               size_t pitch, PIXEL_FORMAT format) {
    Image image;
    image.Width = width;
    image.Height = height;
    image.bitcount = bitcount;
    image.pitch = pitch;
    image.data_size = pitch * height;
    image.pixel_format = format;
    image.data = new uint8_t[image.data_size]{};
    return image;
}
}  // namespace

int main() {
    auto rgba = MakeImage(1, 1, 32, 4, PIXEL_FORMAT::RGBA8);
    rgba.data[0] = 10;
    rgba.data[1] = 20;
    rgba.data[2] = 30;
    rgba.data[3] = 40;
    CHECK(rgba.GetR(0, 0) == 10);
    CHECK(rgba.GetG(0, 0) == 20);
    CHECK(rgba.GetB(0, 0) == 30);
    CHECK(rgba.GetA(0, 0) == 40);
    CHECK(Near(rgba.GetX(0, 0), 10.0f / 255.0f));
    CHECK(Near(rgba.GetY(0, 0), 20.0f / 255.0f));
    CHECK(Near(rgba.GetZ(0, 0), 30.0f / 255.0f));
    CHECK(Near(rgba.GetW(0, 0), 40.0f / 255.0f));
    CHECK(rgba.GetX(1, 0) == 0 && rgba.GetW(1, 0) == 0xFF);
    rgba.SetR(0, 0, 101);
    rgba.SetG(0, 0, 102);
    rgba.SetB(0, 0, 103);
    rgba.SetA(0, 0, 104);
    CHECK(rgba.data[0] == 101 && rgba.data[1] == 102 &&
          rgba.data[2] == 103 && rgba.data[3] == 104);
    rgba.SetR(1, 0, 255);
    CHECK(rgba.data[0] == 101);
    CHECK(to_unorm(-0.5f) == 0 && to_unorm(0.5f) == 128 &&
          to_unorm(2.0f) == 255);

    auto gray = MakeImage(1, 1, 8, 1, PIXEL_FORMAT::R8);
    gray.data[0] = 128;
    CHECK(Near(gray.GetX(0, 0), 128.0f / 255.0f));
    CHECK(gray.GetY(0, 0) == 0 && gray.GetZ(0, 0) == 0);
    CHECK(gray.GetW(0, 0) == 1.0f);
    gray.SetG(0, 0, 7);
    CHECK(gray.data[0] == 128);

    auto rgb8 = MakeImage(2, 1, 24, 6, PIXEL_FORMAT::RGB8);
    const uint8_t rgb8_values[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(rgb8.data, rgb8_values, sizeof(rgb8_values));
    rgb8.mipmaps.emplace_back(2, 1, 6, 3, 6);
    adjust_image(rgb8);
    const uint8_t rgba8_expected[] = {1, 2, 3, 255, 4, 5, 6, 255};
    CHECK(rgb8.pixel_format == PIXEL_FORMAT::RGBA8 && rgb8.bitcount == 32);
    CHECK(rgb8.pitch == 8 && rgb8.data_size == 8);
    CHECK(std::memcmp(rgb8.data, rgba8_expected, sizeof(rgba8_expected)) == 0);
    CHECK(rgb8.mipmaps[0].pitch == 8 && rgb8.mipmaps[0].offset == 4 &&
          rgb8.mipmaps[0].data_size == 8);

    auto rgb16 = MakeImage(1, 1, 48, 6, PIXEL_FORMAT::RGB16);
    const uint16_t rgb16_values[] = {0x3C00, 0x4000, 0x4200};
    std::memcpy(rgb16.data, rgb16_values, sizeof(rgb16_values));
    adjust_image(rgb16);
    uint16_t alpha16 = 0;
    std::memcpy(&alpha16, rgb16.data + 6, sizeof(alpha16));
    CHECK(rgb16.pixel_format == PIXEL_FORMAT::RGBA16 && rgb16.bitcount == 64);
    CHECK(rgb16.pitch == 8 && rgb16.data_size == 8 && alpha16 == 0x3C00);
    CHECK(Near(rgb16.GetW(0, 0), 1.0f));

    auto rgb32 = MakeImage(1, 1, 96, 12, PIXEL_FORMAT::RGB32);
    const float rgb32_values[] = {0.25f, 0.5f, 0.75f};
    std::memcpy(rgb32.data, rgb32_values, sizeof(rgb32_values));
    adjust_image(rgb32);
    float alpha32 = 0;
    std::memcpy(&alpha32, rgb32.data + 12, sizeof(alpha32));
    CHECK(rgb32.pixel_format == PIXEL_FORMAT::RGBA32 && rgb32.bitcount == 128);
    CHECK(rgb32.pitch == 16 && rgb32.data_size == 16 && alpha32 == 1.0f);

    auto compressed = MakeImage(1, 1, 24, 3, PIXEL_FORMAT::RGB8);
    compressed.compressed = true;
    const auto* compressed_data = compressed.data;
    adjust_image(compressed);
    CHECK(compressed.data == compressed_data && compressed.pitch == 3);
    CHECK(compressed.pixel_format == PIXEL_FORMAT::RGB8);

    Image move_source = MakeImage(1, 1, 8, 1, PIXEL_FORMAT::R8);
    move_source.data[0] = 77;
    Image moved(std::move(move_source));
    CHECK(move_source.data == nullptr);
    CHECK(moved.data && moved.data[0] == 77 && moved.Width == 1);

    auto assignment_source = MakeImage(2, 1, 8, 2, PIXEL_FORMAT::R8);
    assignment_source.data[0] = 88;
    auto assignment_target = MakeImage(1, 1, 8, 1, PIXEL_FORMAT::R8);
    assignment_target = std::move(assignment_source);
    CHECK(assignment_source.data == nullptr);
    CHECK(assignment_target.data && assignment_target.data[0] == 88);
    CHECK(assignment_target.Width == 2 && assignment_target.data_size == 2);

    std::ostringstream description;
    description << COMPRESSED_FORMAT::DXT1;
    CHECK(description.str() == "DXT1\n");

    return 0;
}

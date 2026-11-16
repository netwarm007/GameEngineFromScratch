#include <cstring>
#include <iostream>

#include "BMP.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {
Buffer MakeBmp(int32_t width, int32_t height, uint16_t bit_count,
               size_t payload_size = 0) {
    const size_t row_pitch =
        ((static_cast<size_t>(width) * bit_count + 31) / 32) * 4;
    const size_t abs_height =
        static_cast<size_t>(height < 0 ? -static_cast<int64_t>(height) : height);
    if (payload_size == 0) {
        payload_size = row_pitch * abs_height;
    }

    Buffer buffer(BITMAP_FILEHEADER_SIZE + sizeof(BITMAP_HEADER) +
                  payload_size);
    std::memset(buffer.GetData(), 0, buffer.GetDataSize());
    auto* file_header =
        reinterpret_cast<BITMAP_FILEHEADER*>(buffer.GetData());
    file_header->Signature = 0x4D42;
    file_header->Size = static_cast<uint32_t>(buffer.GetDataSize());
    file_header->BitsOffset =
        BITMAP_FILEHEADER_SIZE + sizeof(BITMAP_HEADER);

    auto* bitmap_header = reinterpret_cast<BITMAP_HEADER*>(
        buffer.GetData() + BITMAP_FILEHEADER_SIZE);
    bitmap_header->HeaderSize = sizeof(BITMAP_HEADER);
    bitmap_header->Width = width;
    bitmap_header->Height = height;
    bitmap_header->Planes = 1;
    bitmap_header->BitCount = bit_count;
    bitmap_header->SizeImage = static_cast<uint32_t>(payload_size);
    return buffer;
}
}  // namespace

int main() {
    BmpParser parser;
    auto bmp24 = MakeBmp(2, 2, 24);
    auto* pixels = bmp24.GetData() + BITMAP_FILEHEADER_SIZE +
                   sizeof(BITMAP_HEADER);
    const uint8_t bottom_row[] = {1, 2, 3, 4, 5, 6, 0, 0};
    const uint8_t top_row[] = {7, 8, 9, 10, 11, 12, 0, 0};
    std::memcpy(pixels, bottom_row, sizeof(bottom_row));
    std::memcpy(pixels + sizeof(bottom_row), top_row, sizeof(top_row));

    auto image = parser.Parse(bmp24);
    CHECK(image.Width == 2 && image.Height == 2);
    CHECK(image.bitcount == 32 && image.bitdepth == 8);
    CHECK(image.pixel_format == PIXEL_FORMAT::RGBA8);
    CHECK(image.pitch == 8 && image.data_size == 16);
    CHECK(image.mipmaps.size() == 1);
    CHECK(image.mipmaps[0].Width == 2 && image.mipmaps[0].Height == 2);
    const uint8_t expected[] = {9, 8, 7, 255, 12, 11, 10, 255,
                                3, 2, 1, 255, 6, 5, 4, 255};
    CHECK(std::memcmp(image.data, expected, sizeof(expected)) == 0);

    auto top_down_bmp = MakeBmp(1, -2, 32);
    auto* top_down_pixels = top_down_bmp.GetData() + BITMAP_FILEHEADER_SIZE +
                            sizeof(BITMAP_HEADER);
    const uint8_t top_down_data[] = {13, 14, 15, 16, 17, 18, 19, 20};
    std::memcpy(top_down_pixels, top_down_data, sizeof(top_down_data));
    auto top_down_image = parser.Parse(top_down_bmp);
    CHECK(top_down_image.Width == 1 && top_down_image.Height == 2);
    const uint8_t expected_top_down[] = {15, 14, 13, 16, 19, 18, 17, 20};
    CHECK(std::memcmp(top_down_image.data, expected_top_down,
                      sizeof(expected_top_down)) == 0);

    Buffer truncated;
    CHECK(!parser.Parse(truncated).data);
    Buffer short_header(53);
    CHECK(!parser.Parse(short_header).data);

    Buffer invalid_signature(BITMAP_FILEHEADER_SIZE + sizeof(BITMAP_HEADER));
    std::memset(invalid_signature.GetData(), 0, invalid_signature.GetDataSize());
    CHECK(!parser.Parse(invalid_signature).data);

    auto truncated_pixels = MakeBmp(2, 2, 24, 8);
    CHECK(!parser.Parse(truncated_pixels).data);

    auto unsupported = MakeBmp(1, 1, 8);
    CHECK(!parser.Parse(unsupported).data);

    return 0;
}

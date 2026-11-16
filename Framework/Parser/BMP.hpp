#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include "IImageParser.hpp"

namespace My {
#pragma pack(push, 1)
typedef struct _BITMAP_FILEHEADER {
    uint16_t Signature;
    uint32_t Size;
    uint32_t Reserved;
    uint32_t BitsOffset;
} BITMAP_FILEHEADER;

#define BITMAP_FILEHEADER_SIZE 14

typedef struct _BITMAP_HEADER {
    uint32_t HeaderSize;
    int32_t Width;
    int32_t Height;
    uint16_t Planes;
    uint16_t BitCount;
    uint32_t Compression;
    uint32_t SizeImage;
    int32_t PelsPerMeterX;
    int32_t PelsPerMeterY;
    uint32_t ClrUsed;
    uint32_t ClrImportant;
} BITMAP_HEADER;
#pragma pack(pop)

class BmpParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        constexpr size_t minimum_header_size =
            BITMAP_FILEHEADER_SIZE + sizeof(BITMAP_HEADER);
        const size_t buffer_size = buf.GetDataSize();
        if (buf.GetData() == nullptr || buffer_size < minimum_header_size) {
            std::cerr << "Truncated BMP header" << std::endl;
            return img;
        }

        const auto* pFileHeader =
            reinterpret_cast<const BITMAP_FILEHEADER*>(buf.GetData());
        const auto* pBmpHeader = reinterpret_cast<const BITMAP_HEADER*>(
            reinterpret_cast<const uint8_t*>(buf.GetData()) +
            BITMAP_FILEHEADER_SIZE);
        if (pFileHeader->Signature != 0x4D42 /* 'B''M' */) {
            std::cerr << "File is not a BMP file" << std::endl;
            return img;
        }

        const int64_t signed_height = pBmpHeader->Height;
        if (pBmpHeader->HeaderSize < sizeof(BITMAP_HEADER) ||
            pBmpHeader->HeaderSize > buffer_size - BITMAP_FILEHEADER_SIZE ||
            pBmpHeader->Width <= 0 || signed_height == 0 ||
            signed_height == (std::numeric_limits<int32_t>::min)() ||
            pBmpHeader->Planes != 1 ||
            (pBmpHeader->BitCount != 24 && pBmpHeader->BitCount != 32) ||
            pBmpHeader->Compression != 0) {
            std::cerr << "Invalid or unsupported BMP header" << std::endl;
            return img;
        }

        const uint64_t height =
            static_cast<uint64_t>(signed_height < 0 ? -signed_height
                                                    : signed_height);
        const uint64_t width = static_cast<uint32_t>(pBmpHeader->Width);
        const uint64_t row_bits = width * pBmpHeader->BitCount;
        const uint64_t source_pitch = ((row_bits + 31) / 32) * 4;
        const size_t pixel_offset = pFileHeader->BitsOffset;
        if (pixel_offset < static_cast<size_t>(BITMAP_FILEHEADER_SIZE) +
                               pBmpHeader->HeaderSize ||
            pixel_offset > buffer_size ||
            source_pitch > (std::numeric_limits<size_t>::max)() ||
            source_pitch > (buffer_size - pixel_offset) ||
            height > (buffer_size - pixel_offset) / source_pitch ||
            width > (std::numeric_limits<size_t>::max)() / 4 ||
            height > (std::numeric_limits<size_t>::max)() /
                         (static_cast<size_t>(width) * 4)) {
            std::cerr << "Truncated or invalid BMP pixel data" << std::endl;
            return img;
        }

        img.Width = static_cast<uint32_t>(width);
        img.Height = static_cast<uint32_t>(height);
        img.bitcount = 32;
        img.bitdepth = 8;
        img.pixel_format = PIXEL_FORMAT::RGBA8;
        img.pitch = static_cast<size_t>(width) * 4;
        img.data_size = img.pitch * img.Height;
        img.data = new uint8_t[img.data_size];

        const auto* source_data =
            reinterpret_cast<const uint8_t*>(buf.GetData()) + pixel_offset;
        const size_t source_bytes_per_pixel = pBmpHeader->BitCount / 8;
        for (uint32_t y = 0; y < img.Height; y++) {
            const uint32_t source_y =
                signed_height > 0 ? img.Height - y - 1 : y;
            for (uint32_t x = 0; x < img.Width; x++) {
                const auto* src =
                    source_data + source_pitch * source_y +
                    source_bytes_per_pixel * x;
                auto* dst = img.data + img.pitch * y + 4 * x;
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = source_bytes_per_pixel == 4 ? src[3] : 0xFF;
            }
        }

        img.mipmaps.emplace_back(img.Width, img.Height, img.pitch, 0,
                                 img.data_size);

        return img;
    }
};
}  // namespace My

#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>

#include "IImageParser.hpp"

#define MAGIC_FILE_CONSTANT 0x5CA1AB13

namespace My {
struct astc_header {
    uint8_t magic[4];
    uint8_t block_x;
    uint8_t block_y;
    uint8_t block_z;
    uint8_t dim_x[3];
    uint8_t dim_y[3];
    uint8_t dim_z[3];
};

struct astc_image {
    astc_header header;
    uint8_t data;

    [[nodiscard]] uint32_t DimX() const {
        return (uint32_t)header.dim_x[0] + (header.dim_x[1] << 8) +
               (header.dim_x[2] << 16);
    }

    [[nodiscard]] uint32_t DimY() const {
        return (uint32_t)header.dim_y[0] + (header.dim_y[1] << 8) +
               (header.dim_y[2] << 16);
    }

    [[nodiscard]] uint32_t DimZ() const {
        return (uint32_t)header.dim_z[0] + (header.dim_z[1] << 8) +
               (header.dim_z[2] << 16);
    }
};

class AstcParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        if (buf.GetData() == nullptr ||
            buf.GetDataSize() < sizeof(astc_header)) {
            std::cerr << "Truncated ASTC header" << std::endl;
            return img;
        }

        const astc_image* pImage =
            reinterpret_cast<const astc_image*>(buf.GetData());
        if (pImage->header.magic[0] == 0x13 &&
            pImage->header.magic[1] == 0xAB &&
            pImage->header.magic[2] == 0xA1 &&
            pImage->header.magic[3] == 0x5c) {
#if DUMP_DETAILS
            std::cerr << "Asset is ASTC compressed image" << std::endl;
            std::cerr << "ASTC Header" << std::endl;
            std::cerr << "----------------------------" << std::endl;
            fprintf(stderr, "Image block size: (%d x %d x %d)\n",
                    pImage->header.block_x, pImage->header.block_y,
                    pImage->header.block_z);
#endif

            uint32_t type_index = ((uint32_t)pImage->header.block_x << 16) | ((uint32_t)pImage->header.block_y << 8) | pImage->header.block_z;
            auto it = fmt_lut.find(type_index);
            img.compress_format =
                (it != fmt_lut.end()) ? it->second
                                      : COMPRESSED_FORMAT::UNKNOWN;

            const uint32_t width = pImage->DimX();
            const uint32_t height = pImage->DimY();
            const uint32_t depth = pImage->DimZ();
            const uint32_t block_x = pImage->header.block_x;
            const uint32_t block_y = pImage->header.block_y;
            const uint32_t block_z = pImage->header.block_z;
            if (width == 0 || height == 0 || depth == 0 ||
                width > kMaxDimension || height > kMaxDimension ||
                block_x == 0 || block_x > 12 || block_y == 0 ||
                block_y > 12 || block_z == 0 || block_z > 12) {
                std::cerr << "Invalid ASTC image dimensions" << std::endl;
                return img;
            }

            const size_t data_size = buf.GetDataSize() - sizeof(astc_header);
            // Every ASTC block is 16 bytes; the file must hold all blocks.
            const uint64_t blocks_x = (width + block_x - 1) / block_x;
            const uint64_t blocks_y = (height + block_y - 1) / block_y;
            const uint64_t blocks_z = (depth + block_z - 1) / block_z;
            if (blocks_x * blocks_y * blocks_z * 16 > (uint64_t)data_size) {
                std::cerr << "Truncated ASTC block data" << std::endl;
                return img;
            }

            img.Width = width;
            img.Height = height;
            img.data_size = data_size;
            img.data = new uint8_t[img.data_size];
            img.compressed = true;

            memcpy(img.data, &pImage->data, img.data_size);
        }

        return img;
    }

   private:
    static constexpr uint32_t kMaxDimension = 0x10000;  // ASTC spec max

    protected:
    static std::map<uint32_t, COMPRESSED_FORMAT> fmt_lut;
};

std::map<uint32_t, COMPRESSED_FORMAT> AstcParser::fmt_lut = {
    {0x040401, COMPRESSED_FORMAT::ASTC_4x4},
    {0x060601, COMPRESSED_FORMAT::ASTC_6x6},
    {0x080801, COMPRESSED_FORMAT::ASTC_6x6}
};

}  // namespace My
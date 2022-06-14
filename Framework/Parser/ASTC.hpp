#pragma once
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

    uint32_t DimX() {
        return (uint32_t)header.dim_x[0] + (header.dim_x[1] << 8) +
               (header.dim_x[2] << 16);
    }

    uint32_t DimY() {
        return (uint32_t)header.dim_y[0] + (header.dim_y[1] << 8) +
               (header.dim_y[2] << 16);
    }

    uint32_t DimZ() {
        return (uint32_t)header.dim_z[0] + (header.dim_z[1] << 8) +
               (header.dim_z[2] << 16);
    }
};

class AstcParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        astc_image* pImage = reinterpret_cast<astc_image*>(buf.GetData());
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

            img.Width = pImage->DimX();
            img.Height = pImage->DimY();
            img.data_size = (size_t)buf.GetDataSize() - sizeof(astc_header);
            img.data = new uint8_t[img.data_size];
            img.compressed = true;
            uint32_t type_index = ((uint32_t)pImage->header.block_x << 16) | ((uint32_t)pImage->header.block_y << 8) | pImage->header.block_z;
            auto it = fmt_lut.find(type_index);
            if (it != fmt_lut.end()) {
                img.compress_format = it->second;
            } else {
                img.compress_format = COMPRESSED_FORMAT::UNKNOWN;
            }

            memcpy(img.data, &pImage->data, img.data_size);
        }

        return img;
    }

    protected:
    static std::map<uint32_t, COMPRESSED_FORMAT> fmt_lut;
};
}  // namespace My
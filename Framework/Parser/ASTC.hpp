#pragma once
#include <iostream>

#include "IImageParser.hpp"

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
            std::cerr << "Asset is ASTC compressed image" << std::endl;
            std::cerr << "ASTC Header" << std::endl;
            std::cerr << "----------------------------" << std::endl;
            fprintf(stderr, "Image block size: (%d x %d x %d)\n",
                    pImage->header.block_x, pImage->header.block_y,
                    pImage->header.block_z);

            img.Width = pImage->DimX();
            img.Height = pImage->DimY();
            img.data_size = (size_t)buf.GetDataSize() - sizeof(astc_header);
            img.data = new uint8_t[img.data_size];
            img.compressed = true;
            img.compress_format = COMPRESSED_FORMAT::ASTC;

            memcpy(img.data, &pImage->data, img.data_size);
        }

        return img;
    }
};
}  // namespace My
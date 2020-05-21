#pragma once
#include <iostream>

#include "ImageParser.hpp"

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
        const auto* pFileHeader =
            reinterpret_cast<const BITMAP_FILEHEADER*>(buf.GetData());
        const auto* pBmpHeader = reinterpret_cast<const BITMAP_HEADER*>(
            reinterpret_cast<const uint8_t*>(buf.GetData()) +
            BITMAP_FILEHEADER_SIZE);
        if (pFileHeader->Signature == 0x4D42 /* 'B''M' */) {
            std::cerr << "Asset is Windows BMP file" << std::endl;
            std::cerr << "BMP Header" << std::endl;
            std::cerr << "----------------------------" << std::endl;
            std::cerr << "File Size: " << pFileHeader->Size << std::endl;
            std::cerr << "Data Offset: " << pFileHeader->BitsOffset
                      << std::endl;
            std::cerr << "Image Width: " << pBmpHeader->Width << std::endl;
            std::cerr << "Image Height: " << pBmpHeader->Height << std::endl;
            std::cerr << "Image Planes: " << pBmpHeader->Planes << std::endl;
            std::cerr << "Image BitCount: " << pBmpHeader->BitCount
                      << std::endl;
            std::cerr << "Image Compression: " << pBmpHeader->Compression
                      << std::endl;
            std::cerr << "Image Size: " << pBmpHeader->SizeImage << std::endl;

            img.Width = pBmpHeader->Width;
            img.Height = pBmpHeader->Height;
            img.bitcount = 32;
            auto byte_count = img.bitcount >> 3;
            img.pitch = ((img.Width * byte_count) + 3) & ~3;
            img.data_size = (size_t)img.pitch * img.Height;
            img.data = new uint8_t[img.data_size];

            if (img.bitcount < 24) {
                std::cerr << "Sorry, only true color BMP is supported at now."
                          << std::endl;
            } else {
                const uint8_t* pSourceData =
                    reinterpret_cast<const uint8_t*>(buf.GetData()) +
                    pFileHeader->BitsOffset;
                for (int32_t y = img.Height - 1; y >= 0; y--) {
                    for (uint32_t x = 0; x < img.Width; x++) {
                        auto dst = reinterpret_cast<R8G8B8A8Unorm*>(
                            reinterpret_cast<uint8_t*>(img.data) +
                            (ptrdiff_t)img.pitch *
                                ((ptrdiff_t)img.Height - y - 1) +
                            (ptrdiff_t)x * byte_count);
                        auto src = reinterpret_cast<const R8G8B8A8Unorm*>(
                            pSourceData + (ptrdiff_t)img.pitch * y +
                            (ptrdiff_t)x * byte_count);
                        dst->data[2] = src->data[0];
                        dst->data[1] = src->data[1];
                        dst->data[0] = src->data[2];
                        dst->data[3] = src->data[3];
                    }
                }
            }
        }

        img.mipmaps.emplace_back(img.Width, img.Height, img.pitch, 0,
                                 img.data_size);

        return img;
    }
};
}  // namespace My

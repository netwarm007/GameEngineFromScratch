#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <queue>
#include <string>

#include "ImageParser.hpp"
#include "config.h"
#include "portable.hpp"

namespace My {
#pragma pack(push, 1)
struct TGA_FILEHEADER {
    uint8_t IDLength;
    uint8_t ColorMapType;
    uint8_t ImageType;
    uint8_t ColorMapSpec[5];
    uint8_t ImageSpec[10];
};
#pragma pack(pop)

class TgaParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;

        const uint8_t* pData = buf.GetData();
        const uint8_t* pDataEnd = buf.GetData() + buf.GetDataSize();

        std::cerr << "Parsing as TGA file:" << std::endl;

        const auto* pFileHeader =
            reinterpret_cast<const TGA_FILEHEADER*>(pData);
        pData += sizeof(TGA_FILEHEADER);

#ifdef DEBUG
        std::cerr << "ID Length: " << (uint16_t)pFileHeader->IDLength
                  << std::endl;
        std::cerr << "Color Map Type: " << (uint16_t)pFileHeader->ColorMapType
                  << std::endl;
#endif
        if (pFileHeader->ColorMapType) {
            std::cerr << "Unsupported Color Map. Only Type 0 is supported."
                      << std::endl;
            return img;
        }

#ifdef DEBUG
        std::cerr << "Image Type: " << (uint16_t)pFileHeader->ImageType
                  << std::endl;
#endif
        if (pFileHeader->ImageType != 2) {
            std::cerr << "Unsupported Image Type. Only Type 2 is supported."
                      << std::endl;
            return img;
        }

        img.Width =
            (pFileHeader->ImageSpec[5] << 8) + pFileHeader->ImageSpec[4];
        img.Height =
            (pFileHeader->ImageSpec[7] << 8) + pFileHeader->ImageSpec[6];
        uint8_t pixel_depth = pFileHeader->ImageSpec[8];
        uint8_t alpha_depth = (pFileHeader->ImageSpec[9] & 0x0F);
#ifdef DEBUG
        std::cerr << "Image Width: " << img.Width << std::endl;
        std::cerr << "Image Height: " << img.Height << std::endl;
        std::cerr << "Image Pixel Depth: " << (uint16_t)pixel_depth
                  << std::endl;
        std::cerr << "Image Alpha Depth: " << (uint16_t)alpha_depth
                  << std::endl;
#endif
        // skip Image ID
        pData += pFileHeader->IDLength;
        // skip the Color Map. since we assume the Color Map Type is 0,
        // nothing to skip

        // reading the pixel data
        img.bitcount = (alpha_depth ? 32 : 24);
        img.pitch = (img.Width * (img.bitcount >> 3) + 3) &
                    ~3u;  // for GPU address alignment

        img.data_size = (size_t)img.pitch * img.Height;
        img.data = new uint8_t[img.data_size];

        auto* pOut = (uint8_t*)img.data;
        for (decltype(img.Height) i = 0; i < img.Height; i++) {
            for (decltype(img.Width) j = 0; j < img.Width; j++) {
                switch (pixel_depth) {
                    case 15: {
                        assert(alpha_depth == 0);
                        uint16_t color = *(uint16_t*)pData;
                        pData += 2;
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 3) =
                            ((color & 0x7C00) >> 10);  // R
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 3 +
                          1) = ((color & 0x03E0) >> 5);  // G
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 3 +
                          2) = (color & 0x001F);  // B
                    } break;
                    case 16: {
                        assert(alpha_depth == 1);
                        uint16_t color = *(uint16_t*)pData;
                        pData += 2;
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4) =
                            ((color & 0x7C00) >> 10);  // R
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4 +
                          1) = ((color & 0x03E0) >> 5);  // G
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4 +
                          2) = (color & 0x001F);  // B
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4 +
                          3) = ((color & 0x8000) ? 0xFF : 0x00);  // A
                    } break;
                    case 24: {
                        assert(alpha_depth == 0);
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 3 +
                          2) = *pData++;  // B
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 3 +
                          1) = *pData++;  // G
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 3) =
                            *pData++;  // R
                    } break;
                    case 32: {
                        assert(alpha_depth == 8);
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4 +
                          3) = *pData++;  // A
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4 +
                          2) = *pData++;  // B
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4 +
                          1) = *pData++;  // G
                        *(pOut + (ptrdiff_t)img.pitch * i + (ptrdiff_t)j * 4) =
                            *pData++;  // R
                    } break;
                    default:;
                }
            }
        }

        assert(pData <= pDataEnd);

        img.mipmaps.emplace_back(img.Width, img.Height, img.pitch, 0,
                                 img.data_size);

        return img;
    }
};
}  // namespace My

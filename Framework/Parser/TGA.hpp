#pragma once
#include <cstdio>
#include <iostream>
#include <string>
#include <cassert>
#include <queue>
#include <algorithm>
#include "config.h"
#include "ImageParser.hpp"
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

    class TgaParser : implements ImageParser
    {
    public:
        virtual Image Parse(const Buffer& buf)
        {
            Image img;

            const uint8_t* pData = buf.GetData();
            const uint8_t* pDataEnd = buf.GetData() + buf.GetDataSize();

            std::cout << "Parsing as TGA file:" << std::endl;

            const TGA_FILEHEADER* pFileHeader = reinterpret_cast<const TGA_FILEHEADER*>(pData);
            pData += sizeof(TGA_FILEHEADER);

#ifdef DEBUG
            std::cout << "ID Length: " << (uint16_t)pFileHeader->IDLength << std::endl;
            std::cout << "Color Map Type: " << (uint16_t)pFileHeader->ColorMapType << std::endl;
#endif
            if (pFileHeader->ColorMapType) {
                std::cout << "Unsupported Color Map. Only Type 0 is supported." << std::endl;
                return img;
            }

#ifdef DEBUG
            std::cout << "Image Type: " << (uint16_t)pFileHeader->ImageType << std::endl;
#endif
            if (pFileHeader->ImageType != 2) {
                std::cout << "Unsupported Image Type. Only Type 2 is supported." << std::endl;
                return img;
            }

            img.Width = (pFileHeader->ImageSpec[5] << 8) + pFileHeader->ImageSpec[4];
            img.Height = (pFileHeader->ImageSpec[7] << 8) + pFileHeader->ImageSpec[6];
            uint8_t pixel_depth = pFileHeader->ImageSpec[8];
            uint8_t alpha_depth = (pFileHeader->ImageSpec[9] & 0x0F);
#ifdef DEBUG
            std::cout << "Image Width: " << img.Width << std::endl;
            std::cout << "Image Height: " << img.Height << std::endl;
            std::cout << "Image Pixel Depth: " << (uint16_t)pixel_depth << std::endl;
            std::cout << "Image Alpha Depth: " << (uint16_t)alpha_depth << std::endl;
#endif
            // skip Image ID
            pData += pFileHeader->IDLength;
            // skip the Color Map. since we assume the Color Map Type is 0,
            // nothing to skip

            // reading the pixel data
            img.bitcount = 32;
            img.pitch = (img.Width * (img.bitcount >> 3) + 3) & ~3u; // for GPU address alignment

            img.data_size = img.pitch * img.Height;
            img.data = g_pMemoryManager->Allocate(img.data_size);

            uint8_t* pOut = (uint8_t*)img.data;
            for (auto i = 0; i < img.Height; i++)
            {
                for (auto j = 0; j < img.Width; j++)
                {
                    switch(pixel_depth)
                    {
                        case 15:
                            {
                                uint16_t color = *(uint16_t*)pData;
                                pData += 2;
                                *(pOut + img.pitch * i + j * 4) = ((color & 0x7C00) >> 10);    // R
                                *(pOut + img.pitch * i + j * 4 + 1) = ((color & 0x03E0) >> 5); // G
                                *(pOut + img.pitch * i + j * 4 + 2) = (color & 0x001F);        // B
                                *(pOut + img.pitch * i + j * 4 + 3) = 0xFF;                    // A
                            }
                            break;
                        case 16:
                            {
                                uint16_t color = *(uint16_t*)pData;
                                pData += 2;
                                *(pOut + img.pitch * i + j * 4) = ((color & 0x7C00) >> 10);    // R
                                *(pOut + img.pitch * i + j * 4 + 1) = ((color & 0x03E0) >> 5); // G
                                *(pOut + img.pitch * i + j * 4 + 2) = (color & 0x001F);        // B
                                *(pOut + img.pitch * i + j * 4 + 3) = ((color & 0x8000)?0xFF:0x00);  // A
                            }
                            break;
                        case 24:
                            {
                                *(pOut + img.pitch * i + j * 4) = *pData;    // R
                                pData++;
                                *(pOut + img.pitch * i + j * 4 + 1) = *pData; // G
                                pData++;
                                *(pOut + img.pitch * i + j * 4 + 2) = *pData; // B
                                pData++;
                                *(pOut + img.pitch * i + j * 4 + 3) = 0xFF;   // A
                            }
                            break;
                        case 32:
                            {
                                *(pOut + img.pitch * i + j * 4) = *pData;    // R
                                pData++;
                                *(pOut + img.pitch * i + j * 4 + 1) = *pData; // G
                                pData++;
                                *(pOut + img.pitch * i + j * 4 + 2) = *pData; // B
                                pData++;
                                *(pOut + img.pitch * i + j * 4 + 3) = *pData; // A
                                pData++;
                            }
                            break;
                        default:
                            ;
                    }
                }
            }

            assert(pData <= pDataEnd);

            return img;
        }
    };
}


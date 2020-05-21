#pragma once
#include <cstdio>
#include <cstring>

#include "ImageParser.hpp"

namespace My {
static inline void rgbe2float(float* red, float* green, float* blue,
                              unsigned char rgbe[4]) {
    float f;

    if (rgbe[3]) { /*nonzero pixel*/
        f = ldexp(1.0f, rgbe[3] - (int)(128 + 8));
        *red = rgbe[0] * f;
        *green = rgbe[1] * f;
        *blue = rgbe[2] * f;
    } else
        *red = *green = *blue = 0.0f;
}

class HdrParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        char* pData = reinterpret_cast<char*>(buf.GetData());
        auto remain_size = buf.GetDataSize();

        if (std::strncmp(pData, "#?RADIANCE\n", sizeof("#?RADIANCE\n")) == 0) {
            std::cerr << "Image File is HDR format" << std::endl;
            pData += sizeof("#?RADIANCE\n");

            // process the header
            while (*pData != '\n') {
                char* p = pData;
                // find the line end
                while (*++p != '\n' && --remain_size > 0) {
                }
                if (remain_size == 0) break;
                assert(*p == '\n');
                *p = '\0';
                if (*pData == '#') {
                    // comment line, just ignore it
                    std::cerr << pData << std::endl;
                } else {
                    // assignments
                    std::cerr << pData << std::endl;
                }

                if (remain_size) {
                    pData = p + 1;
                    remain_size--;
                } else {
                    break;
                }
            }

            // process dimension
            assert(remain_size > 8);

            // bypass '\n'
            pData++;
            remain_size--;

            // find the line end
            char* p = pData;
            while (*++p != '\n' && --remain_size > 0) {
            }
            if (remain_size == 0) assert(0);
            assert(*p == '\n');
            *p = '\0';

            char axis1[2];
            char axis2[2];
            uint32_t dimension1;
            uint32_t dimension2;
            std::sscanf(pData, "%2c %u %2c %u", axis1, &dimension1, axis2,
                        &dimension2);

            if (axis1[1] == 'Y') {
                img.Height = dimension1;
                assert(axis2[1] == 'X');
                img.Width = dimension2;
            } else {
                assert(axis1[1] == 'X');
                img.Width = dimension1;
                assert(axis2[1] == 'X');
                img.Height = dimension2;
            }

            pData = p + 1;
            assert(remain_size);
            remain_size--;

            img.bitcount = 32 * 3;  // float[3]
            img.pitch = (img.bitcount >> 3) * img.Width;
            img.data_size = (size_t)img.pitch * img.Height;
            img.data = new uint8_t[img.data_size];

            // now data section
            assert(remain_size <= (size_t)4 * img.Height * img.Width);
            assert(remain_size % 4 == 0);
            float r{0.0f}, g{0.0f}, b{0.0f};
            auto* pRGBE = reinterpret_cast<unsigned char(*)[4]>(pData);
            auto* pOutData = reinterpret_cast<float(*)[3]>(img.data);
            if ((*pRGBE)[0] == 2 && (*pRGBE)[1] == 2 &&
                (*pRGBE)[2] == img.Width >> 8 &&
                (*pRGBE)[3] == (img.Width & 0xFF)) {
                // the file IS run lenght encoded
                std::cerr << "The file *IS* run-length encoded" << std::endl;
            } else {
                std::cerr << "The file is *NOT* run-length encoded"
                          << std::endl;
                // the file is NOT run lenght encoded
                while (remain_size) {
                    if ((*pRGBE)[0] == 255 && (*pRGBE)[1] == 255 &&
                        (*pRGBE)[2] == 255) {
                        uint8_t repeat_times = (*pRGBE)[3];
                        for (uint8_t i = 0; i < repeat_times; i++) {
                            (*pOutData)[0] = r;
                            (*pOutData)[1] = g;
                            (*pOutData)[2] = b;
                            pOutData++;
                        }

                        remain_size -= 4;
                        pRGBE++;
                    } else {
                        rgbe2float(&r, &g, &b, *pRGBE);
                        (*pOutData)[0] = r;
                        (*pOutData)[1] = g;
                        (*pOutData)[2] = b;

                        remain_size -= 4;
                        pRGBE++;
                        pOutData++;
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
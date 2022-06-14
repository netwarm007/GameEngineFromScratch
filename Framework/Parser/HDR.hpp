#pragma once
#include <cstdio>
#include <cstring>
#include <string_view>

#include "IImageParser.hpp"

namespace My {
static inline void rgbe2float(float* red, float* green, float* blue,
                              const unsigned char rgbe[4]) {
    float f;

    if (rgbe[3]) { /*nonzero pixel*/
        f = ldexp(1.0f, rgbe[3] - (int)(128 + 8));
        *red = rgbe[0] * f;
        *green = rgbe[1] * f;
        *blue = rgbe[2] * f;
    } else
        *red = *green = *blue = 0.0f;
}

static inline void rgbe2float(float *buf, const uint8_t *rgbe, uint32_t width) {
    unsigned char tmp[4];
    for (size_t i = 0; i < width; i++) {
        tmp[0] = rgbe[i];
        tmp[1] = rgbe[i + width];
        tmp[2] = rgbe[i + width * 2];
        tmp[3] = rgbe[i + width * 3];
        rgbe2float(&buf[0], &buf[1], &buf[2], tmp);
        buf += 3;
    }
}

static inline void rgbe2float(float out[3],
                              const unsigned char rgbe[4]) {
    rgbe2float(&out[0], &out[1], &out[2], rgbe);
}

class HdrParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        std::string_view sbuf((char *)buf.GetData(), buf.GetDataSize());

        if (sbuf.starts_with("#?RADIANCE\n")) {
            std::cerr << "Image File is HDR format" << std::endl;
            sbuf.remove_prefix(strlen("#?RADIANCE\n"));

            // process the header
            while (sbuf[0] != '\n') {
                // find the line end
                auto line_end = sbuf.find_first_of('\n') + 1;
                if (line_end == sbuf.npos)
                    assert(false && "file is currputed!\n");
                if (sbuf[0] == '#') {
                    // comment line, just ignore it
                    std::cerr << sbuf.substr(0, line_end) << std::endl;
                } else {
                    // assignments
                    std::cerr << sbuf.substr(0, line_end) << std::endl;
                }

                sbuf.remove_prefix(line_end);
            }

            // process dimension

            // bypass '\n'
            sbuf.remove_prefix(1);

            // find the line end
            auto line_end = sbuf.find_first_of('\n') + 1;
            if (line_end == sbuf.npos) assert(false && "file is currputed!\n");

            char axis1[2];
            char axis2[2];
            uint32_t dimension1;
            uint32_t dimension2;
            std::sscanf(sbuf.data(), "%2c %u %2c %u", axis1, &dimension1, axis2,
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

            sbuf.remove_prefix(line_end);

            img.bitcount = 32 * 3;  // float[3]
            img.bitdepth = 32;
            img.pitch = (img.bitcount >> 3) * img.Width;
            img.pixel_format = PIXEL_FORMAT::RGB32;
            img.is_float = true;
            img.data_size = (size_t)img.pitch * img.Height;
            img.data = new uint8_t[img.data_size];

            // now data section
            assert(sbuf.size() <= (size_t)4 * img.Height * img.Width);
            auto* pRGBE =
                reinterpret_cast<const unsigned char(*)[4]>(sbuf.data());
            auto* pOutData = reinterpret_cast<float(*)[3]>(img.data);

            if (img.Width < 8 || img.Width > 0x7fff) {
                // run length encoding is not allowed so read flat
                for (uint32_t y = 0; y < img.Height; y++) {
                    for (uint32_t x = 0; x < img.Width; x++) {
                        rgbe2float(*pOutData++, *pRGBE++);
                    }
                }
            } else {
                auto scanline = new uint8_t[4 * img.Width];

                for (uint32_t i = 0; i < img.Height; i++) {
                    assert(sbuf.size() >= 4 && "corrupted file");

                    if ((*pRGBE)[0] != 2 || (*pRGBE)[1] != 2 ||
                        (*pRGBE)[2] & 0x80) {
                        // the remain part is not run length encoded
                        for (uint32_t y = i; y < img.Height; y++) {
                            for (uint32_t x = 0; x < img.Width; x++) {
                                rgbe2float(*pOutData++, *pRGBE++);
                            }
                        }
                    } else {
                        assert (((*pRGBE)[2] << 8) + (*pRGBE)[3] == img.Width && "wrong scanline width");
                        sbuf.remove_prefix(4); // move forward by 4 bytes
                        uint8_t *pdat = scanline;
                        for (uint32_t j = 0; j < 4; j++) {
                            uint8_t *pend = scanline + (j + 1) * img.Width;
                            while(pdat < pend) {
                                if (sbuf.size() < 2) assert(false && "corrupted file");
                                if (static_cast<unsigned char>(sbuf[0]) > 128) {
                                    int count = static_cast<unsigned char>(sbuf[0]) - 128;
                                    assert(count && count <= pend - pdat && "corrupted file");
                                    while (count--) {
                                        *pdat++ = sbuf[1];
                                    }
                                } else {
                                    int count = static_cast<unsigned char>(sbuf[0]);
                                    assert(count && count <= pend - pdat && "corrupted file");
                                    *pdat++ = sbuf[1];
                                    if (--count > 0) {
                                        assert(sbuf.size() >= count && "corrupted file");
                                        memcpy(pdat, sbuf.data(), count);
                                        sbuf.remove_prefix(count);
                                        pdat += count;
                                    }
                                }
                                sbuf.remove_prefix(2);
                            }
                        }
                        rgbe2float(reinterpret_cast<float *>(img.data), scanline, img.Width);
                    }
                }

                delete[] scanline;
            }
        }

        img.mipmaps.emplace_back(img.Width, img.Height, img.pitch, 0,
                                 img.data_size);

        return img;
    }
};
}  // namespace My
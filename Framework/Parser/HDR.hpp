#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

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
        if (buf.GetData() == nullptr) return img;
        std::string_view sbuf((char *)buf.GetData(), buf.GetDataSize());

        if (sbuf.starts_with("#?RADIANCE\n")) {
            std::cerr << "Image File is HDR format" << std::endl;
            sbuf.remove_prefix(strlen("#?RADIANCE\n"));

            // process the header
            while (!sbuf.empty() && sbuf[0] != '\n') {
                // find the line end
                auto line_end = sbuf.find_first_of('\n');
                if (line_end == sbuf.npos) {
                    std::cerr << "HDR file is corrupted: unterminated header "
                                 "line!"
                              << std::endl;
                    return img;
                }
                ++line_end;
                // assignments and comment lines, just dump them
                std::cerr << sbuf.substr(0, line_end) << std::endl;

                sbuf.remove_prefix(line_end);
            }

            if (sbuf.empty()) {
                std::cerr << "HDR file is corrupted: truncated header!"
                          << std::endl;
                return img;
            }

            // process dimension

            // bypass '\n'
            sbuf.remove_prefix(1);

            // find the line end
            auto dimension_line_end = sbuf.find_first_of('\n');
            if (dimension_line_end == sbuf.npos) {
                std::cerr << "HDR file is corrupted: unterminated dimension "
                             "line!"
                          << std::endl;
                return img;
            }

            char axis1[2] = {0, 0};
            char axis2[2] = {0, 0};
            uint32_t dimension1 = 0;
            uint32_t dimension2 = 0;
            const std::string dimension_line(
                sbuf.substr(0, dimension_line_end));
            if (std::sscanf(dimension_line.c_str(), "%2c %u %2c %u", axis1,
                            &dimension1, axis2, &dimension2) != 4) {
                std::cerr << "HDR file is corrupted: malformed dimension line!"
                          << std::endl;
                return img;
            }

            auto is_axis = [](char c, char expected) {
                return c == expected ||
                       c == (char)(expected + ('a' - 'A'));
            };
            if (is_axis(axis1[1], 'Y')) {
                img.Height = dimension1;
                if (!is_axis(axis2[1], 'X')) {
                    std::cerr << "HDR file is corrupted: bad dimension axis!"
                              << std::endl;
                    return img;
                }
                img.Width = dimension2;
            } else if (is_axis(axis1[1], 'X')) {
                img.Width = dimension1;
                if (!is_axis(axis2[1], 'Y')) {
                    std::cerr << "HDR file is corrupted: bad dimension axis!"
                              << std::endl;
                    return img;
                }
                img.Height = dimension2;
            } else {
                std::cerr << "HDR file is corrupted: bad dimension axis!"
                          << std::endl;
                return img;
            }

            sbuf.remove_prefix(dimension_line_end + 1);

            // reject absurd dimensions before any allocation
            if (img.Width == 0 || img.Height == 0 ||
                (uint64_t)img.Width * img.Height > kMaxPixels) {
                std::cerr << "HDR file has invalid dimensions: " << img.Width
                          << " x " << img.Height << std::endl;
                return img;
            }

            img.bitcount = 32 * 3;  // float[3]
            img.bitdepth = 32;
            img.pitch = (img.bitcount >> 3) * img.Width;
            img.pixel_format = PIXEL_FORMAT::RGB32;
            img.is_float = true;
            img.data_size = (size_t)img.pitch * img.Height;
            img.data = new uint8_t[img.data_size];

            // now data section
            auto* pRGBE =
                reinterpret_cast<const unsigned char(*)[4]>(sbuf.data());
            auto* pOutData = reinterpret_cast<float(*)[3]>(img.data);

            if (img.Width < 8 || img.Width > 0x7fff) {
                // run length encoding is not allowed so read flat
                if (sbuf.size() < (size_t)4 * img.Width * img.Height) {
                    std::cerr << "HDR file is corrupted: truncated scanline "
                                 "data!"
                              << std::endl;
                    return Image {};
                }
                for (uint32_t y = 0; y < img.Height; y++) {
                    for (uint32_t x = 0; x < img.Width; x++) {
                        rgbe2float(*pOutData++, *pRGBE++);
                    }
                }
            } else {
                std::vector<uint8_t> scanline((size_t)4 * img.Width);

                for (uint32_t i = 0; i < img.Height; i++) {
                    if (sbuf.size() < 4) {
                        std::cerr
                            << "HDR file is corrupted: truncated scanline!"
                            << std::endl;
                        return Image {};
                    }

                    if ((*pRGBE)[0] != 2 || (*pRGBE)[1] != 2 ||
                        (*pRGBE)[2] & 0x80) {
                        // the remain part is not run length encoded
                        if (sbuf.size() <
                            (size_t)4 * (img.Height - i) * img.Width) {
                            std::cerr << "HDR file is corrupted: truncated "
                                         "scanline data!"
                                      << std::endl;
                            return Image {};
                        }
                        for (uint32_t y = i; y < img.Height; y++) {
                            for (uint32_t x = 0; x < img.Width; x++) {
                                rgbe2float(*pOutData++, *pRGBE++);
                            }
                        }
                        break;
                    } else {
                        if ((uint32_t)((*pRGBE)[2] << 8) + (*pRGBE)[3] !=
                            img.Width) {
                            std::cerr << "HDR file is corrupted: wrong "
                                         "scanline width!"
                                      << std::endl;
                            return Image {};
                        }
                        sbuf.remove_prefix(4);  // move forward by 4 bytes
                        pRGBE = reinterpret_cast<const unsigned char(*)[4]>(
                            sbuf.data());
                        uint8_t *pdat = scanline.data();
                        for (uint32_t j = 0; j < 4; j++) {
                            uint8_t *pend = scanline.data() +
                                            (j + 1) * (size_t)img.Width;
                            while (pdat < pend) {
                                if (sbuf.size() < 2) {
                                    std::cerr << "HDR file is corrupted: "
                                                 "truncated scanline!"
                                              << std::endl;
                                    return Image {};
                                }
                                if (static_cast<unsigned char>(sbuf[0]) >
                                    128) {
                                    int count =
                                        static_cast<unsigned char>(sbuf[0]) -
                                        128;
                                    if (count > pend - pdat) {
                                        std::cerr << "HDR file is corrupted: "
                                                     "run overflows scanline!"
                                                  << std::endl;
                                        return Image {};
                                    }
                                    while (count--) {
                                        *pdat++ =
                                            static_cast<unsigned char>(sbuf[1]);
                                    }
                                    sbuf.remove_prefix(2);
                                } else {
                                    int count =
                                        static_cast<unsigned char>(sbuf[0]);
                                    if (count == 0 ||
                                        count > pend - pdat ||
                                        sbuf.size() <
                                            (size_t)count + 1) {
                                        std::cerr << "HDR file is corrupted: "
                                                     "bad literal run!"
                                                  << std::endl;
                                        return Image {};
                                    }
                                    *pdat++ =
                                        static_cast<unsigned char>(sbuf[1]);
                                    if (count > 1) {
                                        memcpy(pdat, sbuf.data() + 2,
                                               count - 1);
                                        pdat += count - 1;
                                    }
                                    sbuf.remove_prefix((size_t)count + 1);
                                }
                                pRGBE =
                                    reinterpret_cast<
                                        const unsigned char(*)[4]>(sbuf.data());
                            }
                        }
                        rgbe2float(
                            reinterpret_cast<float *>(img.data) +
                                (ptrdiff_t)img.pitch / sizeof(float) * i,
                            scanline.data(), img.Width);
                    }
                }
            }
        }

        img.mipmaps.emplace_back(img.Width, img.Height, img.pitch, 0,
                                 img.data_size);

        return img;
    }

   private:
    // sanity cap against absurd file-declared dimensions (64M pixels)
    static constexpr uint64_t kMaxPixels = 64ull * 1024 * 1024;
};
}  // namespace My
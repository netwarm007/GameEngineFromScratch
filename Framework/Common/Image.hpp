#pragma once
#include <cstring>
#include <iostream>
#include <vector>

#include "config.h"
#include "geommath.hpp"

namespace My {
enum class COMPRESSED_FORMAT {
    NONE,
    DXT1,
    DXT2,
    DXT3,
    DXT4,
    DXT5,
    DXT10,
    BC1,
    BC1A,
    BC2,
    BC3,
    BC4S,
    BC4U,
    BC5S,
    BC5U,
    BC6S,
    BC6U,
    BC7S,
    BC7U,
    PVRTC,
    ETC,
    ASTC
};

std::ostream& operator<<(std::ostream& out, COMPRESSED_FORMAT format);

struct Image {
    uint32_t Width{0};
    uint32_t Height{0};
    uint8_t* data{nullptr};
    uint32_t bitcount{0};
    size_t pitch{0};
    size_t data_size{0};
    bool compressed{false};
    bool is_float{false};
    COMPRESSED_FORMAT compress_format{COMPRESSED_FORMAT::NONE};
    struct Mipmap {
        uint32_t Width{0};
        uint32_t Height{0};
        size_t pitch{0};
        size_t offset{0};
        size_t data_size{0};

        Mipmap(uint32_t width, uint32_t height, size_t pitch_, size_t offset_,
               size_t data_size_) {
            Width = width;
            Height = height;
            pitch = pitch_;
            offset = offset_;
            data_size = data_size_;
        }
    };
    std::vector<Mipmap> mipmaps;

    Image() = default;
    Image(const Image& rhs) = delete;  // disable copy contruct
    Image(Image&& rhs) noexcept;
    Image& operator=(const Image& rhs) = delete;  // disable copy assignment
    Image& operator=(Image&& rhs) noexcept;
    ~Image() {
        if (data) delete[] data;
    }
};

std::ostream& operator<<(std::ostream& out, const Image& image);

static inline void adjust_image(Image& image) {
    if (!image.compressed) {
        if (image.bitcount == 24) {
            // DXGI does not have 24bit formats so we have to extend it to 32bit
            auto new_pitch = image.pitch / 3 * 4;
            auto data_size = (size_t)new_pitch * image.Height;
            auto* data = new uint8_t[data_size];
            uint8_t* buf;
            uint8_t* src;
            for (decltype(image.Height) row = 0; row < image.Height; row++) {
                buf = data + (ptrdiff_t)row * new_pitch;
                src = image.data + (ptrdiff_t)row * image.pitch;
                for (decltype(image.Width) col = 0; col < image.Width; col++) {
                    memcpy(buf, src, 3);
                    memset(buf + 3, 0x00, 1);  // set alpha to 0
                    buf += 4;
                    src += 3;
                }
            }

            delete[] image.data;
            image.data = data;
            image.data_size = data_size;
            image.pitch = new_pitch;
            image.bitcount = 32;

            // adjust mipmaps
            for (auto& mip : image.mipmaps) {
                mip.pitch = mip.pitch / 3 * 4;
                mip.offset = mip.offset / 3 * 4;
                mip.data_size = mip.data_size / 3 * 4;
            }
        } else if (image.bitcount == 48) {
            // DXGI does not have 48bit formats so we have to extend it to 64bit
            auto new_pitch = image.pitch / 3 * 4;
            auto data_size = new_pitch * image.Height;
            auto* data = new uint8_t[data_size];
            uint8_t* buf;
            uint8_t* src;
            for (decltype(image.Height) row = 0; row < image.Height; row++) {
                buf = data + (ptrdiff_t)row * new_pitch;
                src = image.data + (ptrdiff_t)row * image.pitch;
                for (decltype(image.Width) col = 0; col < image.Width; col++) {
                    memcpy(buf, src, 6);
                    memset(buf + 6, 0x00, 2);  // set alpha to 0
                    buf += 8;
                    src += 6;
                }
            }

            delete[] image.data;
            image.data = data;
            image.data_size = data_size;
            image.pitch = new_pitch;
            image.bitcount = 64;

            // adjust mipmaps
            for (auto& mip : image.mipmaps) {
                mip.pitch = mip.pitch / 3 * 4;
                mip.offset = mip.offset / 3 * 4;
                mip.data_size = mip.data_size / 3 * 4;
            }
        }
    }
}
}  // namespace My
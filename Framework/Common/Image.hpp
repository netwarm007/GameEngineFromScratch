#pragma once
#include <cstring>
#include <iostream>
#include <vector>

#include "config.h"
#include "geommath.hpp"

namespace My {
enum class COMPRESSED_FORMAT : uint16_t {
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

enum class PIXEL_FORMAT : uint16_t {
    UNKNOWN,
    R8,
    RG8,
    RGB8,
    RGBA8,
    R16,
    RG16,
    RGB16,
    RGBA16,
    R32,
    RG32,
    RGB32,
    RGBA32,
    R10G10B10A2,
    R5G6B5
};

std::ostream& operator<<(std::ostream& out, COMPRESSED_FORMAT format);

struct Image {
    uint32_t Width{0};
    uint32_t Height{0};
    uint16_t bitcount{0};
    uint16_t bitdepth{0};
    size_t pitch{0};
    size_t data_size{0};
    bool compressed{false};
    bool is_float{false};
    bool is_signed{false};
    uint8_t* data{nullptr};
    COMPRESSED_FORMAT compress_format{COMPRESSED_FORMAT::NONE};
    PIXEL_FORMAT pixel_format{PIXEL_FORMAT::UNKNOWN};
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

    uint8_t GetR(int32_t x, int32_t y) const {
        if (x >= Width || y >= Height) return 0;

        switch (pixel_format) {
            case PIXEL_FORMAT::UNKNOWN:
                return 0;
            case PIXEL_FORMAT::R8:
            case PIXEL_FORMAT::RG8:
            case PIXEL_FORMAT::RGB8:
            case PIXEL_FORMAT::RGBA8:
                return *(data + y * pitch + x * (bitcount >> 3));
            case PIXEL_FORMAT::R5G6B5:
                return *(data + y * pitch + x * (bitcount >> 3)) & 0xF8;
            case PIXEL_FORMAT::R16:
            case PIXEL_FORMAT::RG16:
            case PIXEL_FORMAT::RGB16:
            case PIXEL_FORMAT::RGBA16:
            case PIXEL_FORMAT::R32:
            case PIXEL_FORMAT::RG32:
            case PIXEL_FORMAT::RGB32:
            case PIXEL_FORMAT::RGBA32:
            case PIXEL_FORMAT::R10G10B10A2:
                // not supported
                return 0;
            default:
                assert(0);
        }
    }

    uint8_t GetG(int32_t x, int32_t y) const {
        if (x >= Width || y >= Height) return 0;

        switch (pixel_format) {
            case PIXEL_FORMAT::UNKNOWN:
            case PIXEL_FORMAT::R8:
            case PIXEL_FORMAT::R16:
            case PIXEL_FORMAT::R32:
                return GetR(x, y);
            case PIXEL_FORMAT::RG8:
            case PIXEL_FORMAT::RGB8:
            case PIXEL_FORMAT::RGBA8:
                return *(data + y * pitch + x * (bitcount >> 3) + 1);
            case PIXEL_FORMAT::RG16:
            case PIXEL_FORMAT::RGB16:
            case PIXEL_FORMAT::RGBA16:
            case PIXEL_FORMAT::RG32:
            case PIXEL_FORMAT::RGB32:
            case PIXEL_FORMAT::RGBA32:
            case PIXEL_FORMAT::R10G10B10A2:
                // not supported
                return 0;
            case PIXEL_FORMAT::R5G6B5:
                return ((*(data + y * pitch + x * (bitcount >> 3)) & 0x07)
                        << 3) +
                       ((*(data + y * pitch + x * (bitcount >> 3) + 1) &
                         0xE0) >>
                        5);
            default:
                assert(0);
        }
    }

    uint8_t GetB(int32_t x, int32_t y) const {
        if (x >= Width || y >= Height) return 0;

        switch (pixel_format) {
            case PIXEL_FORMAT::UNKNOWN:
            case PIXEL_FORMAT::R8:
            case PIXEL_FORMAT::R16:
            case PIXEL_FORMAT::R32:
                return GetR(x, y);
            case PIXEL_FORMAT::RG8:
                return GetG(x, y);
            case PIXEL_FORMAT::RGB8:
            case PIXEL_FORMAT::RGBA8:
                return *(data + y * pitch + x * (bitcount >> 3) + 2);
                break;
            case PIXEL_FORMAT::RG16:
            case PIXEL_FORMAT::RGB16:
            case PIXEL_FORMAT::RGBA16:
            case PIXEL_FORMAT::RG32:
            case PIXEL_FORMAT::RGB32:
            case PIXEL_FORMAT::RGBA32:
            case PIXEL_FORMAT::R10G10B10A2:
                // not supported
                return 0;
            case PIXEL_FORMAT::R5G6B5:
                return (*(data + y * pitch + x * (bitcount >> 3) + 1) & 0x1F);
            default:
                assert(0);
        }
    }

    uint8_t GetA(int32_t x, int32_t y) const {
        if (x >= Width || y >= Height) return 0;

        switch (pixel_format) {
            case PIXEL_FORMAT::UNKNOWN:
            case PIXEL_FORMAT::R8:
            case PIXEL_FORMAT::R16:
            case PIXEL_FORMAT::R32:
            case PIXEL_FORMAT::RG8:
            case PIXEL_FORMAT::RGB8:
            case PIXEL_FORMAT::R5G6B5:
                return 0;
            case PIXEL_FORMAT::RGBA8:
                return *(data + y * pitch + x * (bitcount >> 3) + 3);
            case PIXEL_FORMAT::RG16:
            case PIXEL_FORMAT::RGB16:
            case PIXEL_FORMAT::RGBA16:
            case PIXEL_FORMAT::RG32:
            case PIXEL_FORMAT::RGB32:
            case PIXEL_FORMAT::RGBA32:
            case PIXEL_FORMAT::R10G10B10A2:
                // not supported
                return 0;
            default:
                assert(0);
        }
    }

    uint8_t GetX(int32_t x, int32_t y) const { return GetR(x, y); }

    uint8_t GetY(int32_t x, int32_t y) const { return GetG(x, y); }

    uint8_t GetZ(int32_t x, int32_t y) const { return GetB(x, y); }

    uint8_t GetW(int32_t x, int32_t y) const { return GetA(x, y); }

    void SaveTGA(const char* filename) const {
        assert(filename != NULL);
        // must end in .tga
        const char* ext = &filename[strlen(filename) - 4];
        assert(!strcmp(ext, ".tga"));
        if (compressed) {
            fprintf(stderr, "SaveTGA is called but the image is compressed.\n");
            return;
        }
        FILE* file = fopen(filename, "wb");
        // misc header information
        for (int i = 0; i < 18; i++) {
            if (i == 2)
                fprintf(file, "%c", 2);
            else if (i == 12)
                fprintf(file, "%c", Width % 256);
            else if (i == 13)
                fprintf(file, "%c", Width / 256);
            else if (i == 14)
                fprintf(file, "%c", Height % 256);
            else if (i == 15)
                fprintf(file, "%c", Height / 256);
            else if (i == 16)
                fprintf(file, "%c", 32);
            else if (i == 17)
                fprintf(file, "%c", 32);
            else
                fprintf(file, "%c", 0);
        }
        // the data
        // flip y so that (0,0) is bottom left corner
        for (int32_t y = Height - 1; y >= 0; y--) {
            for (int32_t x = 0; x < Width; x++) {
                // note reversed order: b, g, r
                fprintf(file, "%c", GetA(x, y));
                fprintf(file, "%c", GetB(x, y));
                fprintf(file, "%c", GetG(x, y));
                fprintf(file, "%c", GetR(x, y));
            }
        }
        fclose(file);
    }
};

std::ostream& operator<<(std::ostream& out, const Image& image);

static inline void adjust_image(Image& image) {
    if (!image.compressed) {
        if (image.pixel_format == PIXEL_FORMAT::RGB8) {
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
            image.pixel_format = PIXEL_FORMAT::RGBA8;

            // adjust mipmaps
            for (auto& mip : image.mipmaps) {
                mip.pitch = mip.pitch / 3 * 4;
                mip.offset = mip.offset / 3 * 4;
                mip.data_size = mip.data_size / 3 * 4;
            }
        } else if (image.pixel_format == PIXEL_FORMAT::RGB16) {
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
            image.pixel_format = PIXEL_FORMAT::RGBA16;

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
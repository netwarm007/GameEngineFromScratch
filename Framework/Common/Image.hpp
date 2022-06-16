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
    BC4,
    BC5,
    BC6H,
    BC7,
    PVRTC,
    ETC,
    ASTC_4x4,
    ASTC_5x4,
    ASTC_5x5,
    ASTC_6x5,
    ASTC_6x6,
    ASTC_8x5,
    ASTC_8x6,
    ASTC_8x8,
    ASTC_10x5,
    ASTC_10x6,
    ASTC_10x8,
    ASTC_10x10,
    ASTC_12x10,
    ASTC_12x12,
    ASTC_3x3x3,
    ASTC_4x3x3,
    ASTC_4x4x3,
    ASTC_4x4x4,
    ASTC_5x4x4,
    ASTC_5x5x4,
    ASTC_5x5x5,
    ASTC_6x5x5,
    ASTC_6x6x5,
    ASTC_6x6x6,
    UNKNOWN
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
                return *(data + y * pitch + x * (bitcount >> 3) + 1);
            case PIXEL_FORMAT::R32:
            case PIXEL_FORMAT::RG32:
            case PIXEL_FORMAT::RGB32:
            case PIXEL_FORMAT::RGBA32:
                return *(data + y * pitch + x * (bitcount >> 3) + 3);
            case PIXEL_FORMAT::R10G10B10A2:
                // not supported
                return 0;
            default:
                assert(0);
        }

        return 0;
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

        return 0;
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

        return 0;
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

        return 0;
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

void adjust_image(Image& image);
}  // namespace My
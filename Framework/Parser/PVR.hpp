
#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>
#include "IImageParser.hpp"

namespace My {
namespace PVR {
enum class Flags : uint32_t { NoFlag = 0, PreMultiplied = 0x02 };

enum class PixelFormat : uint64_t {
    PVRTC_2bpp_RGB = 0,
    PVRTC_2bpp_RGBA = 1,
    PVRTC_4bpp_RGB = 2,
    PVRTC_4bpp_RGBA = 3,
    PVRTC_II_2bpp = 4,
    PVRTC_II_4bpp = 5,
    ETC1 = 6,
    DXT1 = 7,
    DXT2 = 8,
    DXT3 = 9,
    DXT4 = 10,
    DXT5 = 11,
    BC1 = 7,
    BC2 = 9,
    BC3 = 11,
    BC4 = 12,
    BC5 = 13,
    BC6H = 14,
    BC7 = 15,
    UYVY = 16,
    YUY2 = 17,
    BW1bpp = 18,
    R9G9B9E5 = 19,
    RGBG8888 = 20,
    GRGB8888 = 21,
    ETC2_RGB = 22,
    ETC2_RGBA = 23,
    ETC2_RGB_A1 = 24,
    EAC_R11 = 25,
    EAC_RG11 = 26,
    ASTC_4x4 = 27,
    ASTC_5x4 = 28,
    ASTC_5x5 = 29,
    ASTC_6x5 = 30,
    ASTC_6x6 = 31,
    ASTC_8x5 = 32,
    ASTC_8x6 = 33,
    ASTC_8x8 = 34,
    ASTC_10x5 = 35,
    ASTC_10x6 = 36,
    ASTC_10x8 = 37,
    ASTC_10x10 = 38,
    ASTC_12x10 = 39,
    ASTC_12x12 = 40,
    ASTC_3x3x3 = 41,
    ASTC_4x3x3 = 42,
    ASTC_4x4x3 = 43,
    ASTC_4x4x4 = 44,
    ASTC_5x4x4 = 45,
    ASTC_5x5x4 = 46,
    ASTC_5x5x5 = 47,
    ASTC_6x5x5 = 48,
    ASTC_6x6x5 = 49,
    ASTC_6x6x6 = 50
};

enum class ColorSpace : uint32_t { LinearRGB = 0, sRGB = 1 };

enum class ChannelType : uint32_t {
    Unsigned_Byte_Normalised = 0,
    Signed_Byte_Normalized = 1,
    Unsigned_Byte = 2,
    Signed_Byte = 3,
    Unsigned_Short_Normalized = 4,
    Signed_Short_Normalized = 5,
    Unsigned_Short = 6,
    Signed_Short = 7,
    Unsigned_Int_Normalized = 8,
    Signed_Int_Normalized = 9,
    Unsigned_Int = 10,
    Signed_Int = 11,
    Float = 12
};

struct Header {
    uint32_t version = 0x03525650;
    Flags flags;
    PixelFormat pixel_format;
    ColorSpace color_space;
    ChannelType channel_type;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t num_surfaces;
    uint32_t num_faces;
    uint32_t mipmap_count;
    uint32_t metadata_size;
};

struct MetaData {
    unsigned char fourCC[4];
    uint32_t key;
    uint32_t data_size;
    uint8_t data;
};

struct File {
    Header header;
    std::vector<MetaData> metaData;
    uint8_t* pTextureData;
    size_t szTextureDataSize;
};

inline std::ostream& operator<<(std::ostream& s, const PVR::Header h) {
    s.write(reinterpret_cast<const char*>(&h), sizeof(PVR::Header));
    return s;
}

inline std::ostream& operator<<(std::ostream& s,
                                const std::vector<MetaData> metas) {
    for (auto& meta : metas) {
        s << meta.fourCC;
        s << meta.key;
        s.write(reinterpret_cast<const char*>(&meta.data), meta.data_size);
    }
    return s;
}

inline std::ostream& operator<<(std::ostream& s, const PVR::File f) {
    s << f.header;
    s << f.metaData;
    s.write(reinterpret_cast<const char*>(f.pTextureData), f.szTextureDataSize);
    return s;
}

class PvrParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        if (buf.GetDataSize() < sizeof(PVR::Header)) {
            std::cerr << "Truncated PVR header" << std::endl;
            throw std::runtime_error("Truncated PVR header");
        }
        PVR::Header header;
        memcpy(&header, buf.GetData(), sizeof(header));
        const auto* pHeader = &header;
        if (pHeader->version != 0x03525650) {
            std::cerr << "Invalid PVR version" << std::endl;
            throw std::runtime_error("Invalid PVR version");
        }
        {
            std::cerr << "Asset is PVR file" << std::endl;
            std::cerr << "PVR Header" << std::endl;
            std::cerr << "----------------------------" << std::endl;
            fprintf(stderr, "Image dimension: (%d x %d x %d)\n", pHeader->width,
                    pHeader->height, pHeader->depth);
            //std::cerr << "Image pixel format: " << pHeader->pixel_format << std::endl;

            img.Width = pHeader->width;
            img.Height = pHeader->height;
            if (!img.Width || !img.Height || pHeader->depth != 1 ||
                pHeader->num_faces != 1 || pHeader->num_surfaces != 1 ||
                !pHeader->mipmap_count ||
                pHeader->metadata_size > buf.GetDataSize() - sizeof(PVR::Header)) {
                std::cerr << "Invalid or unsupported PVR texture layout" << std::endl;
                throw std::runtime_error("Invalid or unsupported PVR texture layout");
            }
            auto data_offset = sizeof(PVR::Header) + pHeader->metadata_size;
            const auto available_size = buf.GetDataSize() - data_offset;
            img.compressed = true;
            size_t block_size = 16;
            switch (pHeader->pixel_format) {
                case PVR::PixelFormat::BC1:
                    img.compress_format = COMPRESSED_FORMAT::BC1;
                    block_size = 8;
                    break;
                case PVR::PixelFormat::BC2:
                    img.compress_format = COMPRESSED_FORMAT::BC2;
                    break;
                case PVR::PixelFormat::BC3:
                    img.compress_format = COMPRESSED_FORMAT::BC3;
                    break;
                case PVR::PixelFormat::BC4:
                    img.compress_format = COMPRESSED_FORMAT::BC4;
                    block_size = 8;
                    break;
                case PVR::PixelFormat::BC5:
                    img.compress_format = COMPRESSED_FORMAT::BC5;
                    break;
                case PVR::PixelFormat::BC6H:
                    img.compress_format = COMPRESSED_FORMAT::BC6H;
                    break;
                case PVR::PixelFormat::BC7:
                    img.compress_format = COMPRESSED_FORMAT::BC7;
                    break;
                default:
                    std::cerr << "Unsupported PVR compression format" << std::endl;
                    throw std::runtime_error("Unsupported PVR compression format");
            }
            uint32_t max_mips = 1;
            for (auto dimension = std::max(img.Width, img.Height); dimension > 1; dimension >>= 1) {
                ++max_mips;
            }
            if (pHeader->mipmap_count > max_mips) {
                std::cerr << "Invalid PVR mip count" << std::endl;
                throw std::runtime_error("Invalid PVR mip count");
            }
            auto width = img.Width;
            auto height = img.Height;
            for (uint32_t level = 0; level < pHeader->mipmap_count; ++level) {
                const auto pitch = ((size_t{width} + 3) / 4) * block_size;
                const auto block_rows = (size_t{height} + 3) / 4;
                if (pitch > (available_size - img.data_size) / block_rows) {
                    std::cerr << "Truncated PVR mip data at level " << level << std::endl;
                    throw std::runtime_error("Truncated PVR mip data");
                }
                const auto size = pitch * block_rows;
                img.mipmaps.emplace_back(width, height, pitch, img.data_size, size);
                img.data_size += size;
                width = std::max(1u, width >> 1);
                height = std::max(1u, height >> 1);
            }
            img.pitch = img.mipmaps.front().pitch;
            img.data = new uint8_t[img.data_size];
            memcpy(img.data, buf.GetData() + data_offset, img.data_size);
        }

        return img;
    }
};

}  // namespace PVR
}  // namespace My

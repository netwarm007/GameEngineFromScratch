
#pragma once
#include <iostream>
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
        PVR::Header* pHeader = reinterpret_cast<PVR::Header*>(buf.GetData());
        if (pHeader->version == 0x03525650) {
            std::cerr << "Asset is PVR file" << std::endl;
            std::cerr << "PVR Header" << std::endl;
            std::cerr << "----------------------------" << std::endl;
            fprintf(stderr, "Image dimension: (%d x %d x %d)\n", pHeader->width,
                    pHeader->height, pHeader->depth);
            fprintf(stderr, "Image pixel format: %llu\n",
                    pHeader->pixel_format);

            img.Width = pHeader->width;
            img.Height = pHeader->height;
            auto data_offset = sizeof(PVR::Header) + pHeader->metadata_size;
            img.data_size = (size_t)buf.GetDataSize() - data_offset;
            img.data = new uint8_t[img.data_size];
            img.compressed = true;
            switch (pHeader->pixel_format) {
                case PVR::PixelFormat::BC1:
                    img.compress_format = COMPRESSED_FORMAT::BC1;
                    break;
                case PVR::PixelFormat::BC2:
                    img.compress_format = COMPRESSED_FORMAT::BC2;
                    break;
                case PVR::PixelFormat::BC3:
                    img.compress_format = COMPRESSED_FORMAT::BC3;
                    break;
                case PVR::PixelFormat::BC4:
                    img.compress_format = COMPRESSED_FORMAT::BC4U;
                    break;
                case PVR::PixelFormat::BC5:
                    img.compress_format = COMPRESSED_FORMAT::BC5U;
                    break;
                case PVR::PixelFormat::BC6H:
                    img.compress_format = COMPRESSED_FORMAT::BC6U;
                    break;
                case PVR::PixelFormat::BC7:
                    img.compress_format = COMPRESSED_FORMAT::BC7U;
                    break;
                default:
                    assert(0);
            }

            memcpy(img.data, buf.GetData() + data_offset, img.data_size);
        }

        return img;
    }
};

}  // namespace PVR
}  // namespace My

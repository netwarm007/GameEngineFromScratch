#pragma once
#include <algorithm>

#ifdef _WIN32
#include <dxgiformat.h>
#else
#endif

#include "ImageParser.hpp"
#include "portable.hpp"

namespace My {
typedef enum MY_DXGI_FORMAT {
    DXGI_FORMAT_UNKNOWN,
    DXGI_FORMAT_R32G32B32A32_TYPELESS,
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    DXGI_FORMAT_R32G32B32A32_UINT,
    DXGI_FORMAT_R32G32B32A32_SINT,
    DXGI_FORMAT_R32G32B32_TYPELESS,
    DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32B32_UINT,
    DXGI_FORMAT_R32G32B32_SINT,
    DXGI_FORMAT_R16G16B16A16_TYPELESS,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_R16G16B16A16_UNORM,
    DXGI_FORMAT_R16G16B16A16_UINT,
    DXGI_FORMAT_R16G16B16A16_SNORM,
    DXGI_FORMAT_R16G16B16A16_SINT,
    DXGI_FORMAT_R32G32_TYPELESS,
    DXGI_FORMAT_R32G32_FLOAT,
    DXGI_FORMAT_R32G32_UINT,
    DXGI_FORMAT_R32G32_SINT,
    DXGI_FORMAT_R32G8X24_TYPELESS,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
    DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
    DXGI_FORMAT_R10G10B10A2_TYPELESS,
    DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R10G10B10A2_UINT,
    DXGI_FORMAT_R11G11B10_FLOAT,
    DXGI_FORMAT_R8G8B8A8_TYPELESS,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UINT,
    DXGI_FORMAT_R8G8B8A8_SNORM,
    DXGI_FORMAT_R8G8B8A8_SINT,
    DXGI_FORMAT_R16G16_TYPELESS,
    DXGI_FORMAT_R16G16_FLOAT,
    DXGI_FORMAT_R16G16_UNORM,
    DXGI_FORMAT_R16G16_UINT,
    DXGI_FORMAT_R16G16_SNORM,
    DXGI_FORMAT_R16G16_SINT,
    DXGI_FORMAT_R32_TYPELESS,
    DXGI_FORMAT_D32_FLOAT,
    DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32_UINT,
    DXGI_FORMAT_R32_SINT,
    DXGI_FORMAT_R24G8_TYPELESS,
    DXGI_FORMAT_D24_UNORM_S8_UINT,
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
    DXGI_FORMAT_X24_TYPELESS_G8_UINT,
    DXGI_FORMAT_R8G8_TYPELESS,
    DXGI_FORMAT_R8G8_UNORM,
    DXGI_FORMAT_R8G8_UINT,
    DXGI_FORMAT_R8G8_SNORM,
    DXGI_FORMAT_R8G8_SINT,
    DXGI_FORMAT_R16_TYPELESS,
    DXGI_FORMAT_R16_FLOAT,
    DXGI_FORMAT_D16_UNORM,
    DXGI_FORMAT_R16_UNORM,
    DXGI_FORMAT_R16_UINT,
    DXGI_FORMAT_R16_SNORM,
    DXGI_FORMAT_R16_SINT,
    DXGI_FORMAT_R8_TYPELESS,
    DXGI_FORMAT_R8_UNORM,
    DXGI_FORMAT_R8_UINT,
    DXGI_FORMAT_R8_SNORM,
    DXGI_FORMAT_R8_SINT,
    DXGI_FORMAT_A8_UNORM,
    DXGI_FORMAT_R1_UNORM,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
    DXGI_FORMAT_R8G8_B8G8_UNORM,
    DXGI_FORMAT_G8R8_G8B8_UNORM,
    DXGI_FORMAT_BC1_TYPELESS,
    DXGI_FORMAT_BC1_UNORM,
    DXGI_FORMAT_BC1_UNORM_SRGB,
    DXGI_FORMAT_BC2_TYPELESS,
    DXGI_FORMAT_BC2_UNORM,
    DXGI_FORMAT_BC2_UNORM_SRGB,
    DXGI_FORMAT_BC3_TYPELESS,
    DXGI_FORMAT_BC3_UNORM,
    DXGI_FORMAT_BC3_UNORM_SRGB,
    DXGI_FORMAT_BC4_TYPELESS,
    DXGI_FORMAT_BC4_UNORM,
    DXGI_FORMAT_BC4_SNORM,
    DXGI_FORMAT_BC5_TYPELESS,
    DXGI_FORMAT_BC5_UNORM,
    DXGI_FORMAT_BC5_SNORM,
    DXGI_FORMAT_B5G6R5_UNORM,
    DXGI_FORMAT_B5G5R5A1_UNORM,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_B8G8R8X8_UNORM,
    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM,
    DXGI_FORMAT_B8G8R8A8_TYPELESS,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_B8G8R8X8_TYPELESS,
    DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
    DXGI_FORMAT_BC6H_TYPELESS,
    DXGI_FORMAT_BC6H_UF16,
    DXGI_FORMAT_BC6H_SF16,
    DXGI_FORMAT_BC7_TYPELESS,
    DXGI_FORMAT_BC7_UNORM,
    DXGI_FORMAT_BC7_UNORM_SRGB,
    DXGI_FORMAT_AYUV,
    DXGI_FORMAT_Y410,
    DXGI_FORMAT_Y416,
    DXGI_FORMAT_NV12,
    DXGI_FORMAT_P010,
    DXGI_FORMAT_P016,
    DXGI_FORMAT_420_OPAQUE,
    DXGI_FORMAT_YUY2,
    DXGI_FORMAT_Y210,
    DXGI_FORMAT_Y216,
    DXGI_FORMAT_NV11,
    DXGI_FORMAT_AI44,
    DXGI_FORMAT_IA44,
    DXGI_FORMAT_P8,
    DXGI_FORMAT_A8P8,
    DXGI_FORMAT_B4G4R4A4_UNORM,
    DXGI_FORMAT_P208,
    DXGI_FORMAT_V208,
    DXGI_FORMAT_V408,
    DXGI_FORMAT_FORCE_UINT
} MY_DXGI_FORMAT;

typedef enum MY_D3D10_RESOURCE_DIMENSION {
    D3D10_RESOURCE_DIMENSION_UNKNOWN,
    D3D10_RESOURCE_DIMENSION_BUFFER,
    D3D10_RESOURCE_DIMENSION_TEXTURE1D,
    D3D10_RESOURCE_DIMENSION_TEXTURE2D,
    D3D10_RESOURCE_DIMENSION_TEXTURE3D
} MY_D3D10_RESOURCE_DIMENSION;

typedef enum MY_D3DFMT {
    D3DFMT_A16B16G16R16F = 113,
    D3DFMT_A32B32G32R32F = 116
} MY_D3DFMT;

typedef struct {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
} DDS_PIXELFORMAT;

typedef struct {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth;
    uint32_t dwMipMapCount;
    uint32_t dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t dwCaps;
    uint32_t dwCaps2;
    uint32_t dwCaps3;
    uint32_t dwCaps4;
    uint32_t dwReserved2;
} DDS_HEADER;

typedef struct {
    MY_DXGI_FORMAT dxgiFormat;
    MY_D3D10_RESOURCE_DIMENSION resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlag2;
} DDS_HEADER_DXT10;

class DdsParser : _implements_ ImageParser {
   public:
    Image Parse(Buffer& buf) override {
        Image img;
        uint8_t* pData = buf.GetData();

        [[maybe_unused]] const auto* pdwMagic = reinterpret_cast<const uint32_t*>(pData);
        pData += sizeof(uint32_t);
        assert(*pdwMagic == endian_net_unsigned_int("DDS "_u32));
        std::cerr << "The image is DDS format" << std::endl;

        const auto* pHeader = reinterpret_cast<const DDS_HEADER*>(pData);
        pData += sizeof(DDS_HEADER);

        assert(pHeader->dwSize == 124);
        img.Width = pHeader->dwWidth;
        img.Height = pHeader->dwHeight;
        img.pitch = pHeader->dwPitchOrLinearSize;  // unreliable
        auto mipmap_count = pHeader->dwMipMapCount;
        assert(pHeader->ddspf.dwSize == 32);

        if (pHeader->ddspf.dwFlags & 0x4 /* DDPF_FOURCC */) {
            const uint32_t* pdwFourCC = &pHeader->ddspf.dwFourCC;
            const char* pCC = reinterpret_cast<const char*>(pdwFourCC);
            if (pCC[0] != 'D') {
                auto format = (MY_D3DFMT)*pdwFourCC;
                img.bitcount = pHeader->ddspf.dwRGBBitCount;

                switch (format) {
                    case MY_D3DFMT::D3DFMT_A16B16G16R16F:
                        std::cerr << "D3DFMT_A16B16G16R16F" << std::endl;
                        img.compressed = false;
                        img.is_float = true;
                        assert(img.bitcount == 64);
                        break;
                    case MY_D3DFMT::D3DFMT_A32B32G32R32F:
                        std::cerr << "D3DFMT_A32B32G32R32F" << std::endl;
                        img.compressed = false;
                        img.is_float = true;
                        assert(img.bitcount == 128);
                        break;
                    default:
                        std::cerr << "format is not supported!" << std::endl;
                        assert(0);
                }
            } else {
                std::cerr << "Compressed: ";
                std::cerr << pCC[0] << pCC[1] << pCC[2] << pCC[3] << std::endl;

                img.compressed = true;
            }
        }

        img.mipmaps.reserve(mipmap_count);

        if (img.compressed) {
            const uint32_t* pdwFourCC = &pHeader->ddspf.dwFourCC;
            if (*pdwFourCC == endian_net_unsigned_int("DXT1"_u32)) {
                img.compress_format = "DXT1"_u32;
                img.pitch = std::max(1u, ALIGN(img.Width, 4)) * 2;
                img.bitcount = 4;
            } else if (*pdwFourCC == endian_net_unsigned_int("DXT2"_u32)) {
                img.compress_format = "DXT2"_u32;
                img.pitch = std::max(1u, ALIGN(img.Width, 4)) * 4;
                img.bitcount = 8;
            } else if (*pdwFourCC == endian_net_unsigned_int("DXT3"_u32)) {
                img.compress_format = "DXT3"_u32;
                img.pitch = std::max(1u, ALIGN(img.Width, 4)) * 4;
                img.bitcount = 8;
            } else if (*pdwFourCC == endian_net_unsigned_int("DXT4"_u32)) {
                img.compress_format = "DXT4"_u32;
                img.pitch = std::max(1u, ALIGN(img.Width, 4)) * 4;
                img.bitcount = 8;
            } else if (*pdwFourCC == endian_net_unsigned_int("DXT5"_u32)) {
                img.compress_format = "DXT5"_u32;
                img.pitch = std::max(1u, ALIGN(img.Width, 4)) * 4;
                img.bitcount = 8;
            } else if (*pdwFourCC == endian_net_unsigned_int("DX10"_u32)) {
                img.compress_format = "DX10"_u32;
                const auto* pHeaderDXT10 =
                    reinterpret_cast<const DDS_HEADER_DXT10*>(pData);
                pData += sizeof(DDS_HEADER_DXT10);
                std::cerr << "DXGI_FORMAT: " << pHeaderDXT10->dxgiFormat
                          << std::endl;
            }

            if (mipmap_count > 0) {
                uint32_t width = img.Width;
                uint32_t height = img.Height;

                for (decltype(mipmap_count) i = 0; i < mipmap_count; i++) {
                    if (width > 0 && height > 0) {
                        auto pitch = std::max(1u, (ALIGN(width, 4)) >> 2) *
                                     img.bitcount *
                                     2;  //  img.bitcount / 8 * 16
                        img.mipmaps.emplace_back(
                            width, height, pitch,
                            pitch * (ALIGN(height, 4) >> 2), img.data_size);
                        img.data_size += img.mipmaps[i].data_size;

                        width >>= 1;   // /2
                        height >>= 1;  // /2
                    } else {
                        break;
                    }
                }
            }
        } else {
            img.pitch = ALIGN(img.Width * img.bitcount, 8) / 8;

            if (mipmap_count > 0) {
                uint32_t width = img.Width;
                uint32_t height = img.Height;

                for (decltype(mipmap_count) i = 0; i < mipmap_count; i++) {
                    if (width > 0 && height > 0) {
                        auto pitch = ALIGN(width * img.bitcount, 8) / 8;
                        img.mipmaps.emplace_back(
                            width, height, pitch, img.data_size /* as offset */,
                            pitch * height /* as data_size */);
                        img.data_size += img.mipmaps[i].data_size;

                        width >>= 1;
                        height >>= 1;
                    } else {
                        break;
                    }
                }
            }
        }

        assert(img.data_size < buf.GetDataSize());

        img.data = new uint8_t[img.data_size];

        memcpy(img.data, pData, img.data_size);

        return img;
    }
};
}  // namespace My

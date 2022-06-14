#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include "ASTC.hpp"
#include "AssetLoader.hpp"
#include "BMP.hpp"
#include "DDS.hpp"
#include "HDR.hpp"
#include "JPEG.hpp"
#include "PNG.hpp"
#include "PVR.hpp"
#include "TGA.hpp"
#include "ispc_texcomp.h"

using namespace My;

void alloc_image(rgba_surface* img, int width, int height) {
    img->width = width;
    img->height = height;
    img->stride = img->width * 4;
    img->ptr = (uint8_t*)new uint8_t[img->height * img->stride];
}

int idiv_ceil(int n, int d) { return (n + d - 1) / d; }

void flip_image(rgba_surface* rec_img)
{
    rec_img->ptr += (rec_img->height - 1) * rec_img->stride;
    rec_img->stride *= -1;
}

void fill_borders(rgba_surface* dst, rgba_surface* src, int block_width,
                  int block_height) {
    int full_width = idiv_ceil(src->width, block_width) * block_width;
    int full_height = idiv_ceil(src->height, block_height) * block_height;
    alloc_image(dst, full_width, full_height);

    ReplicateBorders(dst, src, 0, 0, 32);
}

int main(int argc, char** argv) {
    int error = 0;

    if (argc < 3) {
        fprintf(stderr,
                "Usage: TextureCompressor <input_file> <output_file> [astc]\n");
        error = 1;
    } else {
        AssetLoader assetLoader;
        error = assetLoader.Initialize();

        if (!error) {
            Image image;
            std::string inFileName = argv[1];
            {
                Buffer buf =
                    assetLoader.SyncOpenAndReadBinary(inFileName.c_str());
                auto ext = inFileName.substr(inFileName.find_last_of('.'));
                if (ext == ".jpg" || ext == ".jpeg") {
                    JfifParser jfif_parser;
                    image = jfif_parser.Parse(buf);
                } else if (ext == ".png") {
                    PngParser png_parser;
                    image = png_parser.Parse(buf);
                } else if (ext == ".bmp") {
                    BmpParser bmp_parser;
                    image = bmp_parser.Parse(buf);
                } else if (ext == ".tga") {
                    TgaParser tga_parser;
                    image = tga_parser.Parse(buf);
                } else if (ext == ".dds") {
                    DdsParser dds_parser;
                    image = dds_parser.Parse(buf);
                } else if (ext == ".hdr") {
                    HdrParser hdr_parser;
                    image = hdr_parser.Parse(buf);
                } else if (ext == ".astc") {
                    AstcParser astc_parser;
                    image = astc_parser.Parse(buf);
                } else if (ext == ".pvr") {
                    PVR::PvrParser pvr_parser;
                    image = pvr_parser.Parse(buf);
                } else {
                    assert(0);
                }
            }

            assetLoader.Finalize();

            if (image.compressed) exit(1);

            COMPRESSED_FORMAT compressed_format = COMPRESSED_FORMAT::NONE;
            PVR::PixelFormat pvr_pixel_format;
            size_t compressed_size = 0;
            int astc_block_size_x = 6;
            int astc_block_size_y = 6;
            int astc_image_width_in_blocks =
                idiv_ceil(image.Width, astc_block_size_x);
            int astc_image_height_in_blocks =
                idiv_ceil(image.Height, astc_block_size_x);

            if (argc > 3 && strncmp(argv[3], "astc", 4) == 0) {
                compressed_format = COMPRESSED_FORMAT::ASTC_6x6;
                compressed_size = astc_image_width_in_blocks *
                                  astc_image_height_in_blocks * 16;
            } else {
                switch (image.pixel_format) {
                    case PIXEL_FORMAT::R8:
                        compressed_format = COMPRESSED_FORMAT::BC4;
                        pvr_pixel_format = PVR::PixelFormat::BC4;
                        compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                          (ALIGN(image.Width, 4) >> 2) * 8;
                        break;
                    case PIXEL_FORMAT::RG8:
                        compressed_format = COMPRESSED_FORMAT::BC5;
                        pvr_pixel_format = PVR::PixelFormat::BC5;
                        compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                          (ALIGN(image.Width, 4) >> 2) * 16;
                        break;
                    case PIXEL_FORMAT::RGB8:
                        compressed_format = COMPRESSED_FORMAT::BC1;
                        pvr_pixel_format = PVR::PixelFormat::BC1;
                        compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                          (ALIGN(image.Width, 4) >> 2) * 8;
                        break;
                    case PIXEL_FORMAT::RGBA8:
                        compressed_format = COMPRESSED_FORMAT::BC3;
                        pvr_pixel_format = PVR::PixelFormat::BC3;
                        compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                          (ALIGN(image.Width, 4) >> 2) * 16;
                        break;
                    case PIXEL_FORMAT::RGB16:
                        compressed_format = COMPRESSED_FORMAT::BC6H;
                        pvr_pixel_format = PVR::PixelFormat::BC6H;
                        compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                          (ALIGN(image.Width, 4) >> 2) * 16;
                        break;
                    default:
                        compressed_format = COMPRESSED_FORMAT::ASTC_6x6;
                        compressed_size = astc_image_width_in_blocks *
                                          astc_image_height_in_blocks * 16;
                }
            }

            rgba_surface src_surface;
            src_surface.height = static_cast<int32_t>(image.Height);
            src_surface.width = static_cast<int32_t>(image.Width);
            src_surface.ptr = image.data;
            src_surface.stride = static_cast<int32_t>(image.pitch);

            flip_image(&src_surface);

            rgba_surface edged_img;
            fill_borders(&edged_img, &src_surface, 6, 6);

            std::vector<uint8_t> _dst_buf(compressed_size);

            switch (compressed_format) {
                case COMPRESSED_FORMAT::BC1:
                    fprintf(stderr, "Compress with BC1\n");
                    CompressBlocksBC1(&src_surface, _dst_buf.data());
                    break;
                case COMPRESSED_FORMAT::BC3:
                    fprintf(stderr, "Compress with BC3\n");
                    CompressBlocksBC3(&src_surface, _dst_buf.data());
                    break;
                case COMPRESSED_FORMAT::BC4:
                    fprintf(stderr, "Compress with BC4\n");
                    CompressBlocksBC4(&src_surface, _dst_buf.data());
                    break;
                case COMPRESSED_FORMAT::BC5:
                    fprintf(stderr, "Compress with BC5\n");
                    CompressBlocksBC5(&src_surface, _dst_buf.data());
                    break;
                case COMPRESSED_FORMAT::BC6H:
                    fprintf(stderr, "Compress with BC6H\n");
                    {
                        bc6h_enc_settings settings;
                        GetProfile_bc6h_basic(&settings);
                        CompressBlocksBC6H(&src_surface, _dst_buf.data(),
                                           &settings);
                    }
                    break;
                case COMPRESSED_FORMAT::BC7:
                    fprintf(stderr, "Compress with BC7\n");
                    {
                        bc7_enc_settings settings;
                        GetProfile_alpha_basic(&settings);
                        CompressBlocksBC7(&src_surface, _dst_buf.data(),
                                          &settings);
                    }
                    break;
                case COMPRESSED_FORMAT::ASTC_6x6:
                    fprintf(stderr, "Compress with ASTC 6x6\n");
                    {
                        astc_enc_settings settings;
                        if (image.bitcount / image.bitdepth == 4)
                            GetProfile_astc_alpha_slow(&settings, 6, 6);
                        else
                            GetProfile_astc_fast(&settings, 6, 6);
                        CompressBlocksASTC(&edged_img, _dst_buf.data(),
                                           &settings);
                    }
                    break;
                default:
                    assert(0);
            }

            delete[] edged_img.ptr;

            fprintf(
                stderr,
                "decompressed size: %zu bytes, compressed size: %zu bytes\n",
                image.data_size, compressed_size);

            switch (compressed_format) {
                case COMPRESSED_FORMAT::ASTC_6x6: {
                    astc_image compressedFile;
                    uint32_t magic = MAGIC_FILE_CONSTANT;
                    memcpy(compressedFile.header.magic, &magic, 4);
                    int xsize = image.Width;
                    int ysize = image.Height;
                    int zsize = 1;
                    memcpy(compressedFile.header.dim_x, &xsize, 3);
                    memcpy(compressedFile.header.dim_y, &ysize, 3);
                    memcpy(compressedFile.header.dim_z, &zsize, 3);
                    compressedFile.header.block_x = astc_block_size_x;
                    compressedFile.header.block_y = astc_block_size_y;
                    compressedFile.header.block_z = 1;

                    std::string outputFileName(argv[2]);
                    outputFileName += ".astc";

                    FILE* f = fopen(outputFileName.c_str(), "wb");
                    fwrite(&compressedFile.header, sizeof(astc_header), 1, f);

                    fwrite(_dst_buf.data(), compressed_size, 1, f);

                    fclose(f);
                } break;
                default: {
                    PVR::File compressedFile;

                    compressedFile.header.flags = PVR::Flags::NoFlag;
                    compressedFile.header.pixel_format = pvr_pixel_format;
                    compressedFile.header.color_space =
                        PVR::ColorSpace::LinearRGB;
                    if (image.bitdepth > 8) {
                        compressedFile.header.channel_type =
                            PVR::ChannelType::Float;
                    } else {
                        compressedFile.header.channel_type =
                            PVR::ChannelType::Unsigned_Byte_Normalised;
                    }
                    compressedFile.header.height = ALIGN(image.Height, 4);
                    compressedFile.header.width = ALIGN(image.Width, 4);
                    compressedFile.header.depth = 1;
                    compressedFile.header.num_faces = 1;
                    compressedFile.header.num_surfaces = 1;
                    compressedFile.header.mipmap_count = 1;
                    compressedFile.header.metadata_size = 0;

                    compressedFile.pTextureData = _dst_buf.data();
                    compressedFile.szTextureDataSize = compressed_size;

                    std::string outputFileName = argv[2];
                    std::ofstream outputFile(outputFileName + ".pvr",
                                             std::ios::binary);

                    outputFile << compressedFile;

                    outputFile.close();
                }
            }
        }
    }

    return error;
}

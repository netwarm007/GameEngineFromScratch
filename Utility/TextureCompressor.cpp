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

int main(int argc, char** argv) {
    int error = 0;

    if (argc != 3) {
        fprintf(stderr,
                "Usage: TextureCompressor <input_file> <output_file>\n");
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

            PVR::File compressedFile;

            compressedFile.header.flags = PVR::Flags::NoFlag;
            compressedFile.header.channel_type =
                PVR::ChannelType::Unsigned_Byte;
            compressedFile.header.height = ALIGN(image.Height, 4);
            compressedFile.header.width = ALIGN(image.Width, 4);
            compressedFile.header.depth = 1;
            compressedFile.header.num_faces = 1;
            compressedFile.header.num_surfaces = 1;
            compressedFile.header.mipmap_count = 1;
            compressedFile.header.metadata_size = 0;

            size_t compressed_size = 0;

            switch (image.pixel_format) {
                case PIXEL_FORMAT::R8:
                    compressedFile.header.pixel_format = PVR::PixelFormat::BC4;
                    compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                      (ALIGN(image.Width, 4) >> 2) * 8;
                    break;
                case PIXEL_FORMAT::RG8:
                    compressedFile.header.pixel_format = PVR::PixelFormat::BC5;
                    compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                      (ALIGN(image.Width, 4) >> 2) * 16;
                    break;
                case PIXEL_FORMAT::RGB8:
                    compressedFile.header.pixel_format = PVR::PixelFormat::BC1;
                    compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                      (ALIGN(image.Width, 4) >> 2) * 8;
                    break;
                case PIXEL_FORMAT::RGBA8:
                    compressedFile.header.pixel_format = PVR::PixelFormat::BC3;
                    compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                      (ALIGN(image.Width, 4) >> 2) * 16;
                    break;
                case PIXEL_FORMAT::RGB16:
                    compressedFile.header.pixel_format = PVR::PixelFormat::BC6H;
                    compressed_size = (ALIGN(image.Height, 4) >> 2) *
                                      (ALIGN(image.Width, 4) >> 2) * 16;
                    break;
                default:
                    fprintf(stderr, "format not supported. %hu\n", image.pixel_format);
                    assert(0);
            }

            rgba_surface src_surface;
            src_surface.height = static_cast<int32_t>(image.Height);
            src_surface.width = static_cast<int32_t>(image.Width);
            src_surface.ptr = image.data;
            src_surface.stride = static_cast<int32_t>(image.pitch);

            std::vector<uint8_t> _dst_buf(compressed_size);

            switch (compressedFile.header.pixel_format) {
                case PVR::PixelFormat::BC1:
                    fprintf(stderr, "Compress with BC1\n");
                    CompressBlocksBC1(&src_surface, _dst_buf.data());
                    break;
                case PVR::PixelFormat::BC3:
                    fprintf(stderr, "Compress with BC3\n");
                    CompressBlocksBC3(&src_surface, _dst_buf.data());
                    break;
                case PVR::PixelFormat::BC4:
                    fprintf(stderr, "Compress with BC4\n");
                    CompressBlocksBC4(&src_surface, _dst_buf.data());
                    break;
                case PVR::PixelFormat::BC5:
                    fprintf(stderr, "Compress with BC5\n");
                    CompressBlocksBC5(&src_surface, _dst_buf.data());
                    break;
                case PVR::PixelFormat::BC6H:
                    fprintf(stderr, "Compress with BC6H\n");
                    {
                        bc6h_enc_settings settings;
                        GetProfile_bc6h_basic(&settings);
                        CompressBlocksBC6H(&src_surface, _dst_buf.data(),
                                           &settings);
                    }
                    break;
                case PVR::PixelFormat::BC7:
                    fprintf(stderr, "Compress with BC7\n");
                    {
                        bc7_enc_settings settings;
                        GetProfile_basic(&settings);
                        CompressBlocksBC7(&src_surface, _dst_buf.data(),
                                          &settings);
                    }
                    break;
                default:
                    assert(0);
            }

            fprintf(
                stderr,
                "decompressed size: %zu bytes, compressed size: %zu bytes\n",
                image.data_size, compressed_size);

            compressedFile.pTextureData = _dst_buf.data();
            compressedFile.szTextureDataSize = compressed_size;

            std::ofstream outputFile(argv[2], std::ios::binary);

            outputFile << compressedFile;

            outputFile.close();
        }
    }

    return error;
}

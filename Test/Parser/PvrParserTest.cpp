#include <cstdio>
#include "PVR.hpp"
#include "AssetLoader.hpp"

using namespace My;

static Buffer MakeMipTexture(PVR::PixelFormat format, uint32_t width,
                             uint32_t height, uint32_t levels, size_t payload_size) {
    PVR::Header header{};
    header.pixel_format = format;
    header.width = width;
    header.height = height;
    header.depth = 1;
    header.num_faces = 1;
    header.num_surfaces = 1;
    header.mipmap_count = levels;
    Buffer buffer(sizeof(header) + payload_size);
    memcpy(buffer.GetData(), &header, sizeof(header));
    for (size_t i = 0; i < payload_size; ++i) {
        buffer.GetData()[sizeof(header) + i] = static_cast<uint8_t>(i);
    }
    return buffer;
}

static bool TestMipmaps() {
    PVR::PvrParser parser;
    auto bc1 = MakeMipTexture(PVR::PixelFormat::BC1, 8, 4, 4, 40);
    auto image = parser.Parse(bc1);
    if (image.mipmaps.size() != 4 || image.pitch != 16 || image.data_size != 40 ||
        image.mipmaps[1].Width != 4 || image.mipmaps[1].Height != 2 ||
        image.mipmaps[1].offset != 16 || image.mipmaps[1].data_size != 8 ||
        image.mipmaps[3].Width != 1 || image.mipmaps[3].Height != 1 ||
        image.mipmaps[3].pitch != 8 || image.mipmaps[3].offset != 32 ||
        image.data[image.mipmaps[3].offset] != 32) {
        std::cerr << "BC1 mip layout or payload is incorrect" << std::endl;
        return false;
    }
    auto bc5 = MakeMipTexture(PVR::PixelFormat::BC5, 5, 7, 3, 96);
    image = parser.Parse(bc5);
    if (image.mipmaps.size() != 3 || image.pitch != 32 || image.data_size != 96 ||
        image.mipmaps[0].data_size != 64 ||
        image.mipmaps[1].Width != 2 || image.mipmaps[1].Height != 3 ||
        image.mipmaps[1].pitch != 16 || image.mipmaps[1].offset != 64 ||
        image.mipmaps[2].offset != 80 || image.mipmaps[2].data_size != 16) {
        std::cerr << "Non-power-of-two BC5 mip layout is incorrect" << std::endl;
        return false;
    }
    auto truncated = MakeMipTexture(PVR::PixelFormat::BC5, 5, 7, 3, 95);
    try {
        parser.Parse(truncated);
        std::cerr << "Truncated PVR mip chain was accepted" << std::endl;
        return false;
    } catch (const std::runtime_error&) {
    }
    auto invalid_count = MakeMipTexture(PVR::PixelFormat::BC1, 8, 4, 5, 48);
    try {
        parser.Parse(invalid_count);
        std::cerr << "Invalid PVR mip count was accepted" << std::endl;
        return false;
    } catch (const std::runtime_error&) {
    }
    return true;
}

int main(int argc, char** argv) {
    if (!TestMipmaps()) return 1;
    int error = 0;

    AssetLoader assetLoader;

    error = assetLoader.Initialize();

    if (!error) {
        Buffer buf;
        if (argc >= 2) {
            buf = assetLoader.SyncOpenAndReadBinary(argv[1]);
        } else {
            buf = assetLoader.SyncOpenAndReadBinary(
                "Textures/viking_room.pvr");
        }

        PVR::PvrParser astc_parser;

        Image image = astc_parser.Parse(buf);

        std::cout << image;
        if (argc < 2 && (image.mipmaps.size() != 1 ||
                        image.mipmaps.front().data_size != image.data_size)) {
            std::cerr << "Existing single-mip PVR asset changed layout" << std::endl;
            error = 1;
        }

        //image.SaveTGA("PvrParserTest.tga");
    }

    assetLoader.Finalize();

    return error;
}
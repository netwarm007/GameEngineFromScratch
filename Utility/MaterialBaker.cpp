#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "AssetLoader.hpp"
#include "BaseApplication.hpp"
#include "PVR.hpp"
#include "SceneManager.hpp"
#include "ispc_texcomp.h"

using namespace My;
using namespace std;

template <typename T>
static ostream& operator<<(ostream& out,
                           unordered_map<string, shared_ptr<T>> map) {
    for (auto p : map) {
        out << *p.second << endl;
    }

    return out;
}

void save_as_tga(const rgba_surface& surface, int32_t channels,
                 const std::string&& filename) {
    assert(filename != "");
    // must end in .tga
    FILE* file = fopen(filename.c_str(), "wb");
    // misc header information
    for (int i = 0; i < 18; i++) {
        if (i == 2)
            fprintf(file, "%c", 2);
        else if (i == 12)
            fprintf(file, "%c", surface.width % 256);
        else if (i == 13)
            fprintf(file, "%c", surface.width / 256);
        else if (i == 14)
            fprintf(file, "%c", surface.height % 256);
        else if (i == 15)
            fprintf(file, "%c", surface.height / 256);
        else if (i == 16)
            fprintf(file, "%c", 32);
        else if (i == 17)
            fprintf(
                file, "%c",
                0x28);  // Bit 5: screen origin bit, Bit 3 - 0 alpha bit depth
        else
            fprintf(file, "%c", 0);
    }
    // the data
    for (int32_t y = 0; y < surface.height; y++) {
        for (int32_t x = 0; x < surface.width; x++) {
            // note reversed order: b, g, r, a
            fprintf(file, "%c",
                    (channels > 2)
                        ? *(surface.ptr + y * surface.stride + x * channels + 2)
                        : '\0');
            fprintf(file, "%c",
                    (channels > 1)
                        ? *(surface.ptr + y * surface.stride + x * channels + 1)
                        : '\0');
            fprintf(file, "%c",
                    (channels > 0)
                        ? *(surface.ptr + y * surface.stride + x * channels + 0)
                        : '\0');
            fprintf(file, "%c",
                    (channels > 3)
                        ? *(surface.ptr + y * surface.stride + x * channels + 3)
                        : '\1');
        }
    }
    fclose(file);
}

void save_as_pvr(const rgba_surface& surface, const std::string&& filename,
                 PVR::PixelFormat compress_format) {
    size_t compressed_size = 0;

    switch (compress_format) {
        case PVR::PixelFormat::BC1:
        case PVR::PixelFormat::BC4:
            compressed_size = (ALIGN(surface.height, 4) >> 2) *
                              (ALIGN(surface.width, 4) >> 2) * 8;
            break;
        case PVR::PixelFormat::BC3:
        case PVR::PixelFormat::BC5:
        case PVR::PixelFormat::BC6H:
        case PVR::PixelFormat::BC7:
            compressed_size = (ALIGN(surface.height, 4) >> 2) *
                              (ALIGN(surface.width, 4) >> 2) * 16;
            break;
        default:
            assert(0);
    }

    std::vector<uint8_t> _dst_buf(compressed_size);

    switch (compress_format) {
        case PVR::PixelFormat::BC1:
            CompressBlocksBC1(&surface, _dst_buf.data());
            break;
        case PVR::PixelFormat::BC3:
            CompressBlocksBC3(&surface, _dst_buf.data());
            break;
        case PVR::PixelFormat::BC4:
            CompressBlocksBC4(&surface, _dst_buf.data());
            break;
        case PVR::PixelFormat::BC5:
            CompressBlocksBC5(&surface, _dst_buf.data());
            break;
        case PVR::PixelFormat::BC6H: {
            bc6h_enc_settings settings;
            GetProfile_bc6h_basic(&settings);
            CompressBlocksBC6H(&surface, _dst_buf.data(), &settings);
        } break;
        case PVR::PixelFormat::BC7: {
            bc7_enc_settings settings;
            GetProfile_alpha_basic(&settings);
            CompressBlocksBC7(&surface, _dst_buf.data(), &settings);
        } break;
        default:
            assert(0);
    }

    PVR::File compressedFile;

    compressedFile.header.flags = PVR::Flags::NoFlag;
    compressedFile.header.channel_type = PVR::ChannelType::Unsigned_Byte;
    compressedFile.header.height = ALIGN(surface.height, 4);
    compressedFile.header.width = ALIGN(surface.width, 4);
    compressedFile.header.depth = 1;
    compressedFile.header.num_faces = 1;
    compressedFile.header.num_surfaces = 1;
    compressedFile.header.mipmap_count = 1;
    compressedFile.header.metadata_size = 0;
    compressedFile.header.pixel_format = compress_format;
    compressedFile.pTextureData = _dst_buf.data();
    compressedFile.szTextureDataSize = compressed_size;

    std::cerr << "generate " << filename << std::endl;
    std::ofstream outputFile(filename, std::ios::binary);

    outputFile << compressedFile;

    outputFile.close();
    std::cerr << "finished " << filename << std::endl;
}

int main(int argc, char** argv) {
    int error = 0;

    BaseApplication app;
    AssetLoader assetLoader;
    SceneManager sceneManager;

    app.RegisterManagerModule(&assetLoader);
    app.RegisterManagerModule(&sceneManager);

    error = app.Initialize();

    if (argc >= 2) {
        sceneManager.LoadScene(argv[1]);
    } else {
        sceneManager.LoadScene("Materials/splash.mymaterial");
    }

    auto& scene = sceneManager.GetSceneForRendering();

    std::vector<std::future<void>> tasks;

    cerr << "Baking Materials" << endl;
    cerr << "---------------------------" << endl;
    for (const auto& _it : scene->Materials) {
        auto pMaterial = _it.second;
        if (pMaterial) {
            cerr << pMaterial->GetName() << std::endl;

            auto albedo = pMaterial->GetBaseColor().ValueMap;
            assert(albedo);
            auto albedo_texture = albedo->GetTextureImage();
            assert(albedo_texture);
            auto albedo_texture_width = albedo_texture->Width;
            auto albedo_texture_height = albedo_texture->Height;

            auto normal = pMaterial->GetNormal().ValueMap;
            assert(normal);
            auto normal_texture = normal->GetTextureImage();
            assert(normal_texture);
            auto normal_texture_width = normal_texture->Width;
            auto normal_texture_height = normal_texture->Height;

            auto metallic = pMaterial->GetMetallic().ValueMap;
            assert(metallic);
            auto metallic_texture = metallic->GetTextureImage();
            assert(metallic_texture);
            auto metallic_texture_width = metallic_texture->Width;
            auto metallic_texture_height = metallic_texture->Height;

            auto roughness = pMaterial->GetRoughness().ValueMap;
            assert(roughness);
            auto roughness_texture = roughness->GetTextureImage();
            assert(roughness_texture);
            auto roughness_texture_width = roughness_texture->Width;
            auto roughness_texture_height = roughness_texture->Height;

            auto ao = pMaterial->GetAO().ValueMap;
            assert(ao);
            auto ao_texture = ao->GetTextureImage();
            assert(ao_texture);
            auto ao_texture_width = ao_texture->Width;
            auto ao_texture_height = ao_texture->Height;

            auto max_width_1 = albedo_texture_width;

            auto max_width_2 =
                std::max({metallic_texture_width, roughness_texture_width,
                          ao_texture_width});

            auto max_height_1 = albedo_texture_height;

            auto max_height_2 =
                std::max({metallic_texture_height, roughness_texture_height,
                          ao_texture_height});

            /*
              Now we pack the texture into following format:
              +--------+--------+--------+--------+
              | R      | G      | B      | A      |
              +--------+--------+--------+--------+
              | Albe.R | Albe.G | Albe.B |        |     surf1
              +--------+--------+--------+--------+
              | Metal  | Rough  | AO     |        |     surf2
              +--------+--------+--------+--------+
              | Norm.X | Norm.Y |   -    |    -   |     surf3
              +--------+--------+--------+--------+
            */

            auto combine_textures_1 = [=]() {
                // surf
                rgba_surface surf;
                surf.width = max_width_1;
                surf.height = max_height_1;
                int32_t channels = 4;
                surf.stride = channels * surf.width;
                std::vector<uint8_t> buf1(surf.stride * surf.height);
                surf.ptr = buf1.data();
                float albedo_ratio_x = (float)albedo_texture_width / surf.width;
                float albedo_ratio_y =
                    (float)albedo_texture_height / surf.height;

                for (int32_t y = 0; y < surf.height; y++) {
                    for (int32_t x = 0; x < surf.width; x++) {
                        *(surf.ptr + y * surf.stride + x * channels) =
                            albedo_texture->GetR(
                                std::floor(x * albedo_ratio_x),
                                std::floor(y * albedo_ratio_y));
                        *(surf.ptr + y * surf.stride + x * channels + 1) =
                            albedo_texture->GetG(
                                std::floor(x * albedo_ratio_x),
                                std::floor(y * albedo_ratio_y));
                        *(surf.ptr + y * surf.stride + x * channels + 2) =
                            albedo_texture->GetB(
                                std::floor(x * albedo_ratio_x),
                                std::floor(y * albedo_ratio_y));
                        *(surf.ptr + y * surf.stride + x * channels + 3) = 0;
                    }
                }

                // Now, compress surf with BC1
                auto outputFileName = pMaterial->GetName();
                if (argc >= 3) {
                    outputFileName = argv[2];
                }
                outputFileName += "_1";

                save_as_pvr(surf, outputFileName + ".pvr",
                            PVR::PixelFormat::BC1);
                save_as_tga(surf, channels, outputFileName + ".tga");
            };

            tasks.push_back(std::async(launch::async, combine_textures_1));

            auto combine_textures_2 = [=]() {
                // surf
                rgba_surface surf;
                surf.width = max_width_2;
                surf.height = max_height_2;
                int32_t channels = 4;
                surf.stride = channels * surf.width;
                std::vector<uint8_t> buf2(surf.stride * surf.height);
                surf.ptr = buf2.data();
                float metallic_ratio_x =
                    (float)metallic_texture_width / surf.width;
                float metallic_ratio_y =
                    (float)metallic_texture_height / surf.height;
                float roughness_ratio_x =
                    (float)roughness_texture_width / surf.width;
                float roughness_ratio_y =
                    (float)roughness_texture_height / surf.height;
                float ao_ratio_x = (float)ao_texture_width / surf.width;
                float ao_ratio_y = (float)ao_texture_height / surf.height;

                for (int32_t y = 0; y < surf.height; y++) {
                    for (int32_t x = 0; x < surf.width; x++) {
                        *(surf.ptr + y * surf.stride + x * channels) =
                            metallic_texture->GetX(
                                std::floor(x * metallic_ratio_x),
                                std::floor(y * metallic_ratio_y));
                        *(surf.ptr + y * surf.stride + x * channels + 1) =
                            roughness_texture->GetX(
                                std::floor(x * roughness_ratio_x),
                                std::floor(y * roughness_ratio_y));
                        *(surf.ptr + y * surf.stride + x * channels + 2) =
                            ao_texture->GetX(std::floor(x * ao_ratio_x),
                                             std::floor(y * ao_ratio_y));
                        *(surf.ptr + y * surf.stride + x * channels + 3) = 0;
                    }
                }

                // Now, compress surf with BC1
                auto outputFileName = pMaterial->GetName();
                if (argc >= 3) {
                    outputFileName = argv[2];
                }
                outputFileName += "_2";

                save_as_pvr(surf, outputFileName + ".pvr",
                            PVR::PixelFormat::BC1);
                save_as_tga(surf, channels, outputFileName + ".tga");
            };

            tasks.push_back(std::async(launch::async, combine_textures_2));

            auto combine_textures_3 = [=]() {
                // surf
                rgba_surface surf;
                int32_t channels = 2;
                surf.width = normal_texture_width;
                surf.height = normal_texture_height;
                surf.stride = channels * surf.width;
                std::vector<uint8_t> buf2(surf.stride * surf.height);
                surf.ptr = buf2.data();

                for (int32_t y = 0; y < surf.height; y++) {
                    for (int32_t x = 0; x < surf.width; x++) {
                        *(surf.ptr + y * surf.stride + x * channels) =
                            normal_texture->GetX(x, y);
                        *(surf.ptr + y * surf.stride + x * channels + 1) =
                            normal_texture->GetY(x, y);
                    }
                }

                // Now, compress surf with BC5
                auto outputFileName = pMaterial->GetName();
                if (argc >= 3) {
                    outputFileName = argv[2];
                }
                outputFileName += "_3";

                save_as_pvr(surf, outputFileName + ".pvr",
                            PVR::PixelFormat::BC5);
                save_as_tga(surf, channels, outputFileName + ".tga");
            };

            tasks.push_back(std::async(launch::async, combine_textures_3));
        }
    }

    for (auto& task : tasks) {
        task.wait();
    }

    app.Finalize();

    return error;
}

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "AssetLoader.hpp"
#include "BaseApplication.hpp"
#include "SceneManager.hpp"
#include "ispc_texcomp.h"
#include "PVR.hpp"

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
        sceneManager.LoadScene("Scene/splash.ogex");
    }

    auto& scene = sceneManager.GetSceneForRendering();

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

            auto max_width_1 =
                std::max({albedo_texture_width, normal_texture_width});

            auto max_width_2 =
                std::max({normal_texture_width, metallic_texture_width,
                          roughness_texture_width, ao_texture_width});

            auto max_height_1 =
                std::max({albedo_texture_height, normal_texture_height});

            auto max_height_2 =
                std::max({normal_texture_height, metallic_texture_height,
                          roughness_texture_height, ao_texture_height});

            /*
              Now we pack the texture into following format:
              +--------+--------+--------+--------+
              | R      | G      | B      | A      |
              +--------+--------+--------+--------+
              | Albe.R | Albe.G | Albe.B | Norm.X |   surf1
              +--------+--------+--------+--------+
              | Meta   | Rough  | AO     | Norm.Y |   surf2
              +--------+--------+--------+--------+
            */

            rgba_surface surf1, surf2;
            surf1.width = max_width_1;
            surf1.height = max_height_1;
            surf1.stride = 4 * surf1.width;
            std::vector<uint8_t> buf1 (surf1.stride * surf1.height);
            surf1.ptr = buf1.data();
            
            for (int32_t y = 0; y < max_height_1; y++) {
                for (int32_t x = 0; x < max_width_1; x++) {
                    *(surf1.ptr + y * surf1.stride + x * 4) = albedo_texture->GetR(x, y);
                    *(surf1.ptr + y * surf1.stride + x * 4 + 1) = albedo_texture->GetG(x, y);
                    *(surf1.ptr + y * surf1.stride + x * 4 + 2) = albedo_texture->GetB(x, y);
                    *(surf1.ptr + y * surf1.stride + x * 4 + 3) = normal_texture->GetX(x, y);
                }
            }

            // Now, compress surf1 with BC7
            auto compressed_size = (ALIGN(max_height_1, 4) >> 2) *
                                      (ALIGN(max_width_1, 4) >> 2) * 16;

            std::vector<uint8_t> _dst_buf(compressed_size);

            bc7_enc_settings settings;
            GetProfile_basic(&settings);
            CompressBlocksBC7(&surf1, _dst_buf.data(), &settings);

            PVR::File compressedFile;

            compressedFile.header.flags = PVR::Flags::NoFlag;
            compressedFile.header.channel_type =
                PVR::ChannelType::Unsigned_Byte;
            compressedFile.header.height = ALIGN(max_height_1, 4);
            compressedFile.header.width = ALIGN(max_width_1, 4);
            compressedFile.header.depth = 1;
            compressedFile.header.num_faces = 1;
            compressedFile.header.num_surfaces = 1;
            compressedFile.header.mipmap_count = 1;
            compressedFile.header.metadata_size = 0;
            compressedFile.header.pixel_format = PVR::PixelFormat::BC7;
            compressedFile.pTextureData = _dst_buf.data();
            compressedFile.szTextureDataSize = compressed_size;

            auto outputFileName = pMaterial->GetName();
            if (argc >= 3) {
                outputFileName = argv[2];
            }
            outputFileName += "_1";

            cerr << "generate " << outputFileName << std::endl;
            std::ofstream outputFile(outputFileName.c_str(), std::ios::binary);

            outputFile << compressedFile;

            outputFile.close();
            cerr << "finished " << outputFileName << std::endl;

            // surf2
            surf2.width = max_width_2;
            surf2.height = max_height_2;
            surf2.stride = 4 * surf2.width;
            std::vector<uint8_t> buf2 (surf2.stride * surf2.height);
            surf2.ptr = buf2.data();

            for (int32_t y = 0; y < max_height_2; y++) {
                for (int32_t x = 0; x < max_width_2; x++) {
                    *(surf2.ptr + y * surf2.stride + x * 4) = metallic_texture->GetR(x, y);
                    *(surf2.ptr + y * surf2.stride + x * 4 + 1) = roughness_texture->GetR(x, y);
                    *(surf2.ptr + y * surf2.stride + x * 4 + 2) = ao_texture->GetR(x, y);
                    *(surf2.ptr + y * surf2.stride + x * 4 + 3) = normal_texture->GetY(x, y);
                }
            }

            // Now, compress surf2 with BC7
            compressed_size = (ALIGN(max_height_2, 4) >> 2) *
                                      (ALIGN(max_width_2, 4) >> 2) * 16;

            _dst_buf.resize(compressed_size);

            GetProfile_basic(&settings);
            CompressBlocksBC7(&surf1, _dst_buf.data(), &settings);

            compressedFile.header.height = ALIGN(max_height_2, 4);
            compressedFile.header.width = ALIGN(max_width_2, 4);
            compressedFile.pTextureData = _dst_buf.data();
            compressedFile.szTextureDataSize = compressed_size;

            outputFileName = pMaterial->GetName();
            if (argc >= 3) {
                outputFileName = argv[2];
            }
            outputFileName += "_2";

            cerr << "generate " << outputFileName << std::endl;
            outputFile.open(outputFileName.c_str(), std::ios::binary);

            outputFile << compressedFile;

            outputFile.close();
            cerr << "finished " << outputFileName << std::endl;
        }
    }

    app.Finalize();

    return error;
}

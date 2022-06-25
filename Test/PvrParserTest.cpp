#include <cstdio>
#include "PVR.hpp"
#include "AssetLoader.hpp"

using namespace My;

int main(int argc, char** argv) {
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

        //image.SaveTGA("PvrParserTest.tga");
    }

    assetLoader.Finalize();

    return error;
}
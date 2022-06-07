#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "TGA.hpp"

using namespace My;

int main(int argc, const char** argv) {
    int error = 0;
    AssetLoader assetLoader;
    assetLoader.Initialize();

    if (!error) {
        Buffer buf;
        if (argc >= 2) {
            buf = assetLoader.SyncOpenAndReadBinary(argv[1]);
        } else {
            buf = assetLoader.SyncOpenAndReadBinary(
                "Textures/interior_lod0.tga");
        }

        TgaParser tga_parser;

        Image image = tga_parser.Parse(buf);

        std::cout << image;
    }

    assetLoader.Finalize();

    return error;
}

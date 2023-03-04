#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "PNG.hpp"

using namespace My;

int main(int argc, const char** argv) {
    int error = 0;
    AssetLoader assetLoader;
    error = assetLoader.Initialize();

    if (!error) {
        Buffer buf;
        if (argc >= 2) {
            buf = assetLoader.SyncOpenAndReadBinary(argv[1]);
        } else {
            buf = assetLoader.SyncOpenAndReadBinary("Textures/eye.png");
        }

        PngParser png_parser;

        Image image = png_parser.Parse(buf);

        std::cout << image;
    }

    assetLoader.Finalize();

    return 0;
}

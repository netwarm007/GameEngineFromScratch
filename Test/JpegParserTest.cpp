#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "JPEG.hpp"

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
            buf = assetLoader.SyncOpenAndReadBinary(
                "Textures/huff_simple0.jpg");
        }

        JfifParser jfif_parser;

        Image image = jfif_parser.Parse(buf);

        std::cout << image;

        assetLoader.Finalize();
    }

    return error;
}

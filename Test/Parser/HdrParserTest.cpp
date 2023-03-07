#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "HDR.hpp"

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
                "Textures/hdr/PaperMill_E_3k.hdr");
        }

        HdrParser hdr_parser;

        Image image = hdr_parser.Parse(buf);

        std::cout << image;
    }

    assetLoader.Finalize();

    return error;
}

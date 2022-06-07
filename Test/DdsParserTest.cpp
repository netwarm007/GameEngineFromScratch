#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "DDS.hpp"

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
                "Textures/hdr/PaperMill_posx.dds");
        }

        DdsParser dds_parser;

        Image image = dds_parser.Parse(buf);

        std::cout << image;
    }

    assetLoader.Finalize();

    return error;
}

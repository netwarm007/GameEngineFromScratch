#include <cstdio>
#define DUMP_DETAILS 1
#include "ASTC.hpp"
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
                "Textures/viking_room.astc");
        }

        AstcParser astc_parser;

        Image image = astc_parser.Parse(buf);

        std::cout << image;
    }

    assetLoader.Finalize();

    return error;
}
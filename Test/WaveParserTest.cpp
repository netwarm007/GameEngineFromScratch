#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "WAVE.hpp"

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
                "Audio/test.wav");
        }

        WaveParser wave_parser;

        AudioClip audioClip = wave_parser.Parse(buf);

        std::cout << audioClip;
    }

    assetLoader.Finalize();

    return error;
}

#include <iostream>
#include <string>

#include "AssetLoader.hpp"

using namespace My;

int main(int, char**) {
    int error = 0;

    AssetLoader assetLoader;

    error = assetLoader.Initialize();

    if (!error) {
        std::string shader_pgm = assetLoader.SyncOpenAndReadTextFileToString(
            "Shaders/HLSL/basic.vert.hlsl");

        std::cout << shader_pgm;

        assetLoader.Finalize();
    }

    return error;
}

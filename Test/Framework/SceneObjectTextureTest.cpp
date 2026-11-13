#include <iostream>

#include "AssetLoader.hpp"
#include "SceneObjectTexture.hpp"

using namespace My;

int main() {
    AssetLoader assetLoader;
    if (assetLoader.Initialize() != 0) {
        std::cerr << "Failed to initialize the asset loader" << std::endl;
        return 1;
    }

    SceneObjectTexture texture;
    texture.SetName("Textures/texture_missing_test_asset.png");

    const auto image = texture.GetTextureImage();
    if (!image) {
        std::cerr << "Missing texture did not load the fallback image"
                  << std::endl;
        return 1;
    }

    if (image->Width != 64 || image->Height != 64 || !image->data ||
        image->pixel_format != PIXEL_FORMAT::RGBA8) {
        std::cerr << "Fallback image has unexpected dimensions or format"
                  << std::endl;
        return 1;
    }

    if (image->GetR(4, 4) != 230 || image->GetG(4, 4) != 35 ||
        image->GetB(4, 4) != 155) {
        std::cerr << "Fallback image does not contain the expected icon"
                  << std::endl;
        return 1;
    }

    return 0;
}

#pragma once
#include "AssetLoader.hpp"

namespace My{
    class BundleAssetLoader : public AssetLoader {
        public:
        BundleAssetLoader();
        using AssetLoader::AssetLoader;
    };
}

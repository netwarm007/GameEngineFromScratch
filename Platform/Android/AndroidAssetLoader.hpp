#pragma once
#include "AssetLoader.hpp"
#include <android/asset_manager.h>

namespace My {
    class AndroidAssetLoader : public AssetLoader {
        public:
            using AssetLoader::AssetLoader;
            AssetFilePtr OpenFile(const char* name, AssetOpenMode mode);
            void CloseFile(AssetFilePtr& fp);
            void SetPlatformAssetManager(AAssetManager* assetManager);
            Buffer SyncOpenAndReadText(const char* assetPath);
            Buffer SyncOpenAndReadBinary(const char* assetPath);

        protected:
            AAssetManager* m_pPlatformAssetManager = nullptr;
    };
}

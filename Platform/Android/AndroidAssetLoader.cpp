#include <string>
#include "AndroidAssetLoader.hpp"
#include "platform_defs.hpp"

using namespace My;
using namespace std;

void AndroidAssetLoader::SetPlatformAssetManager(AAssetManager* assetManager)
{
    m_pPlatformAssetManager = assetManager;
}

AndroidAssetLoader::AssetFilePtr AndroidAssetLoader::OpenFile(const char* name, AssetOpenMode mode)
{
    string assetPath = "assets/";
    assetPath.append(name);
    LOGI("Open Asset: %s", assetPath.c_str());
    AAsset* fp = AAssetManager_open(m_pPlatformAssetManager, assetPath.c_str(), AASSET_MODE_BUFFER);
    return (AssetFilePtr)fp;
}

Buffer AndroidAssetLoader::SyncOpenAndReadText(const char* assetPath)
{
    AAsset* fp = (AAsset*)OpenFile(assetPath, MY_OPEN_TEXT);
    Buffer* pBuff = nullptr;

    if (fp) {
        size_t fileLength = AAsset_getLength(fp);

        pBuff = new Buffer(fileLength + 1);
        AAsset_read(fp, pBuff->GetData(), fileLength);
        pBuff->GetData()[fileLength] = '\0';
        AAsset_close(fp);
    } else {
        fprintf(stderr, "Error opening asset file '%s'\n", assetPath);
        pBuff = new Buffer();
    }

    return *pBuff;
}

Buffer AndroidAssetLoader::SyncOpenAndReadBinary(const char* assetPath)
{
    AAsset* fp = (AAsset*)OpenFile(assetPath, MY_OPEN_BINARY);
    Buffer* pBuff = nullptr;

    if (fp) {
        size_t fileLength = AAsset_getLength(fp);

        pBuff = new Buffer(fileLength + 1);
        AAsset_read(fp, pBuff->GetData(), fileLength);
        AAsset_close(fp);
    } else {
        fprintf(stderr, "Error opening asset file '%s'\n", assetPath);
        pBuff = new Buffer();
    }

    return *pBuff;
}


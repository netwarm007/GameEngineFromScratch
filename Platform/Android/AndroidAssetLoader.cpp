#include "AndroidAssetLoader.hpp"

#include <string>

#include "platform_defs.hpp"

using namespace My;
using namespace std;

#include <errno.h>
#include <cstring>
#include <unistd.h>
#include "AssetLoader.hpp"

int AssetLoader::Initialize() {
    int ret = 0;
    char pathbuf[PATH_MAX];

    ret = readlink("/proc/self/exe", pathbuf, PATH_MAX);
    if (ret <= 0 || ret == PATH_MAX) {
        ret = -1;
    } else {
        m_strTargetPath = pathbuf;
        m_strTargetPath =
            m_strTargetPath.substr(0, m_strTargetPath.find_last_of('/') + 1);
        fprintf(stderr, "Working Dir: %s\n", m_strTargetPath.c_str());
        ret = 0;
    }

    return ret;
}

void AndroidAssetLoader::SetPlatformAssetManager(AAssetManager* assetManager) {
    m_pPlatformAssetManager = assetManager;
}

AndroidAssetLoader::AssetFilePtr AndroidAssetLoader::OpenFile(
    const char* name, AssetOpenMode mode) {
    LOGI("Open Asset: %s", name);
    AAsset* fp = nullptr;
    ;
    if (m_pPlatformAssetManager)
        fp = AAssetManager_open(m_pPlatformAssetManager, name,
                                AASSET_MODE_BUFFER);
    else {
        LOGE("m_pPlatfornAssetManager is null!");
    }
    LOGI("fp: %p", fp);
    return (AssetFilePtr)fp;
}

void AndroidAssetLoader::CloseFile(AssetFilePtr& fp) {
    LOGI("Close Asset: %p", fp);
    AAsset* _fp = (AAsset*)fp;

    if (m_pPlatformAssetManager)
        AAsset_close(_fp);
    else {
        LOGE("m_pPlatfornAssetManager is null!");
    }
}

Buffer AndroidAssetLoader::SyncOpenAndReadText(const char* assetPath) {
    AAsset* fp = (AAsset*)OpenFile(assetPath, MY_OPEN_TEXT);
    Buffer buffer;

    if (fp) {
        size_t fileLength = AAsset_getLength(fp);
        LOGD("asset file size: %zu", fileLength);

        AAsset_read(fp, buffer.GetData(), fileLength);
        buffer.GetData()[fileLength] = '\0';
        AAsset_close(fp);
    } else {
        LOGE("Error opening asset file '%s'", assetPath);
    }

    return buffer;
}

Buffer AndroidAssetLoader::SyncOpenAndReadBinary(const char* assetPath) {
    AAsset* fp = (AAsset*)OpenFile(assetPath, MY_OPEN_BINARY);
    Buffer buffer;

    if (fp) {
        size_t fileLength = AAsset_getLength(fp);
        LOGD("asset file size: %zu", fileLength);

        AAsset_read(fp, buffer.GetData(), fileLength);
        AAsset_close(fp);
    } else {
        LOGE("Error opening asset file '%s'", assetPath);
    }

    return buffer;
}

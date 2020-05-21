#include "AssetLoader.hpp"

using namespace My;
using namespace std;

void AssetLoader::ClearSearchPath() { m_strSearchPath.clear(); }

bool AssetLoader::AddSearchPath(const char* path) {
    auto src = m_strSearchPath.begin();

    while (src != m_strSearchPath.end()) {
        if (*src == path) return true;
        src++;
    }

    m_strSearchPath.emplace_back(path);
    return true;
}

bool AssetLoader::RemoveSearchPath(const char* path) {
    auto src = m_strSearchPath.begin();

    while (src != m_strSearchPath.end()) {
        if (*src == path) {
            m_strSearchPath.erase(src);
            return true;
        }
        src++;
    }

    return true;
}

bool AssetLoader::FileExists(const char* filePath) {
    AssetFilePtr fp = OpenFile(filePath, MY_OPEN_BINARY);
    if (fp != nullptr) {
        CloseFile(fp);
        return true;
    }
    return false;
}

AssetLoader::AssetFilePtr AssetLoader::OpenFile(const char* name,
                                                AssetOpenMode mode) {
    FILE* fp = nullptr;
    // loop N times up the hierarchy, testing at each level
#ifdef __psp2__
    std::string upPath = "app0:/";
#elseif __ORBIS__
    std::string upPath = "/app0/";
#else
    std::string upPath;
#endif
    std::string fullPath;
    for (int32_t i = 0; i < 10; i++) {
        auto src = m_strSearchPath.begin();
        bool looping = true;
        while (looping) {
            fullPath.assign(upPath);  // reset to current upPath.
            if (src != m_strSearchPath.end()) {
                fullPath.append(*src);
                fullPath.append("/Asset/");
                src++;
            } else {
                fullPath.append("Asset/");
                looping = false;
            }
            fullPath.append(name);

            switch (mode) {
                case MY_OPEN_TEXT:
                    fp = fopen(fullPath.c_str(), "r");
                    break;
                case MY_OPEN_BINARY:
                    fp = fopen(fullPath.c_str(), "rb");
                    break;
            }

            if (fp) {
                return (AssetFilePtr)fp;
            }
        }

        upPath.append("../");
    }

    return nullptr;
}

Buffer AssetLoader::SyncOpenAndReadText(const char* filePath) {
    AssetFilePtr fp = OpenFile(filePath, MY_OPEN_TEXT);
    Buffer buff;

    if (fp) {
        size_t length = GetSize(fp);

        uint8_t* data = new uint8_t[length + 1];
        length = fread(data, 1, length, static_cast<FILE*>(fp));
#ifdef DEBUG
        fprintf(stderr, "Read file '%s', %zu bytes\n", filePath, length);
#endif

        data[length] = '\0';
        buff.SetData(data, length + 1);

        CloseFile(fp);
    } else {
        fprintf(stderr, "Error opening file '%s'\n", filePath);
    }

    return buff;
}

Buffer AssetLoader::SyncOpenAndReadBinary(const char* filePath) {
    AssetFilePtr fp = OpenFile(filePath, MY_OPEN_BINARY);
    Buffer buff;

    if (fp) {
        size_t length = GetSize(fp);

        uint8_t* data = new uint8_t[length];
        fread(data, length, 1, static_cast<FILE*>(fp));
#ifdef DEBUG
        fprintf(stderr, "Read file '%s', %zu bytes\n", filePath, length);
#endif
        buff.SetData(data, length);

        CloseFile(fp);
    } else {
        fprintf(stderr, "Error opening file '%s'\n", filePath);
    }

    return buff;
}

void AssetLoader::CloseFile(AssetFilePtr& fp) {
    fclose((FILE*)fp);
    fp = nullptr;
}

size_t AssetLoader::GetSize(const AssetFilePtr& fp) {
    FILE* _fp = static_cast<FILE*>(fp);

    long pos = ftell(_fp);
    fseek(_fp, 0, SEEK_END);
    size_t length = ftell(_fp);
    fseek(_fp, pos, SEEK_SET);

    return length;
}

size_t AssetLoader::SyncRead(const AssetFilePtr& fp, Buffer& buf) {
    size_t sz;

    if (!fp) {
        fprintf(stderr, "null file discriptor\n");
        return 0;
    }

    sz = fread(buf.GetData(), buf.GetDataSize(), 1, static_cast<FILE*>(fp));

    return sz;
}

int32_t AssetLoader::Seek(AssetFilePtr fp, long offset, AssetSeekBase where) {
    return fseek(static_cast<FILE*>(fp), offset, static_cast<int>(where));
}

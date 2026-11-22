#include "AssetLoader.hpp"

using namespace My;
using namespace std;

std::string AssetLoader::m_strTargetPath;

std::vector<std::string> AssetLoader::m_strSearchPath;

namespace {
// Upper bound for a single asset file read into memory (512 MB)
constexpr size_t kMaxAssetFileSize = 512ull * 1024 * 1024;

// Reject asset names that could escape the configured search roots: absolute
// paths, drive specs, UNC prefixes and any ".." path segment.
bool IsSafeAssetPath(const char* name) {
    if (name == nullptr || name[0] == '\0') return false;

    const std::string path(name);
    for (size_t i = 0; i < path.size(); i++) {
        const char c = path[i];
        if (c == '/' || c == '\\') {
            if (i == 0) return false;  // rooted path
            if (i + 1 < path.size() && path[i + 1] == '/' &&
                path[i + 2] == '/')
                return false;  // UNC prefix
        } else if (c == ':') {
            return false;  // drive spec or embedded scheme
        }
    }

    // scan for ".." segments
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find_first_of("/\\", start);
        if (end == std::string::npos) end = path.size();
        if (end - start == 2 && path.compare(start, 2, "..") == 0)
            return false;
        start = end + 1;
    }

    return true;
}
}  // namespace

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

std::string AssetLoader::GetFileRealPath(const char* filePath) {
    if (!IsSafeAssetPath(filePath)) {
        fprintf(stderr, "Rejected unsafe asset path '%s'\n",
                filePath ? filePath : "(null)");
        return std::string();
    }

    FILE* fp = nullptr;
    // loop N times up the hierarchy, testing at each level
    std::string upPath(m_strTargetPath);
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
            fullPath.append(filePath);

            fp = fopen(fullPath.c_str(), "r");

            if (fp) {
                fclose(fp);
                return fullPath;
            }
        }

        upPath.append("../");
    }

    return std::string();
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
    if (!IsSafeAssetPath(name)) {
        fprintf(stderr, "Rejected unsafe asset path '%s'\n",
                name ? name : "(null)");
        return nullptr;
    }

    FILE* fp = nullptr;
    // loop N times up the hierarchy, testing at each level
    std::string upPath(m_strTargetPath);
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

        if (length > 0 && length <= kMaxAssetFileSize) {
            uint8_t* data = new uint8_t[length + 1];
            length = fread(data, 1, length, static_cast<FILE*>(fp));
#ifdef DEBUG
            fprintf(stderr, "Read file '%s', %zu bytes\n", filePath, length);
#endif

            data[length] = '\0';
            buff.SetData(data, length + 1);
        } else if (length > kMaxAssetFileSize) {
            fprintf(stderr, "File '%s' too large (%zu bytes), rejected\n",
                    filePath, length);
        }

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

        if (length > 0 && length <= kMaxAssetFileSize) {
            uint8_t* data = new uint8_t[length];
            length = fread(data, 1, length, static_cast<FILE*>(fp));
#ifdef DEBUG
            fprintf(stderr, "Read file '%s', %zu bytes\n", filePath, length);
#endif
            if (length > 0) {
                buff.SetData(data, length);
            } else {
                delete[] data;
            }
        } else if (length > kMaxAssetFileSize) {
            fprintf(stderr, "File '%s' too large (%zu bytes), rejected\n",
                    filePath, length);
        }

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
    if (pos < 0) return 0;
    if (fseek(_fp, 0, SEEK_END) != 0) return 0;
    long end = ftell(_fp);
    if (fseek(_fp, pos, SEEK_SET) != 0) return 0;
    if (end < pos) return 0;

    return (size_t)(end - pos);
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

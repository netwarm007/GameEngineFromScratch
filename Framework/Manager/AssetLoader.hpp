#pragma once

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "IAssetLoader.hpp"
#include "IRuntimeModule.hpp"

namespace My {
class AssetLoader : _implements_ IAssetLoader, _implements_ IRuntimeModule {
   public:
    AssetLoader() = default;
    ~AssetLoader() override = default;
    int Initialize() override { return 0; }
    void Finalize() override {}
    void Tick() override {}

    bool AddSearchPath(const char* path) override;

    bool RemoveSearchPath(const char* path) override;

    void ClearSearchPath() override;

    bool FileExists(const char* filePath) override;

    AssetFilePtr OpenFile(const char* name, AssetOpenMode mode) override;

    Buffer SyncOpenAndReadText(const char* filePath) override;

    Buffer SyncOpenAndReadBinary(const char* filePath) override;

    size_t SyncRead(const AssetFilePtr& fp, Buffer& buf) override;

    void CloseFile(AssetFilePtr& fp) override;

    size_t GetSize(const AssetFilePtr& fp) override;

    int32_t Seek(AssetFilePtr fp, long offset, AssetSeekBase where) override;

    inline std::string SyncOpenAndReadTextFileToString(
        const char* fileName) override {
        std::string result;
        Buffer buffer = SyncOpenAndReadText(fileName);
        if (buffer.GetDataSize()) {
            char* content = reinterpret_cast<char*>(buffer.GetData());

            if (content) {
                result = std::string(content);
            }
        }

        return result;
    }

   protected:
    std::string m_strTargetPath;

   private:
    std::vector<std::string> m_strSearchPath;
};
}  // namespace My

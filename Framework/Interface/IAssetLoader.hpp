#pragma once
#include <string>
#include "Buffer.hpp"
#include "IRuntimeModule.hpp"

namespace My {
_Interface_ IAssetLoader : _inherits_ IRuntimeModule {
   public:
    IAssetLoader() = default;
    virtual ~IAssetLoader() = default;
    using AssetFilePtr = void*;

    enum AssetOpenMode {
        MY_OPEN_TEXT = 0,    /// Open In Text Mode
        MY_OPEN_BINARY = 1,  /// Open In Binary Mode
    };

    enum AssetSeekBase {
        MY_SEEK_SET = 0,  /// SEEK_SET
        MY_SEEK_CUR = 1,  /// SEEK_CUR
        MY_SEEK_END = 2   /// SEEK_END
    };

    virtual bool AddSearchPath(const char* path) = 0;

    virtual bool RemoveSearchPath(const char* path) = 0;

    virtual void ClearSearchPath() = 0;

    virtual bool FileExists(const char* filePath) = 0;

    virtual AssetFilePtr OpenFile(const char* name, AssetOpenMode mode) = 0;

    virtual Buffer SyncOpenAndReadText(const char* filePath) = 0;

    virtual Buffer SyncOpenAndReadBinary(const char* filePath) = 0;

    virtual std::string SyncOpenAndReadTextFileToString(const char* fileName) = 0;

    virtual size_t SyncRead(const AssetFilePtr& fp, Buffer& buf) = 0;

    virtual void CloseFile(AssetFilePtr & fp) = 0;

    virtual size_t GetSize(const AssetFilePtr& fp) = 0;

    virtual int32_t Seek(AssetFilePtr fp, long offset, AssetSeekBase where) = 0;
};
}  // namespace My
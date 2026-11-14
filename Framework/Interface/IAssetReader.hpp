#pragma once

#include <string>

#include "Buffer.hpp"
#include "Interface.hpp"

namespace My {
_Interface_ IAssetReader {
   public:
    virtual ~IAssetReader() = default;
    virtual Buffer SyncOpenAndReadBinary(const char* filePath) = 0;
    virtual std::string SyncOpenAndReadTextFileToString(const char* fileName) = 0;
};
}  // namespace My

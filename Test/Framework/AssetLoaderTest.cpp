#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

#include "AssetLoader.hpp"

using namespace My;

namespace {
class TestBinaryLoader : public AssetLoader {
   public:
    TestBinaryLoader(FILE* file, size_t expectedSize)
        : m_file(file), m_expectedSize(expectedSize) {}

    AssetFilePtr OpenFile(const char*, AssetOpenMode) override {
        return std::exchange(m_file, nullptr);
    }

    size_t GetSize(const AssetFilePtr&) override { return m_expectedSize; }

   private:
    FILE* m_file;
    size_t m_expectedSize;
};

bool TestBinaryRead(const char* data, size_t size, size_t expectedSize,
                    bool expectSuccess) {
    FILE* file = std::tmpfile();
    if (!file) return false;
    if (std::fwrite(data, 1, size, file) != size ||
        std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return false;
    }
    TestBinaryLoader loader(file, expectedSize);
    Buffer buffer = loader.SyncOpenAndReadBinary("test binary");
    if (!expectSuccess) return !buffer.GetData() && buffer.GetDataSize() == 0;
    if (size == 0) return !buffer.GetData() && buffer.GetDataSize() == 0;
    return buffer.GetData() && buffer.GetDataSize() == size &&
           std::memcmp(buffer.GetData(), data, size) == 0;
}
}  // namespace

int main(int, char**) {
    int error = 0;

    AssetLoader assetLoader;

    error = assetLoader.Initialize();

    if (!error) {
        std::string shader_pgm = assetLoader.SyncOpenAndReadTextFileToString(
            "Shaders/HLSL/basic.vert.hlsl");

        std::cout << shader_pgm;

        assetLoader.Finalize();
    }

    const char data[] = {'a', '\0', 'b'};
    if (!TestBinaryRead(data, sizeof(data), sizeof(data), true) ||
        !TestBinaryRead(data, 0, 0, true) ||
        !TestBinaryRead(data, sizeof(data), sizeof(data) + 1, false)) {
        std::cerr << "Binary asset read regression\n";
        return 1;
    }

    return error;
}

#include "GuiIconLoader.hpp"

#include "AssetLoader.hpp"
#include "PNG.hpp"

namespace My {
Image LoadGuiIconImage(const char* path) {
    AssetLoader loader;
    if (!loader.FileExists(path)) {
        std::cerr << "[GuiIconLoader] Icon not found: " << path << std::endl;
        return {};
    }
    Buffer buffer = loader.SyncOpenAndReadBinary(path);
    PngParser parser;
    Image image = parser.Parse(buffer);
    if (!image.data) {
        std::cerr << "[GuiIconLoader] Failed to decode icon: " << path << std::endl;
    }
    return image;
}
}  // namespace My

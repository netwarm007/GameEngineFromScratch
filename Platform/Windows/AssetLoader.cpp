
#include "AssetLoader.hpp"
#include <Windows.h>
#include <string_view>

namespace My {
int AssetLoader::Initialize() {
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, sizeof(buffer));
    std::string::size_type pos = std::string_view(buffer).find_last_of("\\/");

    m_strTargetPath = std::string_view(buffer).substr(0, pos);

    return 0;
}
}  // namespace My
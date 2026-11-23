#include "AssetLoader.hpp"

#include <SDL2/SDL.h>

namespace My {
int AssetLoader::Initialize() {
    // Resolve assets relative to the executable (build tree), not the
    // process working directory: the source Asset/ tree lacks the built
    // Shaders/Materials staged into the build tree.
    char* base = SDL_GetBasePath();
    if (!base) return -1;
    m_strTargetPath.assign(base);
    SDL_free(base);
    fprintf(stderr, "Working Dir: %s\n", m_strTargetPath.c_str());
    return 0;
}
}  // namespace My

#include "AssetLoader.hpp"
#include <errno.h>
#include <cstring>

namespace My {
int AssetLoader::Initialize() { 
    int ret = 0;
    char pathbuf[PATH_MAX];

    ret = readlink("/proc/self/exe", pathbuf, PATH_MAX);
    if (ret <= 0 || ret == PATH_MAX)
    {
        ret = -1;
    } else {
        m_strTargetPath = pathbuf;
        m_strTargetPath = m_strTargetPath.substr(0, m_strTargetPath.find_last_of('/') + 1);
        fprintf(stderr, "Working Dir: %s\n", m_strTargetPath.c_str());
        ret = 0;
    }

    return ret; 
}
}  // namespace My

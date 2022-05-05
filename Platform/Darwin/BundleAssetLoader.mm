#include "BundleAssetLoader.h"

#include <mach-o/dyld.h>
#include <sys/syslimits.h>

namespace My {
    BundleAssetLoader::BundleAssetLoader() {
        char path[PATH_MAX + 1];
        path[0] = '\0';
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) == 0) {
            m_strTargetPath = path;
            m_strTargetPath = m_strTargetPath.substr(0, m_strTargetPath.find_last_of('/') + 1);
        }

        AddSearchPath("Resources");
    }
}
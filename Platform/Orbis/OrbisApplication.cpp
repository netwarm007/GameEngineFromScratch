#include "OrbisApplication.hpp"
#include "GraphicsManager.hpp"
#include "MemoryManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

int OrbisApplication::Initialize()
{
    int ret;
    ret = BaseApplication::Initialize();
    if (!ret)
    {
        g_pAssetLoader->AddSearchPath("/app0");
        g_pAssetLoader->AddSearchPath("/hostapp");
    }

    return ret;
}



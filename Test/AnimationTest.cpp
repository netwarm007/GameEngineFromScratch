#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include "AssetLoader.hpp"
#include "MemoryManager.hpp"
#include "SceneManager.hpp"
#include "AnimationManager.hpp"

using namespace My;
using namespace std;

namespace My {
    IMemoryManager*  g_pMemoryManager = new MemoryManager();
    AssetLoader*    g_pAssetLoader   = new AssetLoader();
    SceneManager*   g_pSceneManager  = new SceneManager();
    AnimationManager* g_pAnimationManager = new AnimationManager();
}

template<typename T>
static ostream& operator<<(ostream& out, unordered_map<string, shared_ptr<T>> map)
{
    for (auto p : map)
    {
        out << *p.second << endl;
    }

    return out;
}

int main(int argc, char** argv)
{
    g_pMemoryManager->Initialize();
    g_pSceneManager->Initialize();
    g_pAssetLoader->Initialize();

    if (argc >= 2) {
        g_pSceneManager->LoadScene(argv[1]);
    }
    else
    {
        g_pSceneManager->LoadScene("Scene/splash.ogex");
    }

    g_pAnimationManager->Initialize();

    auto& scene = g_pSceneManager->GetSceneForRendering();

    cout.precision(4);
    cout.setf(ios::fixed);

    for (auto i = 0; i < 250; i++)
    {
        cout << "Tick #" << i << endl;
        g_pAnimationManager->Tick();
        cout << "Dump of Animatable Nodes" << endl;
        cout << "---------------------------" << endl;
        for (auto node : scene.AnimatableNodes)
        {
            auto pNode = node.lock();
            if (pNode) {
                cout << *pNode->GetCalculatedTransform() << endl;
            }
        }
        const chrono::milliseconds one_frame_time(33);
        this_thread::sleep_for(one_frame_time);
    }

    g_pAnimationManager->Finalize();
    g_pSceneManager->Finalize();
    g_pAssetLoader->Finalize();
    g_pMemoryManager->Finalize();

    delete g_pAnimationManager;
    delete g_pSceneManager;
    delete g_pAssetLoader;
    delete g_pMemoryManager;

    return 0;
}


#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "BaseApplication.hpp"
#include "AnimationManager.hpp"
#include "AssetLoader.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

template <typename T>
static ostream& operator<<(ostream& out,
                           unordered_map<string, shared_ptr<T>> map) {
    for (auto p : map) {
        out << *p.second << endl;
    }

    return out;
}

int main(int argc, char** argv) {
    int error = 0; 
    BaseApplication app;
    AssetLoader assetLoader;
    SceneManager sceneManager;
    AnimationManager animationManager;

    app.RegisterManagerModule(&assetLoader, &assetLoader);
    app.RegisterManagerModule(&sceneManager, &sceneManager);
    app.RegisterManagerModule(&animationManager, &animationManager);

    error = app.Initialize();

    if (argc >= 2) {
        sceneManager.LoadScene(argv[1]);
    } else {
        sceneManager.LoadScene("Scene/splash.ogex");
    }

    animationManager.Initialize();

    auto& scene = sceneManager.GetSceneForRendering();

    cout.precision(4);
    cout.setf(ios::fixed);

    for (auto i = 0; i < 300; i++) {
        animationManager.Tick();
        if (i % 10 == 0) {
            cout << "Tick #" << i << endl;
            cout << "Dump of Animatable Nodes" << endl;
            cout << "---------------------------" << endl;
            for (const auto& node : scene->AnimatableNodes) {
                auto pNode = node.lock();
                if (pNode) {
                    cout << *pNode->GetCalculatedTransform() << endl;
                }
            }
        }

        const chrono::milliseconds one_frame_time(16); // 16ms
        this_thread::sleep_for(one_frame_time);
    }

    app.Finalize();

    return error;
}

#include "BilliardGameLogic.hpp"

#include <iostream>

#include "IPhysicsManager.hpp"
#include "ISceneManager.hpp"

using namespace My;
using namespace std;

int BilliardGameLogic::Initialize() {
    cout << "[BilliardGameLogic] Biiliard Game Logic Initialize" << endl;
    cout << "[BilliardGameLogic] Start Loading Game Scene" << endl;
    return m_Scenes.LoadScene("Scene/billiard.ogex");
}

void BilliardGameLogic::Finalize() {
    cout << "Biiliard Game Logic Finalize" << endl;
}

void BilliardGameLogic::Tick() {}

void BilliardGameLogic::OnLeftKey() {
    auto ptr = m_Scenes.GetSceneGeometryNode("pbb_cue");
    if (auto node = ptr.lock()) {
        m_Physics.ApplyCentralForce(
            *node, Vector3f({-100.0f, 0.0f, 0.0f}));
    }
}

void BilliardGameLogic::OnAnalogStick(int id, float deltaX, float deltaY) {
    if (id == 1) {
        auto& scene = m_Scenes.GetSceneForRendering();
        if (scene) {
            auto pCameraNode = scene->GetFirstCameraNode();
            if (pCameraNode) {
                auto screen_width = m_pApp->GetConfiguration().screenWidth;
                auto screen_height = m_pApp->GetConfiguration().screenHeight;
                // move camera along its local axis -y direction
                pCameraNode->RotateBy(deltaX / screen_width * PI,
                                    deltaY / screen_height * PI, 0.0f);
            }
        }
    }
}

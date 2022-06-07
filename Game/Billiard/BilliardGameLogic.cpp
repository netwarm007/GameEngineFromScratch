#include "BilliardGameLogic.hpp"

#include <iostream>

#include "GraphicsManager.hpp"
#include "IPhysicsManager.hpp"
#include "SceneManager.hpp"
#include "BaseApplication.hpp"

using namespace My;
using namespace std;

int BilliardGameLogic::Initialize() {
    int result;

    cout << "[BilliardGameLogic] Biiliard Game Logic Initialize" << endl;
    cout << "[BilliardGameLogic] Start Loading Game Scene" << endl;
    auto pSceneManager = dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    result = pSceneManager->LoadScene("Scene/billiard.ogex");

    return result;
}

void BilliardGameLogic::Finalize() {
    cout << "Biiliard Game Logic Finalize" << endl;
}

void BilliardGameLogic::Tick() {}

void BilliardGameLogic::OnLeftKey() {
    auto pSceneManager = dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();
    auto pPhysicsManager = dynamic_cast<BaseApplication*>(m_pApp)->GetPhysicsManager();

    auto ptr = pSceneManager->GetSceneGeometryNode("pbb_cue");
    if (auto node = ptr.lock()) {
        auto rigidBody = node->RigidBody();
        if (rigidBody) {
            pPhysicsManager->ApplyCentralForce(
                rigidBody, Vector3f({-100.0f, 0.0f, 0.0f}));
        }
    }
}

void BilliardGameLogic::OnAnalogStick(int id, float deltaX, float deltaY) {
    auto pSceneManager = dynamic_cast<BaseApplication*>(m_pApp)->GetSceneManager();

    if (id == 1) {
        auto& scene = pSceneManager->GetSceneForRendering();
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

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
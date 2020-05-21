#include "BilliardGameLogic.hpp"

#include <iostream>

#include "GraphicsManager.hpp"
#include "IPhysicsManager.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

int BilliardGameLogic::Initialize() {
    int result;

    cout << "[BilliardGameLogic] Biiliard Game Logic Initialize" << endl;
    cout << "[BilliardGameLogic] Start Loading Game Scene" << endl;
    result = g_pSceneManager->LoadScene("Scene/billiard.ogex");

    return result;
}

void BilliardGameLogic::Finalize() {
    cout << "Biiliard Game Logic Finalize" << endl;
}

void BilliardGameLogic::Tick() {}

void BilliardGameLogic::OnLeftKey() {
    auto ptr = g_pSceneManager->GetSceneGeometryNode("pbb_cue");
    if (auto node = ptr.lock()) {
        auto rigidBody = node->RigidBody();
        if (rigidBody) {
            g_pPhysicsManager->ApplyCentralForce(
                rigidBody, Vector3f({-100.0f, 0.0f, 0.0f}));
        }
    }
}
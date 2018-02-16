#include <iostream>
#include "article_41_logic.hpp"
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "IPhysicsManager.hpp"

using namespace My;
using namespace std;

int article_41_logic::Initialize()
{
    int result;

    cout << "[GameLogic] Initialize" << endl;
    cout << "[GameLogic] Start Loading Game Scene" << endl;
    result = g_pSceneManager->LoadScene("Scene/test.ogex");

    return result;
}

void article_41_logic::Finalize()
{
    cout << "[GameLogic] Finalize" << endl;
}

void article_41_logic::Tick()
{

}

void article_41_logic::OnLeftKey()
{
    auto node_weak_ptr = g_pSceneManager->GetSceneGeometryNode("Suzanne");
    if(auto node = node_weak_ptr.lock())
    {
        node->RotateBy(-PI/6.0f, 0.0f, 0.0f);
    }
}

void article_41_logic::OnRightKey()
{
    auto node_weak_ptr = g_pSceneManager->GetSceneGeometryNode("Suzanne");
    if(auto node = node_weak_ptr.lock())
    {
        node->RotateBy(PI/6.0f, 0.0f, 0.0f);
    }
}

void article_41_logic::OnUpKey()
{
    auto node_weak_ptr = g_pSceneManager->GetSceneGeometryNode("Suzanne");
    if(auto node = node_weak_ptr.lock())
    {
        node->RotateBy(0.0f, 0.0f, PI/6.0f);
    }
}

void article_41_logic::OnDownKey()
{
    auto node_weak_ptr = g_pSceneManager->GetSceneGeometryNode("Suzanne");
    if(auto node = node_weak_ptr.lock())
    {
        node->RotateBy(0.0f, 0.0f, -PI/6.0f);
    }
}
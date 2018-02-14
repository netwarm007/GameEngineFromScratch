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
    result = g_pSceneManager->LoadScene("Scene/article_41.ogex");

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

}
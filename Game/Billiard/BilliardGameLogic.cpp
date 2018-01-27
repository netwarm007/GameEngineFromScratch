#include <iostream>
#include "BilliardGameLogic.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

int BilliardGameLogic::Initialize()
{
    int result;

    result = g_pSceneManager->LoadScene("Scene/billiard.ogex");

    return result;
}

void BilliardGameLogic::Finalize()
{

}

void BilliardGameLogic::Tick()
{

}

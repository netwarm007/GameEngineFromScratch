#include <iostream>
#include "BilliardGameLogic.hpp"
#include "SceneManager.hpp"

using namespace My;
using namespace std;

int BilliardGameLogic::Initialize()
{
    int result;

    cout << "Biiliard Game Logic Initialize" << endl;
    cout << "Start Loading Game Scene" << endl;
    result = g_pSceneManager->LoadScene("Scene/billiard.ogex");

    return result;
}

void BilliardGameLogic::Finalize()
{
    cout << "Biiliard Game Logic Finalize" << endl;
}

void BilliardGameLogic::Tick()
{

}

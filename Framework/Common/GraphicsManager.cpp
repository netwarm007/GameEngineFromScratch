#include <iostream>
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "cbuffer.h"

using namespace My;
using namespace std;

int GraphicsManager::Initialize()
{
    int result = 0;
    return result;
}

void GraphicsManager::Finalize()
{
}

void GraphicsManager::Tick()
{
    if (g_pSceneManager->IsSceneChanged())
    {
        cout << "Detected Scene Change, reinitialize Graphics Manager..." << endl;
        Finalize();
        Initialize();
    }
}

void GraphicsManager::Clear()
{
}

void GraphicsManager::Draw()
{
}


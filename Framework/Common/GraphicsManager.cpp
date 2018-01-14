#include <iostream>
#include "GraphicsManager.hpp"
#include "SceneManager.hpp"
#include "cbuffer.h"

using namespace My;
using namespace std;

int GraphicsManager::Initialize()
{
    int result = 0;

    // Initialize the world/model matrix to the identity matrix.
    BuildIdentityMatrix(m_DrawFrameContext.m_worldMatrix);

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


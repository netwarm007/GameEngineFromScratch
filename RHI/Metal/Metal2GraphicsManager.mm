#import <Metal/Metal.h>

#include "Metal2GraphicsManager.h"

using namespace My;

int Metal2GraphicsManager::Initialize()
{
    int result;

    result = GraphicsManager::Initialize();

    return result;
}

void Metal2GraphicsManager::Finalize()
{
    GraphicsManager::Finalize();
}

void Metal2GraphicsManager::Clear()
{
    GraphicsManager::Clear();
}

void Metal2GraphicsManager::Draw()
{
    GraphicsManager::Draw();
    [m_pRenderer drawFrameNumber:0];
}

bool Metal2GraphicsManager::CheckCapability(RHICapability cap)
{
    return true;
}

void Metal2GraphicsManager::BeginScene(const Scene& scene)
{

}

void Metal2GraphicsManager::EndScene()
{

}

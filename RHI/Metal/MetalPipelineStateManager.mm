#include "MetalPipelineStateManager.h"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

int MetalPipelineStateManager::Initialize()
{
    return 0;
}

void MetalPipelineStateManager::Finalize()
{
    ClearShaders();
}

void MetalPipelineStateManager::Tick()
{

}

bool MetalPipelineStateManager::InitializeShaders()
{
    return true;
}

void MetalPipelineStateManager::ClearShaders()
{

}

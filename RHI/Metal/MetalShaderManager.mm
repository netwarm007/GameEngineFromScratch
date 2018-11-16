#include "MetalShaderManager.h"
#include "AssetLoader.hpp"

using namespace My;
using namespace std;

int MetalShaderManager::Initialize()
{
    return 0;
}

void MetalShaderManager::Finalize()
{
    ClearShaders();
}

void MetalShaderManager::Tick()
{

}

bool MetalShaderManager::InitializeShaders()
{
    return true;
}

void MetalShaderManager::ClearShaders()
{

}

#include "SceneManager.hpp"
#include "AssetLoader.hpp"
#include "OGEX.hpp"

using namespace My;
using namespace std;

SceneManager::~SceneManager()
{
}

int SceneManager::Initialize()
{
    int result = 0;
    return result;
}

void SceneManager::Finalize()
{
}

void SceneManager::Tick()
{
}

void SceneManager::LoadOgexScene(const char* scene_file_name)
{
    AssetLoader asset_loader;
    string ogex_text = asset_loader.SyncOpenAndReadTextFileToString(scene_file_name);

    OgexParser ogex_parser;
    m_RootNode = ogex_parser.Parse(ogex_text);
}



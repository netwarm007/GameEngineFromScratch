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

void SceneManager::LoadScene(const char* scene_file_name)
{
    // now we only has ogex scene parser, call it directly
    LoadOgexScene(scene_file_name);
}

void SceneManager::LoadOgexScene(const char* ogex_scene_file_name)
{
    AssetLoader asset_loader;
    string ogex_text = asset_loader.SyncOpenAndReadTextFileToString(ogex_scene_file_name);

    OgexParser ogex_parser;
    m_RootNode = ogex_parser.Parse(ogex_text);
}

const shared_ptr<SceneObjectCamera> SceneManager::GetFirstCamera()
{
    return m_Cameras.front();
}

const shared_ptr<SceneObjectCamera> SceneManager::GetNextCamera()
{
    static thread_local decltype(m_Cameras.cbegin()) _it = m_Cameras.cbegin();
    return ((_it == m_Cameras.cend()) ? *++_it : nullptr);
}

const shared_ptr<SceneObjectLight> SceneManager::GetFirstLight()
{
    return m_Lights.front();
}

const shared_ptr<SceneObjectLight> SceneManager::GetNextLight()
{
    static thread_local decltype(m_Lights.cbegin()) _it = m_Lights.cbegin();
    return ((_it == m_Lights.cend()) ? *++_it : nullptr);
}

const shared_ptr<SceneObjectMaterial> SceneManager::GetFirstMaterial()
{
    return m_Materials.front();
}

const shared_ptr<SceneObjectMaterial> SceneManager::GetNextMaterial()
{
    static thread_local decltype(m_Materials.cbegin()) _it = m_Materials.cbegin();
    return ((_it == m_Materials.cend()) ? *++_it : nullptr);
}

const shared_ptr<SceneObjectGeometry> SceneManager::GetFirstGeometry()
{
    return m_Geometries.front();
}

const shared_ptr<SceneObjectGeometry> SceneManager::GetNextGeometry()
{
    static thread_local decltype(m_Geometries.cbegin()) _it = m_Geometries.cbegin();
    return ((_it == m_Geometries.cend()) ? *++_it : nullptr);
}


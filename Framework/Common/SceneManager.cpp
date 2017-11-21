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
    string ogex_text = g_pAssetLoader->SyncOpenAndReadTextFileToString(ogex_scene_file_name);

    OgexParser ogex_parser;
    m_pScene = ogex_parser.Parse(ogex_text);
}

const shared_ptr<SceneObjectCamera> SceneManager::GetFirstCamera()
{
    if (!m_pScene) return nullptr;
    return (m_pScene->Cameras.empty()? nullptr : m_pScene->Cameras.front());
}

const shared_ptr<SceneObjectCamera> SceneManager::GetNextCamera()
{
    static thread_local decltype(m_pScene->Cameras.cbegin()) _it = m_pScene->Cameras.cbegin();
    return ((_it == m_pScene->Cameras.cend()) ? *++_it : nullptr);
}

const shared_ptr<SceneObjectLight> SceneManager::GetFirstLight()
{
    if (!m_pScene) return nullptr;
    return (m_pScene->Lights.empty()? nullptr : m_pScene->Lights.front());
}

const shared_ptr<SceneObjectLight> SceneManager::GetNextLight()
{
    static thread_local decltype(m_pScene->Lights.cbegin()) _it = m_pScene->Lights.cbegin();
    return ((_it == m_pScene->Lights.cend()) ? *++_it : nullptr);
}

const shared_ptr<SceneObjectMaterial> SceneManager::GetFirstMaterial()
{
    if (!m_pScene) return nullptr;
    return (m_pScene->Materials.empty()? nullptr : m_pScene->Materials.front());
}

const shared_ptr<SceneObjectMaterial> SceneManager::GetNextMaterial()
{
    static thread_local decltype(m_pScene->Materials.cbegin()) _it = m_pScene->Materials.cbegin();
    return ((_it == m_pScene->Materials.cend()) ? *++_it : nullptr);
}

const shared_ptr<SceneObjectGeometry> SceneManager::GetFirstGeometry()
{
    if (!m_pScene) return nullptr;
    return (m_pScene->Geometries.empty()? nullptr : m_pScene->Geometries.front());
}

const shared_ptr<SceneObjectGeometry> SceneManager::GetNextGeometry()
{
    static thread_local decltype(m_pScene->Geometries.cbegin()) _it = m_pScene->Geometries.cbegin();
    return ((_it == m_pScene->Geometries.cend()) ? *++_it : nullptr);
}


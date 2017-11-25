#include "SceneParser.hpp"

using namespace My;
using namespace std;

const shared_ptr<SceneObjectCamera> Scene::GetCamera(string key) const
{
    auto i = Cameras.find(key);
    if (i == Cameras.end())
        return nullptr;
    else
        return i->second;
}

const shared_ptr<SceneObjectLight> Scene::GetLight(string key) const
{
    auto i = Lights.find(key);
    if (i == Lights.end())
        return nullptr;
    else
        return i->second;
}

const shared_ptr<SceneObjectMaterial> Scene::GetMaterial(string key) const
{
    auto i = Materials.find(key);
    if (i == Materials.end())
        return nullptr;
    else
        return i->second;
}

const shared_ptr<SceneObjectGeometry> Scene::GetGeometry(string key) const
{
    auto i = Geometries.find(key);
    if (i == Geometries.end())
        return nullptr;
    else
        return i->second;
}

const shared_ptr<SceneObjectGeometry> Scene::GetFirstGeometry() const
{
    return (Geometries.empty()? nullptr : Geometries.cbegin()->second);
}

const shared_ptr<SceneObjectGeometry> Scene::GetNextGeometry() const
{
    static thread_local auto _it = Geometries.cbegin();
    if (_it == Geometries.cend()) return nullptr;
    return ((++_it == Geometries.cend()) ? nullptr : _it->second);
}

const shared_ptr<SceneObjectMaterial> Scene::GetFirstMaterial() const
{
    return (Materials.empty()? nullptr : Materials.cbegin()->second);
}

const shared_ptr<SceneObjectMaterial> Scene::GetNextMaterial() const
{
    static thread_local auto _it = Materials.cbegin();
    if (_it == Materials.cend()) return nullptr;
    return ((++_it == Materials.cend()) ? nullptr : _it->second);
}

const shared_ptr<SceneObjectLight> Scene::GetFirstLight() const
{
    return (Lights.empty()? nullptr : Lights.cbegin()->second);
}

const shared_ptr<SceneObjectLight> Scene::GetNextLight() const
{
    static thread_local auto _it = Lights.cbegin();
    if (_it == Lights.cend()) return nullptr;
    return ((++_it == Lights.cend()) ? nullptr : _it->second);
}

const shared_ptr<SceneObjectCamera> Scene::GetFirstCamera() const
{
    return (Cameras.empty()? nullptr : Cameras.cbegin()->second);
}

const shared_ptr<SceneObjectCamera> Scene::GetNextCamera() const
{
    static thread_local auto _it = Cameras.cbegin();
    if (_it == Cameras.cend()) return nullptr;
    return ((++_it == Cameras.cend()) ? nullptr : _it->second);
}


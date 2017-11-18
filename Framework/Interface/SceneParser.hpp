#pragma once
#include <memory>
#include <string>
#include <forward_list>
#include "Interface.hpp"
#include "SceneNode.hpp"
#include "SceneObject.hpp"

namespace My {
    class Scene {
    public:
        Scene(const char* scene_name) :
            SceneGraph(new BaseSceneNode(scene_name))
        {
        }
        ~Scene() = default;

    public:
        std::unique_ptr<BaseSceneNode> SceneGraph;
        std::forward_list<std::shared_ptr<SceneObjectCamera>>      Cameras;
        std::forward_list<std::shared_ptr<SceneObjectLight>>       Lights;
        std::forward_list<std::shared_ptr<SceneObjectMaterial>>    Materials;
        std::forward_list<std::shared_ptr<SceneObjectGeometry>>    Geometries;
    };

    Interface SceneParser
    {
    public:
        virtual std::unique_ptr<Scene> Parse(const std::string& buf) = 0;
    };
}


#pragma once
#include <memory>
#include <string>
#include <unordered_map>
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
        std::unordered_map<std::string, std::shared_ptr<SceneObjectCamera>>      Cameras;
        std::unordered_map<std::string, std::shared_ptr<SceneObjectLight>>       Lights;
        std::unordered_map<std::string, std::shared_ptr<SceneObjectMaterial>>    Materials;
        std::unordered_map<std::string, std::shared_ptr<SceneObjectGeometry>>    Geometries;

        const std::shared_ptr<SceneObjectCamera> GetCamera(std::string key) const;
        const std::shared_ptr<SceneObjectCamera> GetFirstCamera() const;
        const std::shared_ptr<SceneObjectCamera> GetNextCamera() const;

        const std::shared_ptr<SceneObjectLight> GetLight(std::string key) const;
        const std::shared_ptr<SceneObjectLight> GetFirstLight() const;
        const std::shared_ptr<SceneObjectLight> GetNextLight() const;

        const std::shared_ptr<SceneObjectMaterial> GetMaterial(std::string key) const;
        const std::shared_ptr<SceneObjectMaterial> GetFirstMaterial() const;
        const std::shared_ptr<SceneObjectMaterial> GetNextMaterial() const;

        const std::shared_ptr<SceneObjectGeometry> GetGeometry(std::string key) const;
        const std::shared_ptr<SceneObjectGeometry> GetFirstGeometry() const;
        const std::shared_ptr<SceneObjectGeometry> GetNextGeometry() const;

    };

    Interface SceneParser
    {
    public:
        virtual std::unique_ptr<Scene> Parse(const std::string& buf) = 0;
    };
}


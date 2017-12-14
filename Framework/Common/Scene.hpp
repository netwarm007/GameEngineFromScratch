#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "SceneObject.hpp"
#include "SceneNode.hpp"

namespace My {
    class Scene {
    public:
        std::shared_ptr<BaseSceneNode> SceneGraph;

        std::unordered_multimap<std::string, std::shared_ptr<SceneCameraNode>>       CameraNodes;
        std::unordered_multimap<std::string, std::shared_ptr<SceneLightNode>>        LightNodes;
        std::unordered_multimap<std::string, std::shared_ptr<SceneGeometryNode>>     GeometryNodes;
        
        std::unordered_map<std::string, std::shared_ptr<SceneObjectCamera>>     Cameras;
        std::unordered_map<std::string, std::shared_ptr<SceneObjectLight>>      Lights;
        std::unordered_map<std::string, std::shared_ptr<SceneObjectMaterial>>   Materials;
        std::unordered_map<std::string, std::shared_ptr<SceneObjectGeometry>>   Geometries;

    public:
        Scene(const std::string& scene_name) :
            SceneGraph(new BaseSceneNode(scene_name))
        {
        }
        ~Scene() = default;

        const std::shared_ptr<SceneObjectCamera> GetCamera(const std::string& key) const;
        const std::shared_ptr<SceneCameraNode> GetFirstCameraNode() const;
        const std::shared_ptr<SceneCameraNode> GetNextCameraNode() const;

        const std::shared_ptr<SceneObjectLight> GetLight(const std::string& key) const;
        const std::shared_ptr<SceneLightNode> GetFirstLightNode() const;
        const std::shared_ptr<SceneLightNode> GetNextLightNode() const;

        const std::shared_ptr<SceneObjectGeometry> GetGeometry(const std::string& key) const;
        const std::shared_ptr<SceneGeometryNode> GetFirstGeometryNode() const;
        const std::shared_ptr<SceneGeometryNode> GetNextGeometryNode() const;

        const std::shared_ptr<SceneObjectMaterial> GetMaterial(const std::string& key) const;
        const std::shared_ptr<SceneObjectMaterial> GetFirstMaterial() const;
        const std::shared_ptr<SceneObjectMaterial> GetNextMaterial() const;

        void LoadResource(void);
    };
}


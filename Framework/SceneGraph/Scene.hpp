#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "SceneNode.hpp"
#include "SceneObject.hpp"

namespace My {
class Scene {
   private:
    std::shared_ptr<SceneObjectMaterial> m_pDefaultMaterial;

   public:
    std::shared_ptr<BaseSceneNode> SceneGraph;

    std::unordered_map<std::string, std::shared_ptr<SceneObjectCamera>> Cameras;
    std::unordered_map<std::string, std::shared_ptr<SceneObjectLight>> Lights;
    std::unordered_map<std::string, std::shared_ptr<SceneObjectMaterial>>
        Materials;
    std::unordered_map<std::string, std::shared_ptr<SceneObjectGeometry>>
        Geometries;

    std::unordered_multimap<std::string, std::weak_ptr<SceneCameraNode>>
        CameraNodes;
    std::unordered_multimap<std::string, std::weak_ptr<SceneLightNode>>
        LightNodes;
    std::unordered_multimap<std::string, std::weak_ptr<SceneGeometryNode>>
        GeometryNodes;
    std::unordered_map<std::string, std::weak_ptr<SceneBoneNode>> BoneNodes;

    std::vector<std::weak_ptr<BaseSceneNode>> AnimatableNodes;

    std::unordered_map<std::string, std::weak_ptr<SceneGeometryNode>>
        LUT_Name_GeometryNode;

    std::shared_ptr<SceneObjectSkyBox> SkyBox;

    std::shared_ptr<SceneObjectTerrain> Terrain;

   public:
    Scene() {
        m_pDefaultMaterial = std::make_shared<SceneObjectMaterial>("default");
        SkyBox = std::make_shared<SceneObjectSkyBox>();
        SkyBox->SetName("Textures/hdr/spruit_sunrise", "dds");
        Terrain = std::make_shared<SceneObjectTerrain>();
        Terrain->SetName("Textures/terrain/area_1", "png");
    }

    explicit Scene(const std::string& scene_name) : Scene() {
        SceneGraph = std::make_shared<BaseSceneNode>(scene_name);
    }

    ~Scene() { std::cerr << "Scene destroyed" << std::endl; }

    [[nodiscard]] std::shared_ptr<SceneObjectCamera> GetCamera(
        const std::string& key) const;
    [[nodiscard]] std::shared_ptr<SceneCameraNode> GetFirstCameraNode() const;

    [[nodiscard]] std::shared_ptr<SceneObjectLight> GetLight(
        const std::string& key) const;
    [[nodiscard]] std::shared_ptr<SceneLightNode> GetFirstLightNode() const;

    [[nodiscard]] std::shared_ptr<SceneObjectGeometry> GetGeometry(
        const std::string& key) const;
    [[nodiscard]] std::shared_ptr<SceneGeometryNode> GetFirstGeometryNode()
        const;

    [[nodiscard]] std::shared_ptr<SceneObjectMaterial> GetMaterial(
        const std::string& key) const;
    [[nodiscard]] std::shared_ptr<SceneObjectMaterial> GetFirstMaterial() const;
};
}  // namespace My

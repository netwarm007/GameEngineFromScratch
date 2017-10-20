#pragma once
#include <vector>
#include <string>
#include "SceneObject.hpp"

namespace My {
    struct BaseSceneNode {
        std::string name;
        std::vector<BaseSceneNode> children;
    };

    template <typename T>
    struct SceneNode : public BaseSceneNode {
        T* t;
    };

    typedef BaseSceneNode SceneEmptyNode;
    typedef SceneNode<SceneObjectMesh> SceneGeometryNode;
    typedef SceneNode<SceneObjectMaterial> SceneMaterialNode;
    typedef SceneNode<SceneObjectLight> SceneLightNode;
    typedef SceneNode<SceneObjectCamera> SceneCameraNode;
}


#pragma once
#include <memory>
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
        std::shared_ptr<T> pSceneObject;
    };

    typedef BaseSceneNode SceneEmptyNode;
    typedef SceneNode<SceneObjectGeometry> SceneGeometryNode;
    typedef SceneNode<SceneObjectLight> SceneLightNode;
    typedef SceneNode<SceneObjectCamera> SceneCameraNode;
}


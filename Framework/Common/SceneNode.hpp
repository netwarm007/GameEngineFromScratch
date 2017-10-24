#pragma once
#include <list>
#include <memory>
#include <string>
#include <vector>
#include "SceneObject.hpp"

namespace My {
    struct BaseSceneNode {
        std::string name;
        std::list<std::unique_ptr<BaseSceneNode>> children;

        std::vector<std::unique_ptr<SceneObjectTransform>> transforms;
    };

    template <typename T>
    struct SceneNode : public BaseSceneNode {
        std::shared_ptr<T> pSceneObject;
    };

    typedef BaseSceneNode SceneEmptyNode;
    class SceneGeometryNode : public SceneNode<SceneObjectGeometry> 
    {
        protected:
            bool        m_bVisible;
            bool        m_bShadow;
            bool        m_bMotionBlur;

        public:
    };

    class SceneLightNode : public SceneNode<SceneObjectLight> 
    {
        protected:
            Vector3f m_Target;

        public:
    };

    class SceneCameraNode : public SceneNode<SceneObjectCamera>
    {
        protected:
            Vector3f m_Target;

        public:
    };
}


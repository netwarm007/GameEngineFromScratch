#pragma once
#include <list>
#include <memory>
#include <string>
#include <vector>
#include "SceneObject.hpp"

namespace My {
    class BaseSceneNode {
        protected:
            std::string m_strName;
            std::list<std::unique_ptr<BaseSceneNode>> m_Children;
            std::vector<std::unique_ptr<SceneObjectTransform>> m_Transforms;

        public:
            BaseSceneNode() {};
            BaseSceneNode(const char* name) { m_strName = name; };
            BaseSceneNode(const std::string& name) { m_strName = name; };
            BaseSceneNode(const std::string&& name) { m_strName = std::move(name); };

            void AppendChild(std::unique_ptr<BaseSceneNode>&& sub_node)
            {
                m_Children.push_back(std::move(sub_node));
            }

            void AppendChild(std::unique_ptr<SceneObjectTransform>&& transform)
            {
                m_Transforms.push_back(std::move(transform));
            }
    };

    template <typename T>
    class SceneNode : public BaseSceneNode {
        protected:
            std::shared_ptr<T> m_pSceneObject;

        public:
            using BaseSceneNode::BaseSceneNode;
    };

    typedef BaseSceneNode SceneEmptyNode;
    class SceneGeometryNode : public SceneNode<SceneObjectGeometry> 
    {
        protected:
            bool        m_bVisible;
            bool        m_bShadow;
            bool        m_bMotionBlur;

        public:
            using SceneNode::SceneNode;

            void SetVisibility(bool visible) { m_bVisible = visible; };
            const bool Visible() { return m_bVisible; };
            void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
            const bool CastShadow() { return m_bShadow; };
            void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; };
            const bool MotionBlur() { return m_bMotionBlur; };
    };

    class SceneLightNode : public SceneNode<SceneObjectLight> 
    {
        protected:
            Vector3f m_Target;

        public:
            using SceneNode::SceneNode;

            void SetTarget(Vector3f& target) { m_Target = target; };
            const Vector3f& GetTarget() { return m_Target; };
    };

    class SceneCameraNode : public SceneNode<SceneObjectCamera>
    {
        protected:
            Vector3f m_Target;

        public:
            using SceneNode::SceneNode;

            void SetTarget(Vector3f& target) { m_Target = target; };
            const Vector3f& GetTarget() { return m_Target; };
    };
}


#pragma once
#include <iostream>
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
            std::list<std::unique_ptr<SceneObjectTransform>> m_Transforms;

        protected:
            virtual void dump(std::ostream& out) const {};

        public:
            BaseSceneNode() {};
            BaseSceneNode(const char* name) { m_strName = name; };
            BaseSceneNode(const std::string& name) { m_strName = name; };
            BaseSceneNode(const std::string&& name) { m_strName = std::move(name); };
			virtual ~BaseSceneNode() {};

            void AppendChild(std::unique_ptr<BaseSceneNode>&& sub_node)
            {
                m_Children.push_back(std::move(sub_node));
            }

            void AppendChild(std::unique_ptr<SceneObjectTransform>&& transform)
            {
                m_Transforms.push_back(std::move(transform));
            }

        friend std::ostream& operator<<(std::ostream& out, const BaseSceneNode& node)
        {
            static thread_local int32_t indent = 0;
            indent++;

            out << std::string(indent, ' ') << "Scene Node" << std::endl;
            out << std::string(indent, ' ') << "----------" << std::endl;
            out << std::string(indent, ' ') << "Name: " << node.m_strName << std::endl;
            node.dump(out);
            out << std::endl;

            for (const std::unique_ptr<BaseSceneNode>& sub_node : node.m_Children) {
                out << *sub_node << std::endl;
            }

            for (const std::unique_ptr<SceneObjectTransform>& sub_node : node.m_Transforms) {
                out << *sub_node << std::endl;
            }

            indent--;

            return out;
        }
    };

    template <typename T>
    class SceneNode : public BaseSceneNode {
        protected:
            std::shared_ptr<T> m_pSceneObject;

        protected:
            virtual void dump(std::ostream& out) const 
            { 
                if ( m_pSceneObject)
                    out << *m_pSceneObject << std::endl;
            };

        public:
            using BaseSceneNode::BaseSceneNode;
            SceneNode() = default;
            SceneNode(const std::shared_ptr<T>& object) { m_pSceneObject = object; };
            SceneNode(const std::shared_ptr<T>&& object) { m_pSceneObject = std::move(object); };

            void AddSceneObjectRef(const std::shared_ptr<T>& object) { m_pSceneObject = object; };
            void AddSceneObjectRef(const std::shared_ptr<T>&& object) { m_pSceneObject = std::move(object); };

    };

    typedef BaseSceneNode SceneEmptyNode;
    class SceneGeometryNode : public SceneNode<SceneObjectGeometry> 
    {
        protected:
            bool        m_bVisible;
            bool        m_bShadow;
            bool        m_bMotionBlur;
            std::vector<std::shared_ptr<SceneObjectMaterial>> m_Materials;

        protected:
            virtual void dump(std::ostream& out) const 
            { 
                SceneNode::dump(out);
                out << "Visible: " << m_bVisible << std::endl;
                out << "Shadow: " << m_bShadow << std::endl;
                out << "Motion Blur: " << m_bMotionBlur << std::endl;
                out << "Material(s): " << std::endl;
                for (auto material : m_Materials) {
                    out << *material << std::endl;
                }
            };

        public:
            using SceneNode::SceneNode;

            void SetVisibility(bool visible) { m_bVisible = visible; };
            const bool Visible() { return m_bVisible; };
            void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
            const bool CastShadow() { return m_bShadow; };
            void SetIfMotionBlur(bool motion_blur) { m_bMotionBlur = motion_blur; };
            const bool MotionBlur() { return m_bMotionBlur; };
            using SceneNode::AddSceneObjectRef;
            void AddSceneObjectRef(const std::shared_ptr<SceneObjectMaterial>& object) { m_Materials.push_back(object); };
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


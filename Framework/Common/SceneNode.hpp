#pragma once
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include "Tree.hpp"
#include "SceneObject.hpp"

namespace My {
    class BaseSceneNode : public TreeNode {
        protected:
            std::string m_strName;
            std::list<std::shared_ptr<SceneObjectTransform>> m_Transforms;
            Matrix4X4f m_RuntimeTransform;

        public:
            BaseSceneNode() { BuildIdentityMatrix(m_RuntimeTransform); };
            BaseSceneNode(const std::string& name) { m_strName = name; BuildIdentityMatrix(m_RuntimeTransform); };
			virtual ~BaseSceneNode() {};

            const std::string GetName() const { return m_strName; };

            void AppendTransform(std::shared_ptr<SceneObjectTransform>&& transform)
            {
                m_Transforms.push_back(std::move(transform));
            }

            void PrependTransform(std::shared_ptr<SceneObjectTransform>&& transform)
            {
                m_Transforms.push_front(std::move(transform));
            }

            const std::shared_ptr<Matrix4X4f> GetCalculatedTransform() const
            {
                std::shared_ptr<Matrix4X4f> result (new Matrix4X4f());
                BuildIdentityMatrix(*result);

                // TODO: cascading calculation
                for (auto trans : m_Transforms)
                {
                    *result = *result * static_cast<Matrix4X4f>(*trans);
                }

                // apply runtime transforms
                *result = *result * m_RuntimeTransform;

                return result;
            }

            void RotateBy(float rotation_angle_x, float rotation_angle_y, float rotation_angle_z)
            {
                Matrix4X4f rotate;
                MatrixRotationYawPitchRoll(rotate, rotation_angle_x, rotation_angle_y, rotation_angle_z);
                m_RuntimeTransform = m_RuntimeTransform * rotate;
            }

            void MoveBy(float distance_x, float distance_y, float distance_z)
            {
                Matrix4X4f translation;
                MatrixTranslation(translation, distance_x, distance_y, distance_z);
                m_RuntimeTransform = m_RuntimeTransform * translation;
            }

            void MoveBy(const Vector3f& distance)
            {
                MoveBy(distance.x, distance.y, distance.z);
            }

            virtual Matrix3X3f GetLocalAxis()
            {
                return {{{
                            {1.0f, 0.0f, 0.0f},
                            {0.0f, 1.0f, 0.0f},
                            {0.0f, 0.0f, 1.0f}
                       }}};
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

            for (auto sub_node : node.m_Children) {
                out << *sub_node << std::endl;
            }

            for (auto sub_node : node.m_Transforms) {
                out << *sub_node << std::endl;
            }

            indent--;

            return out;
        }
    };

    template <typename T>
    class SceneNode : public BaseSceneNode {
        protected:
            std::string m_keySceneObject;

        protected:
            virtual void dump(std::ostream& out) const 
            { 
                out << m_keySceneObject << std::endl;
            };

        public:
            using BaseSceneNode::BaseSceneNode;
            SceneNode() = default;

            void AddSceneObjectRef(const std::string& key) { m_keySceneObject = key; };

            const std::string& GetSceneObjectRef() { return m_keySceneObject; };
    };

    typedef BaseSceneNode SceneEmptyNode;

    class SceneGeometryNode : public SceneNode<SceneObjectGeometry> 
    {
        protected:
            bool        m_bVisible;
            bool        m_bShadow;
            bool        m_bMotionBlur;
            std::vector<std::string> m_Materials;
            void*       m_pRigidBody = nullptr;

        protected:
            virtual void dump(std::ostream& out) const 
            { 
                SceneNode::dump(out);
                out << "Visible: " << m_bVisible << std::endl;
                out << "Shadow: " << m_bShadow << std::endl;
                out << "Motion Blur: " << m_bMotionBlur << std::endl;
                out << "Material(s): " << std::endl;
                for (auto material : m_Materials) {
                    out << material << std::endl;
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
            void AddMaterialRef(const std::string& key) { m_Materials.push_back(key); };
            void AddMaterialRef(const std::string&& key) { m_Materials.push_back(std::move(key)); };
            std::string GetMaterialRef(const size_t index) 
            { 
                if (index < m_Materials.size())
                    return m_Materials[index]; 
                else
                    return std::string("default");
            };

            void LinkRigidBody(void* rigidBody)
            {
                m_pRigidBody = rigidBody;
            }

            void* UnlinkRigidBody()
            {
                void* rigidBody = m_pRigidBody;
                m_pRigidBody = nullptr;

                return rigidBody;
            }

            void* RigidBody() { return m_pRigidBody; }
    };

    class SceneLightNode : public SceneNode<SceneObjectLight> 
    {
        protected:
            bool        m_bShadow;

        public:
            using SceneNode::SceneNode;

            void SetIfCastShadow(bool shadow) { m_bShadow = shadow; };
            const bool CastShadow() { return m_bShadow; };
    };

    class SceneCameraNode : public SceneNode<SceneObjectCamera>
    {
        protected:
            Vector3f m_Target = {0.0f};

        public:
            using SceneNode::SceneNode;

            void SetTarget(Vector3f& target) { m_Target = target; };
            const Vector3f& GetTarget() { return m_Target; };
            Matrix3X3f GetLocalAxis()
            {
                Matrix3X3f result;
                auto pTransform = GetCalculatedTransform();
                Vector3f target = GetTarget();
                Vector3f camera_position = Vector3f(0.0f);
                TransformCoord(camera_position, *pTransform);
                Vector3f up (0.0f, 0.0f, 1.0f);
                Vector3f camera_z_axis = camera_position - target;
                Normalize(camera_z_axis);
                Vector3f camera_x_axis;
                Vector3f camera_y_axis;
                CrossProduct(camera_x_axis, camera_z_axis, up);
                CrossProduct(camera_y_axis, camera_x_axis, camera_z_axis);
                memcpy(result[0], camera_x_axis.data, sizeof(camera_x_axis));
                memcpy(result[1], camera_y_axis.data, sizeof(camera_y_axis));
                memcpy(result[2], camera_z_axis.data, sizeof(camera_z_axis));

                return result;
            }
    };
}


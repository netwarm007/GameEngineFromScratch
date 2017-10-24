#pragma once
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <assert.h>
#include "SceneObject.hpp"

namespace My {
    class SceneObjectTransform
    {
        protected:
            Matrix4X4f m_matrix;
            bool m_bSceneObjectOnly;

        public:
            SceneObjectTransform() { BuildIdentityMatrix(m_matrix); m_bSceneObjectOnly = false; };

            SceneObjectTransform(const Matrix4X4f& matrix, const bool object_only = false) { m_matrix = matrix; m_bSceneObjectOnly = object_only; };
    };

    class SceneObjectTranslation : public SceneObjectTransform
    {
        public:
            SceneObjectTranslation(const char axis, const float amount)  
            { 
                switch (axis) {
                    case 'x':
                        MatrixTranslation(m_matrix, amount, 0.0f, 0.0f);
                        break;
                    case 'y':
                        MatrixTranslation(m_matrix, 0.0f, amount, 0.0f);
                        break;
                    case 'z':
                        MatrixTranslation(m_matrix, 0.0f, 0.0f, amount);
                        break;
                    default:
                        assert(0);
                }
            }

            SceneObjectTranslation(const float x, const float y, const float z) 
            {
                MatrixTranslation(m_matrix, x, y, z);
            }
    };

    class SceneObjectRotation : public SceneObjectTransform
    {
        public:
            SceneObjectRotation(const char axis, const float theta)
            {
                switch (axis) {
                    case 'x':
                        MatrixRotationX(m_matrix, theta);
                        break;
                    case 'y':
                        MatrixRotationY(m_matrix, theta);
                        break;
                    case 'z':
                        MatrixRotationZ(m_matrix, theta);
                        break;
                    default:
                        assert(0);
                }
            }

            SceneObjectRotation(Vector3f& axis, const float theta)
            {
                Normalize(axis);
                MatrixRotationAxis(m_matrix, axis, theta);
            }

            SceneObjectRotation(const Quaternion quaternion)
            {
                MatrixRotationQuaternion(m_matrix, quaternion);
            }
    };

    class SceneObjectScale : public SceneObjectTransform
    {
        public:
    };

    struct BaseSceneNode {
        std::string name;
        std::list<BaseSceneNode> children;

        std::vector<SceneObjectTransform> transforms;
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


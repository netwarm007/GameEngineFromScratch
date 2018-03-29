#pragma once
#include "BaseSceneObject.hpp"
#include "geommath.hpp"
#include "Animatable.hpp"

namespace My {
    class SceneObjectTransform : public BaseSceneObject
    {
        protected:
            Matrix4X4f m_matrix;
            bool m_bSceneObjectOnly;

        public:
            SceneObjectTransform() : BaseSceneObject(SceneObjectType::kSceneObjectTypeTransform) 
            { BuildIdentityMatrix(m_matrix); m_bSceneObjectOnly = false; }

            SceneObjectTransform(const Matrix4X4f& matrix, const bool object_only = false) : SceneObjectTransform() 
            { m_matrix = matrix; m_bSceneObjectOnly = object_only; }

            operator Matrix4X4f() { return m_matrix; }
            operator const Matrix4X4f() const { return m_matrix; }

        friend std::ostream& operator<<(std::ostream& out, const SceneObjectTransform& obj);
    };

    class SceneObjectTranslation : public SceneObjectTransform, implements Animatable<float>
    {
        private:
            char m_Kind = 0;

        public:
            SceneObjectTranslation() { m_Type = SceneObjectType::kSceneObjectTypeTranslate; }
            SceneObjectTranslation(const char axis, const float amount, const bool object_only = false)  
                : SceneObjectTranslation()
            { 
                m_Kind = axis;

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

                m_bSceneObjectOnly = object_only;
            }

            SceneObjectTranslation(const float x, const float y, const float z, const bool object_only = false) 
                : SceneObjectTranslation()
            {
                m_Kind = 0;
                MatrixTranslation(m_matrix, x, y, z);
                m_bSceneObjectOnly = object_only;
            }

            void Update(const float amount) final
            {
                switch (m_Kind) {
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
    };

    class SceneObjectRotation : public SceneObjectTransform, implements Animatable<float>
    {
        private:
            char m_Kind = 0;

        public:
            SceneObjectRotation() { m_Type = SceneObjectType::kSceneObjectTypeRotate; }
            SceneObjectRotation(const char axis, const float theta, const bool object_only = false)
                : SceneObjectRotation()
            {
                m_Kind = axis;

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

                m_bSceneObjectOnly = object_only;
            }

            SceneObjectRotation(Vector3f axis, const float theta, const bool object_only = false)
                : SceneObjectRotation()
            {
                m_Kind = 0;

                Normalize(axis);
                MatrixRotationAxis(m_matrix, axis, theta);

                m_bSceneObjectOnly = object_only;
            }

            SceneObjectRotation(const Quaternion quaternion, const bool object_only = false)
                : SceneObjectRotation()
            {
                m_Kind = 0;

                MatrixRotationQuaternion(m_matrix, quaternion);

                m_bSceneObjectOnly = object_only;
            }

            void Update(const float theta) final
            {
                switch (m_Kind) {
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
    };

    class SceneObjectScale : public SceneObjectTransform, implements Animatable<float>
    {
        private:
            char m_Kind = 0;

        public:
            SceneObjectScale() { m_Type = SceneObjectType::kSceneObjectTypeScale; }
            SceneObjectScale(const char axis, const float amount)  
                : SceneObjectScale()
            { 
                m_Kind = axis;

                switch (axis) {
                    case 'x':
                        MatrixScale(m_matrix, amount, 0.0f, 0.0f);
                        break;
                    case 'y':
                        MatrixScale(m_matrix, 0.0f, amount, 0.0f);
                        break;
                    case 'z':
                        MatrixScale(m_matrix, 0.0f, 0.0f, amount);
                        break;
                    default:
                        assert(0);
                }
            }

            SceneObjectScale(const float x, const float y, const float z) 
                : SceneObjectScale()
            {
                m_Kind = 0;
                MatrixScale(m_matrix, x, y, z);
            }

            void Update(const float amount) final
            {
                switch (m_Kind) {
                    case 'x':
                        MatrixScale(m_matrix, amount, 0.0f, 0.0f);
                        break;
                    case 'y':
                        MatrixScale(m_matrix, 0.0f, amount, 0.0f);
                        break;
                    case 'z':
                        MatrixScale(m_matrix, 0.0f, 0.0f, amount);
                        break;
                    default:
                        assert(0);
                }
            }
    };


}
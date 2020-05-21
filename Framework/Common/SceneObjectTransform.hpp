#pragma once
#include "Animatable.hpp"
#include "BaseSceneObject.hpp"
#include "geommath.hpp"

namespace My {
class SceneObjectTransform : public BaseSceneObject,
                             _implements_ Animatable<float>,
                             Animatable<Vector3f>,
                             Animatable<Quaternion<float>>,
                             Animatable<Matrix4X4f> {
   protected:
    Matrix4X4f m_matrix;
    bool m_bSceneObjectOnly;

   public:
    SceneObjectTransform()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeTransform) {
        BuildIdentityMatrix(m_matrix);
        m_bSceneObjectOnly = false;
    }

    explicit SceneObjectTransform(const Matrix4X4f& matrix,
                                  const bool object_only = false)
        : SceneObjectTransform() {
        m_matrix = matrix;
        m_bSceneObjectOnly = object_only;
    }

    explicit operator Matrix4X4f() { return m_matrix; }
    explicit operator const Matrix4X4f() const { return m_matrix; }

    void Update(const float amount) override {
        // should not be used.
        assert(0);
    }

    void Update(const Vector3f amount) override {
        // should not be used.
        assert(0);
    }

    void Update(const Quaternion<float> amount) override {
        // should not be used.
        assert(0);
    }

    void Update(const Matrix4X4f amount) final { m_matrix = amount; }

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectTransform& obj);
};

class SceneObjectTranslation : public SceneObjectTransform {
   private:
    char m_Kind = 0;

   public:
    SceneObjectTranslation() {
        m_Type = SceneObjectType::kSceneObjectTypeTranslate;
    }
    SceneObjectTranslation(const char axis, const float amount,
                           const bool object_only = false)
        : SceneObjectTranslation() {
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

    SceneObjectTranslation(const float x, const float y, const float z,
                           const bool object_only = false)
        : SceneObjectTranslation() {
        m_Kind = 0;
        MatrixTranslation(m_matrix, x, y, z);
        m_bSceneObjectOnly = object_only;
    }

    void Update(const float amount) final {
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

    void Update(const Vector3f amount) final {
        MatrixTranslation(m_matrix, amount);
    }
};

class SceneObjectRotation : public SceneObjectTransform {
   private:
    char m_Kind = 0;

   public:
    SceneObjectRotation() { m_Type = SceneObjectType::kSceneObjectTypeRotate; }
    SceneObjectRotation(const char axis, const float theta,
                        const bool object_only = false)
        : SceneObjectRotation() {
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

    SceneObjectRotation(Vector3f axis, const float theta,
                        const bool object_only = false)
        : SceneObjectRotation() {
        m_Kind = 0;

        Normalize(axis);
        MatrixRotationAxis(m_matrix, axis, theta);

        m_bSceneObjectOnly = object_only;
    }

    template <typename T>
    explicit SceneObjectRotation(const Quaternion<T> quaternion,
                                 const bool object_only = false)
        : SceneObjectRotation() {
        m_Kind = 0;

        MatrixRotationQuaternion(m_matrix, quaternion);

        m_bSceneObjectOnly = object_only;
    }

    void Update(const float theta) final {
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

    void Update(const Vector3f amount) final {
        MatrixRotationYawPitchRoll(m_matrix, amount[0], amount[1], amount[2]);
    }

    void Update(const Quaternion<float> quaternion) final {
        MatrixRotationQuaternion(m_matrix, quaternion);
    }
};

class SceneObjectScale : public SceneObjectTransform {
   private:
    char m_Kind = 0;

   public:
    SceneObjectScale() { m_Type = SceneObjectType::kSceneObjectTypeScale; }
    SceneObjectScale(const char axis, const float amount,
                     const bool object_only = false)
        : SceneObjectScale() {
        m_Kind = axis;

        switch (axis) {
            case 'x':
                MatrixScale(m_matrix, amount, 1.0f, 1.0f);
                break;
            case 'y':
                MatrixScale(m_matrix, 1.0f, amount, 1.0f);
                break;
            case 'z':
                MatrixScale(m_matrix, 1.0f, 1.0f, amount);
                break;
            default:
                assert(0);
        }

        m_bSceneObjectOnly = object_only;
    }

    SceneObjectScale(const float x, const float y, const float z,
                     const bool object_only = false)
        : SceneObjectScale() {
        m_Kind = 0;
        MatrixScale(m_matrix, x, y, z);
        m_bSceneObjectOnly = object_only;
    }

    void Update(const float amount) final {
        switch (m_Kind) {
            case 'x':
                MatrixScale(m_matrix, amount, 1.0f, 1.0f);
                break;
            case 'y':
                MatrixScale(m_matrix, 1.0f, amount, 1.0f);
                break;
            case 'z':
                MatrixScale(m_matrix, 1.0f, 1.0f, amount);
                break;
            default:
                Update(Vector3f(amount));
        }
    }

    void Update(const Vector3f amount) final { MatrixScale(m_matrix, amount); }
};
}  // namespace My

#pragma once
#include "BaseSceneObject.hpp"
#include "geommath.hpp"

namespace My {
class SceneObjectCamera : public BaseSceneObject {
   protected:
    float m_fAspect;
    float m_fNearClipDistance{1.0f};
    float m_fFarClipDistance{100.0f};

   public:
    void SetColor(std::string& attrib, Vector4f& color){
        // TODO: extension
    };

    void SetParam(std::string& attrib, float param) {
        if (attrib == "near") {
            m_fNearClipDistance = param;
        } else if (attrib == "far") {
            m_fFarClipDistance = param;
        }
    };

    void SetTexture(std::string& attrib, std::string& textureName){
        // TODO: extension
    };

    [[nodiscard]] float GetNearClipDistance() const {
        return m_fNearClipDistance;
    };
    [[nodiscard]] float GetFarClipDistance() const {
        return m_fFarClipDistance;
    };

   protected:
    // can only be used as base class
    SceneObjectCamera()
        : BaseSceneObject(SceneObjectType::kSceneObjectTypeCamera),
          m_fAspect(16.0f / 9.0f){};

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectCamera& obj);
};

class SceneObjectOrthogonalCamera : public SceneObjectCamera {
   public:
    using SceneObjectCamera::SceneObjectCamera;

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectOrthogonalCamera& obj);
};

class SceneObjectPerspectiveCamera : public SceneObjectCamera {
   protected:
    float m_fFov;

   public:
    void SetParam(std::string& attrib, float param) {
        // TODO: handle fovx, fovy
        if (attrib == "fov") {
            m_fFov = param;
        }
        SceneObjectCamera::SetParam(attrib, param);
    };

   public:
    explicit SceneObjectPerspectiveCamera(float fov = PI / 2.0) : m_fFov(fov){};
    [[nodiscard]] float GetFov() const { return m_fFov; };

    friend std::ostream& operator<<(std::ostream& out,
                                    const SceneObjectPerspectiveCamera& obj);
};

}  // namespace My
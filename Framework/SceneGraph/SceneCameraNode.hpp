#pragma once
#include "BaseSceneNode.hpp"

namespace My {
class SceneCameraNode : public SceneNode<SceneObjectCamera> {
   protected:
    Vector3f m_Target = {0.0f};

   public:
    using SceneNode::SceneNode;

    void SetTarget(const Vector3f& target) { 
        m_Target = target; 
    };
    const Vector3f& GetTarget() { return m_Target; };
    Matrix3X3f GetLocalAxis() override {
        Matrix3X3f result;
        auto pTransform = GetCalculatedTransform();
        Vector3f target = GetTarget();
        auto camera_position = Vector3f(0.0f);
        TransformCoord(camera_position, *pTransform);
        Vector3f camera_z_axis ({0.0f, 0.0f, 1.0f});
        Vector3f camera_y_axis = target - camera_position;
        Normalize(camera_y_axis);
        Vector3f camera_x_axis;
        CrossProduct(camera_x_axis, camera_y_axis, camera_z_axis);
        CrossProduct(camera_z_axis, camera_x_axis, camera_y_axis);
        result[0] = camera_x_axis;
        result[1] = camera_y_axis;
        result[2] = camera_z_axis;

        return result;
    }
};
}  // namespace My
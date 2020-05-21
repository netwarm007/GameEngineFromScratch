#pragma once
#include "BaseSceneNode.hpp"

namespace My {
class SceneCameraNode : public SceneNode<SceneObjectCamera> {
   protected:
    Vector3f m_Target = {0.0f};

   public:
    using SceneNode::SceneNode;

    void SetTarget(Vector3f& target) { m_Target = target; };
    const Vector3f& GetTarget() { return m_Target; };
    Matrix3X3f GetLocalAxis() override {
        Matrix3X3f result;
        auto pTransform = GetCalculatedTransform();
        Vector3f target = GetTarget();
        auto camera_position = Vector3f(0.0f);
        TransformCoord(camera_position, *pTransform);
        Vector3f up({0.0f, 0.0f, 1.0f});
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
}  // namespace My
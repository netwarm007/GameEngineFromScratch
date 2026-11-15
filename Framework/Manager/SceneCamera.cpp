#include "SceneCamera.hpp"

#include <cmath>
#include <iostream>

#include "Scene.hpp"

using namespace My;

bool My::PrepareSceneCamera(const Scene& scene, uint32_t canvasWidth,
                            uint32_t canvasHeight, bool openGLProjection,
                            CameraFrame& camera) {
    if (!canvasWidth || !canvasHeight) {
        std::cerr << "[SceneCamera] Canvas dimensions must be positive\n";
        return false;
    }

    Vector3f position = {0.0f, -5.0f, 0.0f};
    Vector3f target = {0.0f, 0.0f, 0.0f};
    const Vector3f up = {0.0f, 0.0f, 1.0f};
    float fieldOfView = PI / 3.0f;
    float nearClipDistance = 10.0f;
    float farClipDistance = 100.0f;

    const auto node = scene.GetFirstCameraNode();
    if (node) {
        const auto perspective =
            std::dynamic_pointer_cast<SceneObjectPerspectiveCamera>(
                scene.GetCamera(node->GetSceneObjectRef()));
        if (!perspective) {
            std::cerr
                << "[SceneCamera] Camera node must reference a perspective "
                   "camera: "
                << node->GetSceneObjectRef() << '\n';
            return false;
        }
        const auto transform = *node->GetCalculatedTransform();
        position = {transform[3][0], transform[3][1], transform[3][2]};
        target = node->GetTarget();
        fieldOfView = perspective->GetFov();
        nearClipDistance = perspective->GetNearClipDistance();
        farClipDistance = perspective->GetFarClipDistance();
    }

    if (!std::isfinite(fieldOfView) || fieldOfView <= 0.0f ||
        fieldOfView >= PI || !std::isfinite(nearClipDistance) ||
        !std::isfinite(farClipDistance) || nearClipDistance <= 0.0f ||
        farClipDistance <= nearClipDistance) {
        std::cerr << "[SceneCamera] Invalid perspective camera parameters\n";
        return false;
    }

    CameraFrame result;
    BuildViewRHMatrix(result.viewMatrix, position, target, up);
    result.position = {position[0], position[1], position[2], 0.0f};
    const float aspect = float(canvasWidth) / float(canvasHeight);
    if (openGLProjection) {
        BuildOpenglPerspectiveFovRHMatrix(result.projectionMatrix, fieldOfView,
                                          aspect, nearClipDistance,
                                          farClipDistance);
    } else {
        BuildPerspectiveFovRHMatrix(result.projectionMatrix, fieldOfView,
                                    aspect, nearClipDistance, farClipDistance);
    }
    camera = result;
    return true;
}

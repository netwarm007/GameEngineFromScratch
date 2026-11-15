#pragma once

#include <cstdint>

#include "geommath.hpp"

namespace My {
class Scene;

struct CameraFrame {
    Matrix4X4f viewMatrix;
    Matrix4X4f projectionMatrix;
    Vector4f position;
};

bool PrepareSceneCamera(const Scene& scene, uint32_t canvasWidth,
                        uint32_t canvasHeight, bool openGLProjection,
                        CameraFrame& camera);
}  // namespace My

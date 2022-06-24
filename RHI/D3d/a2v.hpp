#pragma once
#include "geommath.hpp"

namespace My {
struct Vertex {
    Vector3f pos;
    Vector3f color;
    Vector2f texCoord;
};

struct UniformBufferObject {
    Matrix4X4f model;
    Matrix4X4f view;
    Matrix4X4f proj;
};
}  // namespace My
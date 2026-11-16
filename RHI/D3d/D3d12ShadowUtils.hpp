#pragma once

#include "PointShadowMatrices.hpp"

namespace My {
inline void BuildD3d12PointShadowMatrices(Matrix4X4f (&matrices)[6],
                                          const Vector3f& position,
                                          const Matrix4X4f& projection) {
    BuildPointShadowMatrices(matrices, position, projection, true);
}
}

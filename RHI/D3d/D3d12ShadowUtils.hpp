#pragma once

#include "geommath.hpp"

namespace My {
inline void BuildD3d12PointShadowMatrices(Matrix4X4f (&matrices)[6],
                                          const Vector3f& position,
                                          const Matrix4X4f& projection) {
    const Vector3f directions[] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                   {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const Vector3f up[] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                           {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    for (size_t i = 0; i < 6; ++i) {
        Matrix4X4f view;
        BuildViewRHMatrix(view, position, position + directions[i], up[i]);
        MatrixMultiply(matrices[i], view, projection);
        // D3D viewport Y points down; cube-map face coordinates use the opposite orientation.
        for (size_t row = 0; row < 4; ++row) matrices[i][row][1] *= -1.0f;
    }
}
}

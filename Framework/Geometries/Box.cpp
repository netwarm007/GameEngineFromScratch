#include "Box.hpp"

using namespace My;

void Box::GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                  Vector3f& aabbMax) const {
    TransformAabb(m_vHalfExtents, m_fMargin, trans, aabbMin, aabbMax);
}

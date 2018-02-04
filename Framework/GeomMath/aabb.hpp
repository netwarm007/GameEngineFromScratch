#pragma once
#include "geommath.hpp"

namespace My {
    inline void TransformAabb(const Vector3f& halfExtents, float margin, const Matrix4X4f& trans, 
                               Vector3f& aabbMinOut, Vector3f& aabbMaxOut)
    {
        Vector3f halfExtentsWithMargin = halfExtents + Vector3f(margin,margin,margin);
        Vector3f  center;
        Vector3f  extent;
        GetOrigin(center, trans);
        DotProduct3(extent, trans);
        aabbMinOut = center - extent;
        aabbMaxOut = center + extent;
    }
}
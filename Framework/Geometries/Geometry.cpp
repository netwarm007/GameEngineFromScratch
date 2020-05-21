#include "Geometry.hpp"

using namespace My;

void Geometry::CalculateTemporalAabb(const Matrix4X4f& curTrans,
                                     const Vector3f& linvel,
                                     const Vector3f& angvel, float timeStep,
                                     Vector3f& temporalAabbMin,
                                     Vector3f& temporalAabbMax) const {
    // start with static aabb
    GetAabb(curTrans, temporalAabbMin, temporalAabbMax);

    float temporalAabbMaxx = temporalAabbMax[0];
    float temporalAabbMaxy = temporalAabbMax[1];
    float temporalAabbMaxz = temporalAabbMax[2];
    float temporalAabbMinx = temporalAabbMin[0];
    float temporalAabbMiny = temporalAabbMin[1];
    float temporalAabbMinz = temporalAabbMin[2];

    // add linear motion
    Vector3f linMotion = linvel * timeStep;
    ///@todo: simd would have a vector max/min operation, instead of per-element
    /// access
    if (linMotion[0] > 0.0f)
        temporalAabbMaxx += linMotion[0];
    else
        temporalAabbMinx += linMotion[0];
    if (linMotion[1] > 0.0f)
        temporalAabbMaxy += linMotion[1];
    else
        temporalAabbMiny += linMotion[1];
    if (linMotion[2] > 0.0f)
        temporalAabbMaxz += linMotion[2];
    else
        temporalAabbMinz += linMotion[2];

    // add conservative angular motion
    float angularMotion = Length(angvel) * GetAngularMotionDisc() * timeStep;
    Vector3f angularMotion3d({angularMotion, angularMotion, angularMotion});
    temporalAabbMin =
        Vector3f({temporalAabbMinx, temporalAabbMiny, temporalAabbMinz});
    temporalAabbMax =
        Vector3f({temporalAabbMaxx, temporalAabbMaxy, temporalAabbMaxz});

    temporalAabbMin = temporalAabbMin - angularMotion3d;
    temporalAabbMax = temporalAabbMax + angularMotion3d;
}

void Geometry::GetBoundingSphere(Vector3f& center, float& radius) const {
    Matrix4X4f tran;
    BuildIdentityMatrix(tran);
    Vector3f aabbMin, aabbMax;

    GetAabb(tran, aabbMin, aabbMax);

    radius = Length(aabbMax - aabbMin) * 0.5f;
    center = (aabbMin + aabbMax) * 0.5f;
}

float Geometry::GetAngularMotionDisc() const {
    Vector3f center;
    float disc = 0.0f;
    GetBoundingSphere(center, disc);
    disc += Length(center);
    return disc;
}
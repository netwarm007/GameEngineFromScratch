#pragma once
#include <limits>

#include "aabb.hpp"
#include "portable.hpp"

namespace My {
ENUM(GeometryType){kBox,   kCapsule,    kCone,   kCylinder,
                   kPlane, kPolyhydron, kSphere, kTriangle};

class Geometry {
   public:
    explicit Geometry(GeometryType geometry_type)
        : m_kGeometryType(geometry_type){};
    Geometry() = delete;
    virtual ~Geometry() = default;

    // GetAabb returns the axis aligned bounding box in the coordinate frame of
    // the given transform trans.
    virtual void GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                         Vector3f& aabbMax) const = 0;

    virtual void GetBoundingSphere(Vector3f& center, float& radius) const;

    // GetAngularMotionDisc returns the maximum radius needed for Conservative
    // Advancement to handle
    // time-of-impact with rotations.
    [[nodiscard]] virtual float GetAngularMotionDisc() const;

    // CalculateTemporalAabb calculates the enclosing aabb for the moving object
    // over interval [0..timeStep) result is conservative
    void CalculateTemporalAabb(const Matrix4X4f& curTrans,
                               const Vector3f& linvel, const Vector3f& angvel,
                               float timeStep, Vector3f& temporalAabbMin,
                               Vector3f& temporalAabbMax) const;

    [[nodiscard]] GeometryType GetGeometryType() const {
        return m_kGeometryType;
    };

   protected:
    GeometryType m_kGeometryType;
    float m_fMargin = std::numeric_limits<float>::epsilon();
    Vector3f m_color;
};
}  // namespace My
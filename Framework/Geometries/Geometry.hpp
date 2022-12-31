#pragma once
#include "portable.hpp"
#include "Hitable.hpp"

namespace My {
ENUM(GeometryType){kBox,   kCapsule,    kCone,   kCylinder,
                   kPlane, kPolyhydron, kSphere, kTriangle};

template <class T>
class Geometry : _implements_ Hitable<T> {
   public:
    __device__ explicit Geometry(GeometryType geometry_type)
        : Hitable<T>(HitableType::kGeometry) {
        m_kGeometryType = geometry_type;
     }
    Geometry() = delete;

    [[nodiscard]] GeometryType GetGeometryType() const {
        return m_kGeometryType;
    }

   protected:
    std::ostream& dump(std::ostream& out) const override {
        return out;
    }

   protected:
    GeometryType m_kGeometryType;
    T m_fMargin = 0.0001;
};
}  // namespace My
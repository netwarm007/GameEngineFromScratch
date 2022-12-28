#pragma once
#include <limits>

#include "portable.hpp"
#include "Hitable.hpp"

namespace My {
ENUM(GeometryType){kBox,   kCapsule,    kCone,   kCylinder,
                   kPlane, kPolyhydron, kSphere, kTriangle};

template <class T, class MaterialPtr>
class Geometry : _implements_ Hitable<T> {
   public:
    __device__ explicit Geometry(GeometryType geometry_type)
        : m_kGeometryType(geometry_type){ Hitable<T>::type = HitableType::kGeometry; }
    Geometry() = delete;
    virtual ~Geometry() = default;

    [[nodiscard]] GeometryType GetGeometryType() const {
        return m_kGeometryType;
    }

    void SetMaterial(MaterialPtr m) {
        m_ptrMat = m;
    }

    auto GetMaterial() {
        return m_ptrMat;
    }

   protected:
    std::ostream& dump(std::ostream& out) const override {
        return out;
    }

   protected:
    GeometryType m_kGeometryType;
    T m_fMargin = std::numeric_limits<T>::epsilon();
    MaterialPtr m_ptrMat;
};
}  // namespace My
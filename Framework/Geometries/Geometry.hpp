#pragma once
#include <limits>

#include "aabb.hpp"
#include "portable.hpp"
#include "Intersectable.hpp"

class material;

namespace My {
ENUM(GeometryType){kBox,   kCapsule,    kCone,   kCylinder,
                   kPlane, kPolyhydron, kSphere, kTriangle};

template <class T>
class Geometry : _implements_ Intersectable<T> {
   public:
    explicit Geometry(GeometryType geometry_type)
        : m_kGeometryType(geometry_type){};
    Geometry() = delete;
    virtual ~Geometry() = default;

    [[nodiscard]] GeometryType GetGeometryType() const {
        return m_kGeometryType;
    }

    void SetMaterial(std::shared_ptr<material> m) {
        m_ptrMat = m;
    }

    auto GetMaterial() {
        return m_ptrMat;
    }

   protected:
    GeometryType m_kGeometryType;
    T m_fMargin = std::numeric_limits<T>::epsilon();
    std::shared_ptr<material> m_ptrMat;
};
}  // namespace My
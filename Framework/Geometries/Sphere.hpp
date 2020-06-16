#pragma once
#include "Geometry.hpp"
#include "Intersectable.hpp"

namespace My {
class Sphere : public Geometry, _implements_ Intersectable<Sphere> {
   public:
    Sphere() = delete;
    explicit Sphere(const float radius)
        : Geometry(GeometryType::kSphere), m_fRadius(radius){};

    explicit Sphere(const float radius, const Vector3f center)
        : Geometry(GeometryType::kSphere), m_fRadius(radius), m_center(center) {};

    void GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                 Vector3f& aabbMax) const final;

    [[nodiscard]] float GetRadius() const { return m_fRadius; };
    [[nodiscard]] Vector3f GetCenter() const { return m_center; };

    bool Intersect(const Ray &r, Hit &h, float tmin) const override;

   protected:
    float m_fRadius;
    Vector3f m_center;
};
}  // namespace My
#pragma once
#include "Shape.hpp"

namespace My {
    class Sphere : public Shape
    {
    public:
        Sphere() : Shape(ShapeType::kSphere) {};

        virtual void GetAabb(const Matrix4X4f& trans, 
                                Vector3f& aabbMin, 
                                Vector3f& aabbMax) const;

        float GetRadius() const { return m_fRadius; };

    protected:
        float m_fRadius;
    };
}
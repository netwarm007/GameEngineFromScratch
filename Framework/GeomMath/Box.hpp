#pragma once
#include "Shape.hpp"

namespace My {
    class Box : public Shape
    {
    public:
        Box() : Shape(ShapeType::kBox) {};
        Box(Vector3f dimension) : Shape(ShapeType::kBox), m_vDimension(dimension) {};

        virtual void GetAabb(const Matrix4X4f& trans, 
                                Vector3f& aabbMin, 
                                Vector3f& aabbMax) const;

        Vector3f GetDimension() const { return m_vDimension; };
        Vector3f GetDimensionWithMargin() const { return m_vDimension + m_fMargin; };

    protected:
        Vector3f m_vDimension;
    };
}
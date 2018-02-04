#pragma once
#include "Geometry.hpp"

namespace My {
    class Box : public Geometry
    {
    public:
        Box() : Geometry(GeometryType::kBox) {};
        Box(Vector3f dimension) : Geometry(GeometryType::kBox), m_vDimension(dimension) {};

        virtual void GetAabb(const Matrix4X4f& trans, 
                                Vector3f& aabbMin, 
                                Vector3f& aabbMax) const;

        Vector3f GetDimension() const { return m_vDimension; };
        Vector3f GetDimensionWithMargin() const { return m_vDimension + m_fMargin; };

    protected:
        Vector3f m_vDimension;
    };
}
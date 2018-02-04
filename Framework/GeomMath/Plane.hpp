#pragma once
#include "Shape.hpp"

namespace My {
    class Plane : public Shape
    {
    public:
        Plane() : Shape(ShapeType::kPlane) {};
        Plane(Vector3f normal, float intercept) : Shape(ShapeType::kPlane), m_vNormal(normal), m_fIntercept(intercept) {};

        virtual void GetAabb(const Matrix4X4f& trans, 
                                Vector3f& aabbMin, 
                                Vector3f& aabbMax) const;

        Vector3f GetNormal() const { return m_vNormal; };
        float    GetIntercept() const { return m_fIntercept; };

    protected:
        Vector3f m_vNormal;
        float    m_fIntercept;
    };
}
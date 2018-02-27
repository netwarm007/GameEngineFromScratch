#pragma once
#include "Geometry.hpp"

namespace My {
    class Plane : public Geometry 
    {
    public:
        Plane() = delete;
        Plane(Vector3f normal, float intercept) : Geometry(GeometryType::kPlane), m_vNormal(normal), m_fIntercept(intercept) {};

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
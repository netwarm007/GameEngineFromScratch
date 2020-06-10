#include "Sphere.hpp"

using namespace My;

void Sphere::GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                     Vector3f& aabbMax) const {
    Vector3f center;
    GetOrigin(center, trans);
    center += m_center;
    Vector3f extent({m_fMargin, m_fMargin, m_fMargin});
    aabbMin = center - extent;
    aabbMax = center + extent;
}

bool Sphere::Intersect(const Ray& r, Hit& h, float tmin) const {
    bool result = false;

    // Ray: R(t) = O + V dot t
    // Sphere: || X - C || = r
    // Intersect equation: at^2  + bt + c = 0; a = V dot V; b = 2V dot (O - C);
    // C = ||O - C||^2 - r^2 Intersect condition: b^2 - 4ac > 0
    Vector3f V = r.getDirection();
    Vector3f O = r.getOrigin();
    Vector3f tmp = O - m_center;
    float dist = Length(tmp);

    float b = 2 * V.Dot3(tmp);
    float c = dist * dist - m_fRadius * m_fRadius;
    float disc = b * b - 4 * c;

    float t = INFINITY;

    if (disc > 0) {
        float sroot = sqrt(disc);

        float t1 = (-b - sroot) * 0.5;
        float t2 = (-b + sroot) * 0.5;

        if (t1 >= tmin)
            t = t1;
        else if (t2 >= tmin)
            t = t2;

        if (t < h.getT()) {
            h.set(t, m_color);
        }

        result = true;

    } else
        result = false;

    return result;
}
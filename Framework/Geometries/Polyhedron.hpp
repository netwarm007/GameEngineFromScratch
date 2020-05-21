#pragma once
#include "Geometry.hpp"

namespace My {
struct Polyhedron : public Geometry {
    FaceSet Faces;

    Polyhedron() : Geometry(GeometryType::kPolyhydron) {}

    // GetAabb returns the axis aligned bounding box in the coordinate frame of
    // the given transform trans.
    void GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                 Vector3f& aabbMax) const final;

    void AddFace(PointList vertices, const PointPtr& inner_point);

    void AddTetrahedron(const PointList& vertices);
};
}  // namespace My
#pragma once
#include "Geometry.hpp"

namespace My {
template <typename T>
struct Polyhedron : public Geometry {
    FaceSet<T> Faces;

    Polyhedron() : Geometry(GeometryType::kPolyhydron) {}

    // GetAabb returns the axis aligned bounding box in the coordinate frame of
    // the given transform trans.
    void GetAabb(const Matrix4X4<T>& trans, Vector3<T>& aabbMin,
                 Vector3<T>& aabbMax) const final {
        aabbMin = Vector3f((std::numeric_limits<T>::max)()); // Windows: Work around of warning C4003: not enough arguments for function-like macro invocation 'max'
        aabbMax = Vector3f(std::numeric_limits<T>::lowest());

        for (const auto& pFace : Faces) {
            for (const auto& pEdge : pFace->Edges) {
                auto pVertex = pEdge->first;
                aabbMin[0] = (aabbMin[0] < pVertex->data[0]) ? aabbMin[0]
                                                             : pVertex->data[0];
                aabbMin[1] = (aabbMin[1] < pVertex->data[1]) ? aabbMin[1]
                                                             : pVertex->data[1];
                aabbMin[2] = (aabbMin[2] < pVertex->data[2]) ? aabbMin[2]
                                                             : pVertex->data[2];
                aabbMax[0] = (aabbMax[0] > pVertex->data[0]) ? aabbMax[0]
                                                             : pVertex->data[0];
                aabbMax[1] = (aabbMax[1] > pVertex->data[1]) ? aabbMax[1]
                                                             : pVertex->data[1];
                aabbMax[2] = (aabbMax[2] > pVertex->data[2]) ? aabbMax[2]
                                                             : pVertex->data[2];
            }
        }

        Vector3f halfExtents = (aabbMax - aabbMin) * 0.5f;
        TransformAabb(halfExtents, m_fMargin, trans, aabbMin, aabbMax);
    }

    void AddFace(PointList<T> vertices, const PointPtr<T>& inner_point) {
        if (isPointAbovePlane(vertices, *inner_point)) {
            std::reverse(std::begin(vertices), std::end(vertices));
        }

        FacePtr<T> pFace = std::make_shared<Face<T>>();
        auto count = vertices.size();
        assert(count >= 3);
        for (auto i = 0; i < vertices.size(); i++) {
            pFace->Edges.push_back(std::make_shared<Edge<T>>(
                vertices[i], vertices[(i + 1) == count ? 0 : i + 1]));
        }
        assert(pFace->Edges.size() >= 3);
        CrossProduct(pFace->Normal, (*vertices[1] - *vertices[0]),
                     (*vertices[2] - *vertices[1]));
        Normalize(pFace->Normal);
        Faces.insert(std::move(pFace));
    }

    void AddTetrahedron(const PointList<T>& vertices) {
        assert(vertices.size() == 4);

        // ABC
        AddFace({vertices[0], vertices[1], vertices[2]}, vertices[3]);

        // ABD
        AddFace({vertices[0], vertices[1], vertices[3]}, vertices[2]);

        // CDB
        AddFace({vertices[2], vertices[3], vertices[1]}, vertices[0]);

        // ADC
        AddFace({vertices[0], vertices[3], vertices[2]}, vertices[1]);
    }
};
}  // namespace My
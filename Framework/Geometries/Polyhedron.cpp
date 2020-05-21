#include "Polyhedron.hpp"

using namespace My;
using namespace std;

void Polyhedron::AddFace(PointList vertices, const PointPtr& inner_point) {
    if (isPointAbovePlane(vertices, *inner_point)) {
        std::reverse(std::begin(vertices), std::end(vertices));
    }

    FacePtr pFace = std::make_shared<Face>();
    auto count = vertices.size();
    assert(count >= 3);
    for (auto i = 0; i < vertices.size(); i++) {
        pFace->Edges.push_back(std::make_shared<Edge>(
            vertices[i], vertices[(i + 1) == count ? 0 : i + 1]));
    }
    assert(pFace->Edges.size() >= 3);
    CrossProduct(pFace->Normal, (*vertices[1] - *vertices[0]),
                 (*vertices[2] - *vertices[1]));
    Normalize(pFace->Normal);
    Faces.insert(std::move(pFace));
}

void Polyhedron::AddTetrahedron(const PointList& vertices) {
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

void Polyhedron::GetAabb(const Matrix4X4f& trans, Vector3f& aabbMin,
                         Vector3f& aabbMax) const {
    aabbMin = Vector3f(numeric_limits<float>::max());
    aabbMax = Vector3f(numeric_limits<float>::lowest());

    for (const auto& pFace : Faces) {
        for (const auto& pEdge : pFace->Edges) {
            auto pVertex = pEdge->first;
            aabbMin[0] =
                (aabbMin[0] < pVertex->data[0]) ? aabbMin[0] : pVertex->data[0];
            aabbMin[1] =
                (aabbMin[1] < pVertex->data[1]) ? aabbMin[1] : pVertex->data[1];
            aabbMin[2] =
                (aabbMin[2] < pVertex->data[2]) ? aabbMin[2] : pVertex->data[2];
            aabbMax[0] =
                (aabbMax[0] > pVertex->data[0]) ? aabbMax[0] : pVertex->data[0];
            aabbMax[1] =
                (aabbMax[1] > pVertex->data[1]) ? aabbMax[1] : pVertex->data[1];
            aabbMax[2] =
                (aabbMax[2] > pVertex->data[2]) ? aabbMax[2] : pVertex->data[2];
        }
    }

    Vector3f halfExtents = (aabbMax - aabbMin) * 0.5f;
    TransformAabb(halfExtents, m_fMargin, trans, aabbMin, aabbMax);
}

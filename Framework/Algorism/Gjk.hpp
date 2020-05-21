#pragma once
#include <functional>
#include <limits>
#include <list>

#include "Polyhedron.hpp"
#include "geommath.hpp"

namespace My {
using SupportFunction = std::function<const Point(const Vector3f&)>;

inline void NearestPointInTriangleToPoint(const PointList& vertices,
                                          const Point& point, float& s,
                                          float& t) {
    assert(vertices.size() == 3);
    auto A = vertices[0];
    auto B = vertices[1];
    auto C = vertices[2];
    Vector3f edge0 = *B - *A;
    Vector3f edge1 = *C - *A;
    Vector3f v0 = *A - point;

    float a, b, c, d, e;
    DotProduct(a, edge0, edge0);
    DotProduct(b, edge0, edge1);
    DotProduct(c, edge1, edge1);
    DotProduct(d, edge0, v0);
    DotProduct(e, edge1, v0);

    float det = a * c - b * b;
    s = b * e - c * d;
    t = b * d - a * e;

    if (s + t < det) {
        if (s < 0.0f) {
            if (t < 0.0f) {
                if (d < 0.0f) {
                    s = std::clamp(-d / a, 0.0f, 1.0f);
                    t = 0.0f;
                } else {
                    s = 0.0f;
                    t = std::clamp(-e / c, 0.0f, 1.0f);
                }
            } else {
                s = 0.0f;
                t = std::clamp(-e / c, 0.0f, 1.0f);
            }
        } else if (t < 0.0f) {
            s = std::clamp(-d / a, 0.0f, 1.0f);
            t = 0.0f;
        } else {
            float invDet = 1.0f / det;
            s *= invDet;
            t *= invDet;
        }
    } else {
        if (s < 0.0f) {
            float tmp0 = b + d;
            float tmp1 = c + e;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0;
                float denom = a - 2.0f * b + c;
                s = std::clamp(numer / denom, 0.0f, 1.0f);
                t = 1.0f - s;
            } else {
                t = std::clamp(-e / c, 0.0f, 1.0f);
                s = 0.0f;
            }
        } else if (t < 0.0f) {
            if (a + d > b + e) {
                float numer = c + e - b - d;
                float denom = a - 2.0f * b + c;
                s = std::clamp(numer / denom, 0.0f, 1.0f);
                t = 1.0f - s;
            } else {
                s = std::clamp(-e / c, 0.0f, 1.0f);
                t = 0.0f;
            }
        } else {
            float numer = c + e - b - d;
            float denom = a - 2.0f * b + c;
            s = std::clamp(numer / denom, 0.0f, 1.0f);
            t = 1.0f - s;
        }
    }
}

inline int GjkIntersection(const SupportFunction& a, const SupportFunction& b,
                           Vector3f& direction, PointList& simplex) {
    Point A;
    if (simplex.empty()) {
        // initialize
        A = a(direction) - b(direction * -1.0f);
        direction = A * -1.0f;
        simplex.push_back(std::make_shared<Point>(A));
    }

    A = a(direction) - b(direction * -1.0f);

    float dot_product;
    DotProduct(dot_product, A, direction);
    // A must be further than origin on the direction
    // otherwise we know the minkowski substraction does
    // not contain the origin, thus the two shape is not
    // colliding
    if (dot_product < 0) {
        return 0;
    }

    // update the simplex
    simplex.push_back(std::make_shared<Point>(A));

    // calculate the nearest point to origin on simplex
    Point P;
    switch (simplex.size()) {
        case 0:
            // should never happen or we have wrong assumption
            assert(0);
            break;
        case 1: {
            // the simplex is a vertex.
            // simply returns AO (from the vertex to origin)
            // as next search direction
            direction = A * -1.0f;
        } break;
        case 2: {
            // the simplex is a line segment.
            // find the nearest point to the origin
            // on the line segment, naming it P
            // then update the next search direction
            // to PO
            auto A = simplex[0];
            auto B = simplex[1];
            float t;

            auto line_seg = *B - *A;
            DotProduct(t, *A * -1.0f, line_seg);
            float sqr_length;
            DotProduct(sqr_length, line_seg, line_seg);
            t = std::clamp(t / sqr_length, 0.0f, 1.0f);

            // this means P is at A which should never happen
            // because B is reported to be closer to origin than A
            assert(t > std::numeric_limits<float>::epsilon());

            if (1.0f - t < std::numeric_limits<float>::epsilon()) {
                // this means P is at B
                // remove A because it is not needed now
                P = *B;
                simplex.clear();
                simplex.push_back(B);
            } else {
                P = *A + (*B - *A) * t;
            }

            direction = P * -1.0f;
        } break;
        case 3: {
            // the simplex is a triangle.
            // find the nearest point to the origin
            // on the triangle, naming it P
            // then updte the next search direction
            // to PO
            float s, t;
            NearestPointInTriangleToPoint(simplex, Point(0.0f), s, t);

            assert(s > std::numeric_limits<float>::epsilon() ||
                   t > std::numeric_limits<float>::epsilon());

            auto A = simplex[0];
            auto B = simplex[1];
            auto C = simplex[2];
            if (s < std::numeric_limits<float>::epsilon()) {
                // P is on edge 1 (AC) so B can be removed
                simplex.clear();
                simplex = {A, C};
            }

            if (t < std::numeric_limits<float>::epsilon()) {
                // P is on edge 0 (AB) so C can be removed
                simplex.clear();
                simplex = {A, B};
            }

            if (abs(1.0f - (s + t)) < std::numeric_limits<float>::epsilon()) {
                // P is on edge 3 (BC) so A can be removed
                simplex.clear();
                simplex = {B, C};
            }

            P = *A + (*B - *A) * s + (*C - *A) * t;

            direction = P * -1.0f;
        } break;
        case 4: {
            Polyhedron tetrahedron;
            tetrahedron.AddTetrahedron(simplex);
            FacePtr pNextFace = nullptr;
            for (const auto& pFace : tetrahedron.Faces) {
                if (isPointAbovePlane(pFace, Point(0.0f))) {
                    pNextFace = pFace;
                    break;
                }
            }

            if (pNextFace == nullptr) {
                // the origin is inside the tetrahedron
                return 1;
            }

            float s, t;
            auto A = pNextFace->Edges[0]->first;
            auto B = pNextFace->Edges[1]->first;
            auto C = pNextFace->Edges[2]->first;

            PointList vertices;
            vertices.push_back(A);
            vertices.push_back(B);
            vertices.push_back(C);
            NearestPointInTriangleToPoint(vertices, Point(0.0f), s, t);

            if (s < std::numeric_limits<float>::epsilon()) {
                // P is on edge 1 (AC) so B can be removed
                simplex.clear();
                simplex = {A, C};
            }

            if (t < std::numeric_limits<float>::epsilon()) {
                // P is on edge 0 (AB) so C can be removed
                simplex.clear();
                simplex = {A, B};
            }

            if (abs(1.0f - (s + t)) < std::numeric_limits<float>::epsilon()) {
                // P is on edge 3 (BC) so A can be removed
                simplex.clear();
                simplex = {B, C};
            }

            if (simplex.size() == 4) {
                simplex.clear();
                simplex = {A, B, C};
            }

            P = *A + (*B - *A) * s + (*C - *A) * t;

            direction = P * -1.0f;
        } break;
        default:
            // should never happen in a R^3 space
            assert(0);
    }

    // check if we are at origin
    if (Length(direction) < std::numeric_limits<float>::epsilon()) {
        return 1;
    }

    return -1;
}

inline Point ConvexPolyhedronSupportFunction(const Polyhedron& polyhedron,
                                             const Vector3f& direction) {
    float max_score = std::numeric_limits<float>::lowest();
    PointPtr extreme_point;
    for (const auto& pFace : polyhedron.Faces) {
        float score;
        DotProduct(score, pFace->Normal, direction);
        if (score <= 0.0f) {
            // back-facing face, ignore it
            continue;
        }

        for (const auto& pEdge : pFace->Edges) {
            DotProduct(score, *pEdge->first, direction);
            if (score > max_score) {
                max_score = score;
                extreme_point = pEdge->first;
            }

            assert(extreme_point != nullptr);
        }
    }

    return *extreme_point;
}
}  // namespace My
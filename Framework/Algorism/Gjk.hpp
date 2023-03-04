#pragma once
#include <functional>
#include <limits>
#include <list>

#include "Polyhedron.hpp"
#include "geommath.hpp"

namespace My {
template <typename T>
using SupportFunction = std::function<const Point<T>(const Vector3<T>&)>;

template <typename T>
inline void NearestPointInTriangleToPoint(const PointList<T>& vertices,
                                          const Point<T>& point, T& s,
                                          T& t) {
    assert(vertices.size() == 3);
    auto A = vertices[0];
    auto B = vertices[1];
    auto C = vertices[2];
    Vector3<T> edge0 = *B - *A;
    Vector3<T> edge1 = *C - *A;
    Vector3<T> v0 = *A - point;

    T a, b, c, d, e;
    DotProduct(a, edge0, edge0);
    DotProduct(b, edge0, edge1);
    DotProduct(c, edge1, edge1);
    DotProduct(d, edge0, v0);
    DotProduct(e, edge1, v0);

    auto det = a * c - b * b;
    s = b * e - c * d;
    t = b * d - a * e;

    if (s + t < det) {
        if (s < 0.0) {
            if (t < 0.0) {
                if (d < 0.0) {
                    s = std::clamp(-d / a, static_cast<T>(0.0), static_cast<T>(1.0));
                    t = 0.0;
                } else {
                    s = 0.0;
                    t = std::clamp(-e / c, static_cast<T>(0.0), static_cast<T>(1.0));
                }
            } else {
                s = 0.0;
                t = std::clamp(-e / c, static_cast<T>(0.0), static_cast<T>(1.0));
            }
        } else if (t < 0.0) {
            s = std::clamp(-d / a, static_cast<T>(0.0), static_cast<T>(1.0));
            t = 0.0;
        } else {
            auto invDet = 1.0 / det;
            s *= invDet;
            t *= invDet;
        }
    } else {
        if (s < 0.0) {
            auto tmp0 = b + d;
            auto tmp1 = c + e;
            if (tmp1 > tmp0) {
                auto numer = tmp1 - tmp0;
                auto denom = a - 2.0 * b + c;
                s = std::clamp(numer / denom, 0.0, 1.0);
                t = 1.0 - s;
            } else {
                t = std::clamp(-e / c, static_cast<T>(0.0), static_cast<T>(1.0));
                s = 0.0;
            }
        } else if (t < 0.0) {
            if (a + d > b + e) {
                auto numer = c + e - b - d;
                auto denom = a - 2.0 * b + c;
                s = std::clamp(numer / denom, 0.0, 1.0);
                t = 1.0 - s;
            } else {
                s = std::clamp(-e / c, static_cast<T>(0.0), static_cast<T>(1.0));
                t = 0.0;
            }
        } else {
            auto numer = c + e - b - d;
            auto denom = a - 2.0 * b + c;
            s = std::clamp(numer / denom, 0.0, 1.0);
            t = 1.0 - s;
        }
    }
};

template <typename T>
inline int GjkIntersection(const SupportFunction<T>& a, const SupportFunction<T>& b,
                           Vector3<T>& direction, PointList<T>& simplex) {
    Point<T> A;
    if (simplex.empty()) {
        // initialize
        A = a(direction) - b(direction * (T)-1.0);
        direction = A * (T)-1.0;
        simplex.push_back(std::make_shared<Point<T>>(A));
    }

    A = a(direction) - b(direction * (T)-1.0);

    T dot_product;
    DotProduct(dot_product, A, direction);
    // A must be further than origin on the direction
    // otherwise we know the minkowski substraction does
    // not contain the origin, thus the two shape is not
    // colliding
    if (dot_product < 0) {
        return 0;
    }

    // update the simplex
    simplex.push_back(std::make_shared<Point<T>>(A));

    // calculate the nearest point to origin on simplex
    Point<T> P;
    switch (simplex.size()) {
        case 0:
            // should never happen or we have wrong assumption
            assert(0);
            break;
        case 1: {
            // the simplex is a vertex.
            // simply returns AO (from the vertex to origin)
            // as next search direction
            direction = A * (T)-1.0;
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
            DotProduct(t, *A * (T)-1.0, line_seg);
            T sqr_length;
            DotProduct(sqr_length, line_seg, line_seg);
            t = std::clamp(t / sqr_length, static_cast<T>(0.0), static_cast<T>(1.0));

            // this means P is at A which should never happen
            // because B is reported to be closer to origin than A
            assert(t > std::numeric_limits<T>::epsilon());

            if (1.0 - t < std::numeric_limits<T>::epsilon()) {
                // this means P is at B
                // remove A because it is not needed now
                P = *B;
                simplex.clear();
                simplex.push_back(B);
            } else {
                P = *A + (*B - *A) * t;
            }

            direction = P * (T)-1.0;
        } break;
        case 3: {
            // the simplex is a triangle.
            // find the nearest point to the origin
            // on the triangle, naming it P
            // then updte the next search direction
            // to PO
            float s, t;
            NearestPointInTriangleToPoint(simplex, Point<T>(0.0), s, t);

            assert(s > std::numeric_limits<T>::epsilon() ||
                   t > std::numeric_limits<T>::epsilon());

            auto A = simplex[0];
            auto B = simplex[1];
            auto C = simplex[2];
            if (s < std::numeric_limits<T>::epsilon()) {
                // P is on edge 1 (AC) so B can be removed
                simplex.clear();
                simplex = {A, C};
            }

            if (t < std::numeric_limits<T>::epsilon()) {
                // P is on edge 0 (AB) so C can be removed
                simplex.clear();
                simplex = {A, B};
            }

            if (std::abs(1.0 - (s + t)) < std::numeric_limits<T>::epsilon()) {
                // P is on edge 3 (BC) so A can be removed
                simplex.clear();
                simplex = {B, C};
            }

            P = *A + (*B - *A) * s + (*C - *A) * t;

            direction = P * (T)-1.0;
        } break;
        case 4: {
            Polyhedron<T> tetrahedron;
            tetrahedron.AddTetrahedron(simplex);
            FacePtr<T> pNextFace = nullptr;
            for (const auto& pFace : tetrahedron.Faces) {
                if (isPointAbovePlane(pFace, Point<T>(0.0))) {
                    pNextFace = pFace;
                    break;
                }
            }

            if (pNextFace == nullptr) {
                // the origin is inside the tetrahedron
                return 1;
            }

            T s, t;
            auto A = pNextFace->Edges[0]->first;
            auto B = pNextFace->Edges[1]->first;
            auto C = pNextFace->Edges[2]->first;

            PointList<T> vertices;
            vertices.push_back(A);
            vertices.push_back(B);
            vertices.push_back(C);
            NearestPointInTriangleToPoint(vertices, Point<T>(0.0), s, t);

            if (s < std::numeric_limits<T>::epsilon()) {
                // P is on edge 1 (AC) so B can be removed
                simplex.clear();
                simplex = {A, C};
            }

            if (t < std::numeric_limits<T>::epsilon()) {
                // P is on edge 0 (AB) so C can be removed
                simplex.clear();
                simplex = {A, B};
            }

            if (std::abs(1.0 - (s + t)) < std::numeric_limits<T>::epsilon()) {
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
    if (Length(direction) < std::numeric_limits<T>::epsilon()) {
        return 1;
    }

    return -1;
}

template <typename T>
inline Point<T> ConvexPolyhedronSupportFunction(const Polyhedron<T>& polyhedron,
                                             const Vector3<T>& direction) {
    T max_score = std::numeric_limits<T>::lowest();
    PointPtr<T> extreme_point;
    for (const auto& pFace : polyhedron.Faces) {
        T score;
        DotProduct(score, pFace->Normal, direction);
        if (score <= 0.0) {
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
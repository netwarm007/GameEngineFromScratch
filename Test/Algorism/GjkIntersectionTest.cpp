#include <array>
#include <cmath>
#include <iostream>

#include "ConvexHull.hpp"
#include "Gjk.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

namespace {
// Support function of an axis-aligned cube with half extent 1 at `center`.
SupportFunction<float> CubeSupport(Vector3f center) {
    return [center](const Vector3f& d) {
        return Point<float>({center[0] + (d[0] >= 0 ? 1.0f : -1.0f),
                             center[1] + (d[1] >= 0 ? 1.0f : -1.0f),
                             center[2] + (d[2] >= 0 ? 1.0f : -1.0f)});
    };
}

int Run(const SupportFunction<float>& a, const SupportFunction<float>& b) {
    PointList<float> simplex;
    Vector3f direction({1.0f, 0.0f, 0.0f});
    int result;
    int guard = 0;
    while ((result = GjkIntersection(a, b, direction, simplex)) == -1) {
        if (++guard > 1000) return -2;
    }
    return result;
}

bool ClosestPointMatches(const PointList<float>& triangle, const Point<float>& point,
                         float expected_s, float expected_t) {
    float s, t;
    NearestPointInTriangleToPoint(triangle, point, s, t);
    return std::abs(s - expected_s) < 1e-5f &&
           std::abs(t - expected_t) < 1e-5f;
}
}  // namespace

int main() {
    {
        ConvexHull<float> hull;
        const std::array<std::array<float, 3>, 4> vertices = {{
            {{0, 0, 0}}, {{10, 10, 10}}, {{2, 3, 4}}, {{7, 4, 5}},
        }};
        for (const auto& vertex : vertices) {
            hull.AddPoint(std::make_shared<Point<float>>(
                Point<float>({vertex[0], vertex[1], vertex[2]})));
        }
        while (hull.Iterate()) {
        }
        CHECK(hull.Faces.size() == 4);
        const auto support = ConvexPolyhedronSupportFunction(hull, Vector3f({0, 0, 0}));
        bool is_vertex = false;
        for (const auto& vertex : vertices) {
            if (support[0] == vertex[0] && support[1] == vertex[1] &&
                support[2] == vertex[2]) {
                is_vertex = true;
            }
        }
        CHECK(is_vertex);
    }

    {
        bool rejected = false;
        try {
            ConvexPolyhedronSupportFunction(Polyhedron<float>{}, Vector3f({1, 0, 0}));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CHECK(rejected);
    }

    for (const float side : {-1.0f, 1.0f}) {
        const std::array<std::array<float, 3>, 4> tetrahedron = {{
            {{0, 0, 0}}, {{10, 0, 0}}, {{3, 3, 0}}, {{4, 1, side}},
        }};
        ConvexHull<float> hull;
        for (const auto& vertex : tetrahedron) {
            hull.AddPoint(std::make_shared<Point<float>>(
                Point<float>({vertex[0], vertex[1], vertex[2]})));
        }
        while (hull.Iterate()) {
        }
        CHECK(hull.Faces.size() == 4);
        const auto extreme =
            ConvexPolyhedronSupportFunction(hull, Vector3f({0, 0, side}));
        CHECK(extreme[0] == 4 && extreme[1] == 1 && extreme[2] == side);
    }

    PointList<float> triangle = {
        std::make_shared<Point<float>>(Point<float>({0, 0, 0})),
        std::make_shared<Point<float>>(Point<float>({1, 0, 0})),
        std::make_shared<Point<float>>(Point<float>({0, 1, 0})),
    };
    CHECK(ClosestPointMatches(triangle, Point<float>({0.25f, 0.25f, 1}), 0.25f, 0.25f));
    CHECK(ClosestPointMatches(triangle, Point<float>({-1, -1, 0}), 0, 0));
    CHECK(ClosestPointMatches(triangle, Point<float>({0.4f, -0.5f, 0}), 0.4f, 0));
    CHECK(ClosestPointMatches(triangle, Point<float>({-0.5f, 0.4f, 0}), 0, 0.4f));
    CHECK(ClosestPointMatches(triangle, Point<float>({0.8f, 0.8f, 0}), 0.5f, 0.5f));
    CHECK(ClosestPointMatches(triangle, Point<float>({2, -1, 0}), 1, 0));
    CHECK(ClosestPointMatches(triangle, Point<float>({-1, 2, 0}), 0, 1));

    auto origin = CubeSupport(Vector3f({0, 0, 0}));

    CHECK(Run(origin, CubeSupport(Vector3f({0.5f, 0.3f, -0.4f}))) == 1);
    CHECK(Run(origin, CubeSupport(Vector3f({5.0f, 0.0f, 0.0f}))) == 0);
    CHECK(Run(origin, CubeSupport(Vector3f({0.0f, -4.0f, 3.0f}))) == 0);

    // Support function of a convex hull returns its extreme vertex.
    ConvexHull<float> hull;
    const float v[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                           {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    for (auto& p : v) {
        hull.AddPoint(std::make_shared<Point<float>>(Point<float>({p[0], p[1], p[2]})));
    }
    while (hull.Iterate())
        ;
    Polyhedron<float> poly = hull.GetHull();
    auto top = ConvexPolyhedronSupportFunction(poly, Vector3f({0.1f, 0.2f, 1.0f}));
    CHECK(top[2] == 1.0f);
    CHECK(top[0] == 1.0f && top[1] == 1.0f);

    return 0;
}

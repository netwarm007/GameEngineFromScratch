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
}  // namespace

int main() {
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

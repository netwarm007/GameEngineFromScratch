#include <iostream>

#include "AaBb.hpp"
#include "Color.hpp"

using namespace My;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                  \
        }                                                              \
    } while (0)

int main() {
    AaBb<float, 3> box(Vector3f({-1.0f, -1.0f, -1.0f}),
                       Vector3f({1.0f, 1.0f, 1.0f}));

    CHECK(box.min_point()[0] == -1.0f && box.max_point()[2] == 1.0f);

    // Rays pointing at the box from both sides hit; pointing away or
    // passing beside it miss.
    CHECK(box.Intersect(Ray<float>(Vector3f({-5, 0, 0}), Vector3f({1, 0, 0})), 0.0f, 100.0f));
    CHECK(box.Intersect(Ray<float>(Vector3f({5, 0, 0}), Vector3f({-1, 0, 0})), 0.0f, 100.0f));
    CHECK(!box.Intersect(Ray<float>(Vector3f({-5, 0, 0}), Vector3f({-1, 0, 0})), 0.0f, 100.0f));
    CHECK(!box.Intersect(Ray<float>(Vector3f({-5, 3, 0}), Vector3f({1, 0, 0})), 0.0f, 100.0f));
    CHECK(!box.Intersect(Ray<float>(Vector3f({-5, 0, 0}), Vector3f({1, 0, 0})), 0.0f, 2.0f));

    AaBb<float, 3> other(Vector3f({0.0f, -2.0f, 0.5f}),
                         Vector3f({3.0f, 0.0f, 4.0f}));
    auto merged = SurroundingBox(box, other);
    CHECK(merged.min_point()[0] == -1.0f && merged.min_point()[1] == -2.0f &&
          merged.min_point()[2] == -1.0f);
    CHECK(merged.max_point()[0] == 3.0f && merged.max_point()[1] == 1.0f &&
          merged.max_point()[2] == 4.0f);

    // Identity transform with a margin grows the box symmetrically.
    Matrix4X4f identity;
    BuildIdentityMatrix(identity);
    AaBb<float, 3> out;
    TransformAabb(Vector3f({1.0f, 2.0f, 3.0f}), 0.5f, identity, out);
    CHECK(out.min_point()[0] == -1.5f && out.max_point()[1] == 2.5f &&
          out.max_point()[2] == 3.5f);

    return 0;
}

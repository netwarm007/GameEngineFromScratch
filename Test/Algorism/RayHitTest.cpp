#include <cmath>
#include <iostream>

#include "Color.hpp"
#include "Hit.hpp"
#include "Ray.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main() {
    // The ray constructor normalizes its direction.
    Ray<float> ray(Vector3f({1, 2, 3}), Vector3f({0, 0, 10}));
    CHECK(std::fabs(ray.getDirection()[2] - 1.0f) < 1e-6f);
    auto p = ray.pointAtParameter(4.0f);
    CHECK(p[0] == 1.0f && p[1] == 2.0f && p[2] == 7.0f);

    Ray<float> copy(ray);
    CHECK(copy.getOrigin()[1] == 2.0f);

    Hit<float> hit;
    CHECK(hit.getT() > 1e30f);
    int mat = 0;
    hit.set(2.5f, Point<float>({1, 1, 1}), Vector3f({0, 1, 0}), true, &mat);
    CHECK(hit.getT() == 2.5f && hit.isFrontFace());
    CHECK(hit.getNormal()[1] == 1.0f && hit.getP()[0] == 1.0f);
    CHECK(hit.getMaterial() == &mat);

    // Quantization clamps out-of-range channels.
    auto c = QuantizeUnsigned8Bits(RGBf({-1.0f, 0.5f, 2.0f}));
    CHECK(c[0] == 0 && c[1] == 127 && c[2] == 255);

    return 0;
}

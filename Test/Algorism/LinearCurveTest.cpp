#include <cmath>
#include <iostream>

#include "Linear.hpp"

using namespace My;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAILED: " #cond " at line " << __LINE__ << "\n"; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static bool Near(float a, float b) { return std::fabs(a - b) < 1e-3f; }

int main() {
    // Scalar curve over knots {0, 2, 6}.
    Linear<float, float> curve(std::vector<float>{0.0f, 2.0f, 6.0f});
    CHECK(curve.GetCurveType() == CurveType::kLinear);

    size_t index = 99;
    CHECK(curve.Reverse(-1.0f, index) == 0.0f && index == 0);
    CHECK(curve.Reverse(10.0f, index) == 1.0f && index == 3);
    CHECK(Near(curve.Reverse(1.0f, index), 0.5f) && index == 1);
    CHECK(Near(curve.Reverse(4.0f, index), 0.5f) && index == 2);

    CHECK(Near(curve.Interpolate(0.5f, 1), 1.0f));
    CHECK(Near(curve.Interpolate(0.5f, 2), 4.0f));
    CHECK(curve.Interpolate(0.3f, 0) == 0.0f);
    CHECK(curve.Interpolate(0.3f, 7) == 6.0f);

    Linear<float, float> empty;
    CHECK(empty.Interpolate(0.5f, 1) == 0.0f);
    CHECK(empty.Reverse(1.0f, index) == 0.0f);
    empty.AddKnot(5.0f);
    CHECK(empty.Interpolate(0.5f, 1) == 5.0f);

    const float raw[] = {1.0f, 3.0f};
    Linear<float, float> from_array(raw, 2);
    CHECK(Near(from_array.Interpolate(0.5f, 1), 2.0f));

    // Matrix curve interpolates decomposed translation and scale.
    Matrix4X4f m1, m2;
    Matrix4X4fCompose(m1, Vector3f({0, 0, 0}), Vector3f({1, 1, 1}),
                      Vector3f({0, 0, 0}));
    Matrix4X4fCompose(m2, Vector3f({0, 0, 0}), Vector3f({3, 3, 3}),
                      Vector3f({10, 0, 0}));
    Linear<Matrix4X4f, float> mcurve(std::vector<Matrix4X4f>{m1, m2});

    auto mid = mcurve.Interpolate(0.5f, 1);
    Vector3f rot, scale, trans;
    Matrix4X4fDecompose(mid, rot, scale, trans);
    CHECK(Near(trans[0], 5.0f) && Near(scale[0], 2.0f));

    auto first = mcurve.Interpolate(0.5f, 0);
    Matrix4X4fDecompose(first, rot, scale, trans);
    CHECK(Near(trans[0], 0.0f));
    auto last = mcurve.Interpolate(0.5f, 9);
    Matrix4X4fDecompose(last, rot, scale, trans);
    CHECK(Near(trans[0], 10.0f));

    Linear<Matrix4X4f, float> mempty;
    auto identity = mempty.Interpolate(0.5f, 1);
    CHECK(identity[0][0] == 1.0f && identity[3][3] == 1.0f &&
          identity[0][3] == 0.0f);
    mempty.AddKnot(m2);
    Matrix4X4fDecompose(mempty.Interpolate(0.5f, 1), rot, scale, trans);
    CHECK(Near(trans[0], 10.0f));

    return 0;
}

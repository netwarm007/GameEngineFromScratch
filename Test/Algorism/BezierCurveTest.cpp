#include <cmath>
#include <iostream>

#include "Bezier.hpp"

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
    // Control points at thirds of each segment make the curve a straight line,
    // so Reverse/Interpolate have closed-form expected values.
    const float third = 1.0f / 3.0f;
    Bezier<float, float> curve(
        std::vector<float>{0.0f, 1.0f, 3.0f},
        std::vector<float>{0.0f, 1.0f - third, 3.0f - 2.0f * third},
        std::vector<float>{third, 1.0f + 2.0f * third, 3.0f});
    CHECK(curve.GetCurveType() == CurveType::kBezier);

    size_t index = 99;
    CHECK(curve.Reverse(-1.0f, index) == 0.0f && index == 0);
    CHECK(curve.Reverse(5.0f, index) == 1.0f && index == 3);

    CHECK(Near(curve.Reverse(0.5f, index), 0.5f) && index == 1);
    CHECK(Near(curve.Reverse(2.0f, index), 0.5f) && index == 2);

    CHECK(Near(curve.Interpolate(0.5f, 1), 0.5f));
    CHECK(Near(curve.Interpolate(0.25f, 2), 1.5f));
    CHECK(curve.Interpolate(0.5f, 0) == 0.0f);
    CHECK(curve.Interpolate(0.5f, 9) == 3.0f);

    // Reverse then Interpolate on the same curve recovers the input value.
    for (float t : {0.1f, 0.9f, 1.5f, 2.9f}) {
        auto s = curve.Reverse(t, index);
        CHECK(Near(curve.Interpolate(s, index), t));
    }

    // Constructing from raw arrays gives the same curve.
    const float knots[] = {0.0f, 2.0f};
    const float in_cp[] = {0.0f, 2.0f - 2.0f / 3.0f};
    const float out_cp[] = {2.0f / 3.0f, 2.0f};
    Bezier<float, float> from_array(knots, in_cp, out_cp, 2);
    CHECK(Near(from_array.Interpolate(0.5f, 1), 1.0f));

    Bezier<float, float> empty;
    CHECK(empty.Interpolate(0.5f, 1) == 0.0f);
    CHECK(empty.Reverse(1.0f, index) == 0.0f);
    empty.AddKnot(4.0f);
    CHECK(empty.Interpolate(0.5f, 1) == 4.0f);

    return 0;
}

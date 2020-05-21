#include <cassert>
#include <iostream>
#include <memory>

#include "geommath.hpp"

using namespace std;

namespace My {
Point2DList BresenhamLineInterpolate(const Point2D& start, const Point2D& end) {
    Point2DList result;

    auto delta_x = end[0] - start[0];
    auto delta_y = end[1] - start[1];
    assert(delta_x != 0);
    auto delta_err =
        abs(delta_y / delta_x);  // Assume deltax != 0 (line is not vertical),
                                 // note that this division needs to be done in
                                 // a way that preserves the fractional part
    decltype(delta_err) error = 0;  // No error at start
    int32_t y = static_cast<int32_t>(start[1]);
    for (int32_t x = static_cast<int32_t>(start[0]);
         x <= static_cast<int32_t>(end[0]); x++) {
        result.push_back(make_shared<Point2D>(Point2D({(float)x, (float)y})));
        error += delta_err;
        while (error >= 0.5) {
            y += (delta_y > 0 ? 1 : -1) * 1;
            error -= 1.0;
        }
    }

    return result;
}
}  // namespace My
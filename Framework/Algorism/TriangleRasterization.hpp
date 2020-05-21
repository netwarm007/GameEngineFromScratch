#include <algorithm>
#include <cassert>
#include <iostream>

#include "geommath.hpp"

using namespace std;
using namespace My;

namespace My {
Point2DList BottomFlatTriangleInterpolation(const Point2D& v1,
                                            const Point2D& v2,
                                            const Point2D& v3) {
    Point2DList result;

    assert(v2[1] == v3[1]);  // the bottom must be flat
    assert(v1[1] < v2[1]);   // v1 must be the top vertex

    auto invslope1 = (v2[0] - v1[0]) / (v2[1] - v1[1]);
    auto invslope2 = (v3[0] - v1[0]) / (v3[1] - v1[1]);

    auto start_x = v1[0];
    auto end_x = v1[0];

    for (int32_t scanline_y = (int32_t)round(v1[1]);
         scanline_y < (int32_t)round(v2[1]); scanline_y++) {
        for (int32_t scanline_x = (int32_t)round(start_x);
             scanline_x <= (int32_t)round(end_x); scanline_x++) {
            result.push_back(make_shared<Point2D>(
                Point2D({(float)scanline_x, (float)scanline_y})));
        }
        start_x += invslope1;
        end_x += invslope2;
    }

    return result;
}

Point2DList TopFlatTriangleInterpolation(const Point2D& v1, const Point2D& v2,
                                         const Point2D& v3) {
    Point2DList result;

    assert(v1[1] == v2[1]);  // the top must be flat
    assert(v1[1] < v3[1]);   // v3 must be the bottom vertex

    auto invslope1 = (v1[0] - v3[0]) / (v1[1] - v3[1]);
    auto invslope2 = (v2[0] - v3[0]) / (v2[1] - v3[1]);

    auto start_x = v1[0];
    auto end_x = v2[0];

    for (int32_t scanline_y = (int32_t)round(v1[1]);
         scanline_y <= (int32_t)round(v3[1]); scanline_y++) {
        for (int32_t scanline_x = (int32_t)round(start_x);
             scanline_x <= (int32_t)round(end_x); scanline_x++) {
            result.push_back(make_shared<Point2D>(
                Point2D({(float)scanline_x, (float)scanline_y})));
        }
        start_x += invslope1;
        end_x += invslope2;
    }

    return result;
}

void SortTriangleVerticesAccording2YAscend(Point2DPtr& pV1, Point2DPtr& pV2,
                                           Point2DPtr& pV3) {
    if (pV2->data[1] < pV1->data[1]) {
        swap(pV1, pV2);
    }

    if (pV3->data[1] < pV1->data[1]) {
        swap(pV1, pV3);
    }

    if (pV3->data[1] < pV2->data[1]) {
        swap(pV2, pV3);
    }
}

Point2DList StandardTriangleInterpolation(const Point2D& v1, const Point2D& v2,
                                          const Point2D& v3) {
    Point2DList result;
    Point2DPtr pV1 = make_shared<Point2D>(v1);
    Point2DPtr pV2 = make_shared<Point2D>(v2);
    Point2DPtr pV3 = make_shared<Point2D>(v3);
    SortTriangleVerticesAccording2YAscend(pV1, pV2, pV3);

    // now v1.y <= v2.y <= v3.y
    if (pV1->data[1] == pV2->data[1]) {
        result = TopFlatTriangleInterpolation(*pV1, *pV2, *pV3);
    } else if (pV2->data[1] == pV3->data[1]) {
        result = BottomFlatTriangleInterpolation(*pV1, *pV2, *pV3);
    } else {
        Point2D v4 = *pV2;
        v4[0] = pV1->data[0] + ((pV2->data[1] - pV1->data[1]) /
                                (pV3->data[1] - pV1->data[1])) *
                                   (pV3->data[0] - pV1->data[0]);

        Point2DPtr pV4 = make_shared<Point2D>(v4);

        if (pV4->data[0] < pV2->data[0]) {
            swap(pV2, pV4);
        }

        auto result1 = BottomFlatTriangleInterpolation(*pV1, *pV2, *pV4);
        auto result2 = TopFlatTriangleInterpolation(*pV2, *pV4, *pV3);

        result.reserve(result1.size() + result2.size());
        result.insert(result.end(), result1.begin(), result1.end());
        result.insert(result.end(), result2.begin(), result2.end());
    }

    return result;
}

Point2DList BaryCentricTriangleInterpolation(const Point2D& v1,
                                             const Point2D& v2,
                                             const Point2D& v3) {
    Point2DList result;

    auto minX = min({v1[0], v2[0], v3[0]});
    auto maxX = max({v1[0], v2[0], v3[0]});
    auto minY = min({v1[1], v2[1], v3[1]});
    auto maxY = max({v1[1], v2[1], v3[1]});

    Vector2f edge_1_2 = v2 - v1;
    Vector2f edge_1_3 = v3 - v1;

    for (int32_t col = (int32_t)round(minY); col <= (int32_t)round(maxY);
         col++) {
        for (int32_t row = (int32_t)round(minX); row <= (int32_t)round(maxX);
             row++) {
            Vector2f p = {(float)row, (float)col};
            p = p - v1;

            float area1, area2;
            CrossProduct(area1, p, edge_1_2);
            CrossProduct(area2, edge_1_3, edge_1_2);
            auto s = area1 / area2;

            CrossProduct(area1, edge_1_3, p);
            auto t = area1 / area2;

            if (s >= 0.0f && t >= 0.0f && s + t <= 1.0f) {
                // the point is inside the triangle
                result.push_back(make_shared<Point2D>(p + v1));
            }
        }
    }

    return result;
}
}  // namespace My
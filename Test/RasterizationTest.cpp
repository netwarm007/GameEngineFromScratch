#include <string>

#include "Bresenham.hpp"
#include "TriangleRasterization.hpp"

using namespace My;
using namespace std;

template <class T>
void visualize(T points, const string& name) {
    cout << name << ':' << endl;

    // dump the result
    for (auto point : points) {
        cout << *point;
    }

    // visualize in the console
    // note y is fliped towarding downside for easy print
    int row = 0, col = 0;
    for (auto point : points) {
        while (col < point->data[1]) {
            col++;
            cout << endl;
            row = 0;
        }

        while (row++ < point->data[0]) {
            cout << ' ';
        }

        cout << '*';
    }

    cout << endl;
}

int main(int argc, char** argv) {
    // raster a line
    Point2D start_point({0, 0});
    Point2D end_point({11, 4});
    auto points = BresenhamLineInterpolate(start_point, end_point);

    visualize(points, "Line");

    // raster a bottom flat triangle
    Point2D v1 = {5.0f, 0.0f};
    Point2D v2 = {0.0f, 8.0f};
    Point2D v3 = {21.0f, 8.0f};

    points = BottomFlatTriangleInterpolation(v1, v2, v3);

    visualize(points, "Bottom Flat Triangle");

    // raster a top flat triangle
    v1 = {0.0f, 0.0f};
    v2 = {21.0f, 0.0f};
    v3 = {12.0f, 8.0f};

    points = TopFlatTriangleInterpolation(v1, v2, v3);

    visualize(points, "Top Flat Triangle");

    // raster a normal triangle
    v1 = {1.0f, 0.0f};
    v2 = {16.0f, 9.0f};
    v3 = {30.0f, 4.0f};

    points = StandardTriangleInterpolation(v1, v2, v3);

    visualize(points, "General Triangle Standard Rasterization");

    points = BaryCentricTriangleInterpolation(v1, v2, v3);

    visualize(points, "General Triangle Barycentric Rasterization");

    return 0;
}
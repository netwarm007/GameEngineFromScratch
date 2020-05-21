#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "quickhull.hpp"

using namespace std;
using namespace My;

int main(int argc, char** argv) {
    int point_num = 30;

    if (argc > 1) {
        point_num = atoi(argv[1]);
    }

    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    auto dice = std::bind(distribution, generator);

    QuickHull quick_hull;
    PointSet point_set;
    cout << "Points Generated:" << endl;
    for (auto i = 0; i < point_num; i++) {
        PointPtr point_ptr = make_shared<Point>(Point{dice(), dice(), dice()});
        cout << *point_ptr;
        point_set.insert(std::move(point_ptr));
    }

    Polyhedron convex_hull;
    while (quick_hull.Iterate(convex_hull, point_set)) {
        cerr << "num of faces after this iteration: "
             << convex_hull.Faces.size() << endl;
    }

    cerr << "num of faces generated: " << convex_hull.Faces.size() << endl;

    return 0;
}
